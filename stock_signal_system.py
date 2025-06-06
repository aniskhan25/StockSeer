from curl_cffi import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
import json
from pathlib import Path
import ta  # Technical analysis library for additional indicators
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumePriceTrendIndicator

# Set up enhanced logging
log_file = 'trading_signal.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StockSignalSystem:
    def __init__(self, tickers, ema_short=10, ema_long=20, min_volume=20_000_000, 
                 email_notifications=False, email_settings=None, 
                 data_resolution='1d', data_period=100):
        if ema_short >= ema_long:
            raise ValueError("ema_short must be less than ema_long")
        if not isinstance(tickers, list) or not tickers:
            raise ValueError("tickers must be a non-empty list")
            
        self.tickers = tickers
        self.ema_short = ema_short
        self.ema_long = ema_long
        self.min_volume = min_volume
        self.stock_data = {}
        self.signals = []
        self.email_notifications = email_notifications
        self.email_settings = email_settings
        self.data_resolution = data_resolution
        self.data_period = data_period
        self.cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
        self.resistance_levels = {}
        self.support_levels = {}
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        # Validate email settings if enabled
        if email_notifications and not self._validate_email_settings():
            raise ValueError("Invalid email settings provided")

    def _validate_email_settings(self):
        """Validate email configuration settings."""
        required_keys = ['smtp_server', 'smtp_port', 'username', 'password', 'from_email', 'to_email']
        return (self.email_settings is not None and 
                all(key in self.email_settings for key in required_keys))

    def _get_cache_filename(self, ticker):
        """Generate a cache filename for a ticker and resolution."""
        return os.path.join(self.cache_dir, f"{ticker}_{self.data_resolution}.pkl")
    
    def _load_from_cache(self, ticker):
        """Load stock data from cache if available and not too old."""
        cache_file = self._get_cache_filename(ticker)
        
        if os.path.exists(cache_file):
            try:
                # Load cached data
                cached_data = pd.read_pickle(cache_file)
                
                # Check if cache is recent enough
                now = datetime.now()
                last_date = cached_data.index[-2]
                
                # Convert timezone-naive datetime to timezone-aware
                if last_date.tzinfo is not None and now.tzinfo is None:
                    now = now.replace(tzinfo=last_date.tzinfo)
                elif last_date.tzinfo is None and now.tzinfo is not None:
                    last_date = last_date.replace(tzinfo=now.tzinfo)
                
                cache_age = now - last_date

                max_age_days = 1 if self.data_resolution == '1d' else 1/6  # 4 hours for intraday data
                
                resolution_seconds = 86400 if self.data_resolution == '1d' else \
                                   14400 if self.data_resolution == '4h' else \
                                   int(self.data_resolution[:-1]) * 60
                if cache_age.total_seconds() < (max_age_days * 86400 + resolution_seconds):
                    logging.info(f"Using cached data for {ticker} ({self.data_resolution})")
                    return cached_data
                else:
                    logging.info(f"Cached data for {ticker} is too old ({cache_age.total_seconds()/3600:.1f} hours), refreshing")
            except Exception as e:
                logging.error(f"Error loading cache for {ticker}: {e}")
                
        return None
    
    def _save_to_cache(self, ticker, data):
        """Save stock data to cache."""
        if data is None or data.empty:
            return
            
        cache_file = self._get_cache_filename(ticker)
        try:
            data.to_pickle(cache_file)
            logging.info(f"Saved {ticker} data to cache")
        except Exception as e:
            logging.error(f"Error saving cache for {ticker}: {e}")
    
    def fetch_data(self):
        session = requests.Session(impersonate="chrome")

        # We'll use different date ranges based on resolution
        end_date = datetime.now(pytz.UTC)
        
        # Determine appropriate intervals based on resolution
        interval_lookup = {
            '1d': {'interval': '1d', 'days': self.data_period},
            '4h': {'interval': '60m', 'days': max(30, self.data_period // 6)},
            '1h': {'interval': '60m', 'days': max(7, self.data_period // 24)},
            '30m': {'interval': '30m', 'days': max(7, self.data_period // 48)},
            '15m': {'interval': '15m', 'days': max(7, self.data_period // 96)}
        }
        
        # Use default 1d if specified resolution is not supported
        interval_settings = interval_lookup.get(self.data_resolution, interval_lookup['1d'])
        interval = interval_settings['interval']
        days = interval_settings['days']
        
        start_date = end_date - timedelta(days=days)
        
        for ticker in self.tickers:
            try:
                # First try to load from cache
                cached_data = self._load_from_cache(ticker)
                
                if cached_data is not None:
                    # We have valid cached data
                    self.stock_data[ticker] = cached_data
                    continue
                 
                # No valid cache, download fresh data
                logging.info(f"Downloading {ticker} data from {start_date} to {end_date} with {interval} resolution")
                yfticker = yf.Ticker(ticker, session=session)
                # Use date strings without time component to avoid timezone issues
                data = yfticker.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
                
                # Make sure the index doesn't have timezone info to match with datetime.now()
                if not data.empty and data.index.tzinfo is not None:
                    data.index = data.index.tz_localize(None)

                # data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
                
                # Add debugging information for diagnostics
                if not data.empty:
                    logging.info(f"Downloaded data for {ticker}: {len(data)} rows, columns: {data.columns}")
                    if isinstance(data.columns, pd.MultiIndex):
                        logging.info(f"MultiIndex columns for {ticker}: {[col for col in data.columns]}")
                else:
                    logging.warning(f"Empty dataframe returned for {ticker}")
                
                if not data.empty:
                    # Check if we have the expected columns
                    expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    
                    # Yahoo Finance returns columns with either uppercase or lowercase first letter
                    # depending on the market/ticker - we'll normalize them
                    # Also handle MultiIndex columns (tuples) that some market data might return
                    rename_map = {}
                    
                    # Handle possible MultiIndex columns (e.g., ('AMD', 'Close'))
                    if isinstance(data.columns, pd.MultiIndex):
                        try:
                            # According to the logs, the format is different than expected
                            # For AAPL the format is ('Close', 'AAPL') where the first element 
                            # is the column name and the second is the ticker
                            
                            # Get the column level names to help diagnose
                            level_names = data.columns.names
                            logging.info(f"Column level names for {ticker}: {level_names}")
                            
                            # Check if the levels are something like ['Price', 'Ticker']
                            if level_names and 'Price' in level_names:
                                # In this case, we just want to drop the ticker level and keep the price level
                                if 'Ticker' in level_names:
                                    # Get the index of the 'Ticker' level
                                    ticker_level = level_names.index('Ticker')
                                    # Drop that level
                                    data.columns = data.columns.droplevel(ticker_level)
                                    logging.info(f"Dropped 'Ticker' level for {ticker}")
                                else:
                                    # Assuming the first column has price type and second has ticker
                                    data.columns = data.columns.droplevel(1)
                                    logging.info(f"Dropped second level for {ticker}")
                            else:
                                # Use a more general approach based on the structure observed in the logs
                                # From the logs, it looks like column names are sometimes in the first position
                                # and sometimes in the second position
                                
                                # Check if first elements are standard column names
                                first_elements = [col[0] if isinstance(col, tuple) and len(col) > 0 else str(col) for col in data.columns]
                                standard_names = {'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'}
                                
                                if any(name in standard_names for name in first_elements):
                                    # First elements are the column names, drop the second elements
                                    data.columns = [col[0] if isinstance(col, tuple) and len(col) > 0 else col for col in data.columns]
                                    logging.info(f"Used first elements as column names for {ticker}")
                                else:
                                    # Check second elements
                                    second_elements = [col[1] if isinstance(col, tuple) and len(col) > 1 else str(col) for col in data.columns]
                                    if any(name in standard_names for name in second_elements):
                                        # Second elements are the column names
                                        data.columns = [col[1] if isinstance(col, tuple) and len(col) > 1 else col for col in data.columns]
                                        logging.info(f"Used second elements as column names for {ticker}")
                                    else:
                                        # Just try to flatten the multi-index completely as a last resort
                                        data.columns = [col[-1] if isinstance(col, tuple) else col for col in data.columns]
                                        logging.info(f"Used last elements of tuples as column names for {ticker}")
                                
                            logging.info(f"After conversion, columns for {ticker}: {list(data.columns)}")
                            
                        except Exception as e:
                            logging.error(f"Error processing MultiIndex columns for {ticker}: {e}")
                            logging.error(f"Columns: {data.columns}")
                            
                            # Try a direct droplevel approach as a fallback
                            try:
                                # Just drop the first level and keep the second level which is usually standard columns
                                data.columns = data.columns.droplevel(0)
                                logging.info(f"Used droplevel(0) to fix MultiIndex columns for {ticker}")
                            except Exception as e2:
                                try:
                                    # Try dropping the second level instead
                                    data.columns = data.columns.droplevel(1)
                                    logging.info(f"Used droplevel(1) to fix MultiIndex columns for {ticker}")
                                except Exception as e3:
                                    logging.error(f"All column conversion methods failed for {ticker}: {e3}")
                    
                    # Now normalize column names to expected format
                    for col in data.columns:
                        # Skip non-string columns just in case
                        if not isinstance(col, str):
                            continue
                            
                        # Handle both 'Adj Close' and 'Adj_Close' variations
                        if col.lower() == 'adj close' or col.lower() == 'adj_close':
                            rename_map[col] = 'Adj Close'
                        elif col.lower() in [c.lower() for c in expected_columns]:
                            # Find the expected column that matches (case-insensitive)
                            for expected_col in expected_columns:
                                if col.lower() == expected_col.lower():
                                    rename_map[col] = expected_col
                                    break
                    
                    if rename_map:
                        data = data.rename(columns=rename_map)
                    
                    # Check if we now have the required columns
                    missing_columns = [col for col in expected_columns if col not in data.columns]
                    if missing_columns:
                        logging.warning(f"Missing columns for {ticker}: {missing_columns}")
                        # For simple price data with only OHLC, we can try to fill in missing Volume
                        if 'Volume' in missing_columns and len(missing_columns) == 1:
                            data['Volume'] = 0
                            logging.info(f"Added missing Volume column with zeros for {ticker}")
                        else:
                            logging.error(f"Cannot process {ticker} data - missing required columns: {missing_columns}")
                            continue
                    
                    # For intraday data, resample to desired resolution if needed
                    if interval != self.data_resolution and self.data_resolution != '1d':
                        # For resolutions like 4h that don't directly map to yfinance intervals
                        if self.data_resolution == '4h' and interval == '60m':
                            try:
                                # Resample 1h to 4h
                                data = data.resample('4H').agg({
                                    'Open': 'first', 
                                    'High': 'max', 
                                    'Low': 'min', 
                                    'Close': 'last', 
                                    'Volume': 'sum'
                                }).dropna()
                            except Exception as e:
                                logging.error(f"Error resampling data for {ticker}: {e}")
                                # If resampling fails, just use the original data
                                pass
                            
                    self.stock_data[ticker] = data
                    self._save_to_cache(ticker, data)
                    logging.info(f"Fetched data for {ticker}: {len(data)} periods at {self.data_resolution} resolution")
                else:
                    logging.warning(f"No data available for {ticker}")
            except Exception as e:
                logging.error(f"Error fetching data for {ticker}: {e}")

    
    def _detect_resistance_levels(self, data, lookback=200, min_touches=2, proximity_percent=0.03):
        """
        Detect resistance levels based on price highs.
        
        Args:
            data (pd.DataFrame): Price data with High, Low, Close columns
            lookback (int): Number of bars to look back for resistance detection
            min_touches (int): Minimum number of touches required to confirm a resistance
            proximity_percent (float): How close prices need to be to be considered the same level (as % of price)
            
        Returns:
            list: List of detected resistance levels
        """
        try:
            if len(data) < lookback:
                return []
                
            # Use only the last 'lookback' periods for the analysis
            recent_data = data.iloc[-lookback:].copy()
            
            # We'll consider swing highs as potential resistance points
            highs = recent_data['High'].values
            
            # Get local high points (swing highs)
            swing_highs = []
            
            # Using 2-bar look around (typical swing high)
            for i in range(2, len(highs) - 2):
                # A swing high occurs when a high is higher than surrounding bars
                if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and 
                    highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                    swing_highs.append(highs[i])
            
            # Also detect major swing highs with 5-bar look around
            for i in range(5, len(highs) - 5):
                # Major swing high (wider context)
                if all(highs[i] > highs[i-j] for j in range(1, 6)) and all(highs[i] > highs[i+j] for j in range(1, 6)):
                    # This is a significant high point - give it extra weight by adding it twice
                    swing_highs.append(highs[i])
                    swing_highs.append(highs[i])  # Add twice to give more weight in clustering
            
            # Always include the all-time high in the dataset
            all_time_high = max(highs)
            swing_highs.append(all_time_high)
            swing_highs.append(all_time_high)  # Add twice for more weight
            
            # If not enough swing highs found, use highest points
            if len(swing_highs) < min_touches * 2:  # Need more potential points for clustering
                # Just use the highest points in the dataset
                top_highs = sorted(highs, reverse=True)[:15]  # Use top 15 highest points
                swing_highs.extend(top_highs)
            
            # Cluster similar price levels (within proximity_percent of each other)
            resistance_levels = []
            for high in swing_highs:
                # Check if this high is close to any existing resistance level
                found_cluster = False
                for i, level in enumerate(resistance_levels):
                    # If within proximity, update the cluster average
                    if abs(high - level['price']) / level['price'] <= proximity_percent:
                        level['touches'] += 1
                        # Recalculate the average
                        level['price'] = (level['price'] * (level['touches'] - 1) + high) / level['touches']
                        found_cluster = True
                        break
                
                # If not part of an existing cluster, create a new one
                if not found_cluster:
                    resistance_levels.append({'price': high, 'touches': 1})
            
            # Filter levels with enough touches
            valid_levels = [level for level in resistance_levels if level['touches'] >= min_touches]
            
            # Sort by price
            valid_levels.sort(key=lambda x: x['price'])
            
            # Extract just the price values and return them
            return [level['price'] for level in valid_levels]
            
        except Exception as e:
            logging.error(f"Error detecting resistance levels: {e}")
            return []
            
    def _detect_support_levels(self, data, lookback=200, min_touches=2, proximity_percent=0.03):
        """
        Detect support levels based on price lows.
        
        Args:
            data (pd.DataFrame): Price data with High, Low, Close columns
            lookback (int): Number of bars to look back for support detection
            min_touches (int): Minimum number of touches required to confirm a support
            proximity_percent (float): How close prices need to be to be considered the same level (as % of price)
            
        Returns:
            list: List of detected support levels
        """
        try:
            if len(data) < lookback:
                return []
                
            # Use only the last 'lookback' periods for the analysis
            recent_data = data.iloc[-lookback:].copy()
            
            # We'll consider swing lows as potential support points
            lows = recent_data['Low'].values
            
            # Get local low points (swing lows)
            swing_lows = []
            
            # Using 2-bar look around (typical swing low)
            for i in range(2, len(lows) - 2):
                # A swing low occurs when a low is lower than surrounding bars
                if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and 
                    lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                    swing_lows.append(lows[i])
            
            # Also detect major swing lows with 5-bar look around
            for i in range(5, len(lows) - 5):
                # Major swing low (wider context)
                if all(lows[i] < lows[i-j] for j in range(1, 6)) and all(lows[i] < lows[i+j] for j in range(1, 6)):
                    # This is a significant low point - give it extra weight by adding it twice
                    swing_lows.append(lows[i])
                    swing_lows.append(lows[i])  # Add twice to give more weight in clustering
            
            # Always include the all-time low in the dataset
            all_time_low = min(lows)
            swing_lows.append(all_time_low)
            swing_lows.append(all_time_low)  # Add twice for more weight
            
            # If not enough swing lows found, use lowest points
            if len(swing_lows) < min_touches * 2:  # Need more potential points for clustering
                # Just use the lowest points in the dataset
                bottom_lows = sorted(lows)[:15]  # Use bottom 15 lowest points
                swing_lows.extend(bottom_lows)
            
            # Cluster similar price levels (within proximity_percent of each other)
            support_levels = []
            for low in swing_lows:
                # Check if this low is close to any existing support level
                found_cluster = False
                for i, level in enumerate(support_levels):
                    # If within proximity, update the cluster average
                    if abs(low - level['price']) / level['price'] <= proximity_percent:
                        level['touches'] += 1
                        # Recalculate the average
                        level['price'] = (level['price'] * (level['touches'] - 1) + low) / level['touches']
                        found_cluster = True
                        break
                
                # If not part of an existing cluster, create a new one
                if not found_cluster:
                    support_levels.append({'price': low, 'touches': 1})
            
            # Filter levels with enough touches
            valid_levels = [level for level in support_levels if level['touches'] >= min_touches]
            
            # Sort by price
            valid_levels.sort(key=lambda x: x['price'])
            
            # Extract just the price values and return them
            return [level['price'] for level in valid_levels]
            
        except Exception as e:
            logging.error(f"Error detecting support levels: {e}")
            return []
    
    def _calculate_indicators_for_ticker(self, ticker):
        """Calculate technical indicators for a specific ticker"""
        if ticker not in self.stock_data:
            logging.warning(f"No data available for {ticker}")
            return False
            
        data = self.stock_data[ticker]
        
        # Check if we have the required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logging.error(f"Missing required columns for {ticker}: {missing_columns}")
            return False
        
        try:
            # Ensure data has enough points for EMA calculation
            if len(data) < max(self.ema_short, self.ema_long):
                logging.warning(f"Insufficient data for {ticker} to calculate EMAs")
                return False
                
            # Make a copy to avoid modifying original data
            data = data.copy()
            
            # Calculate EMAs
            data['EMA_short'] = data['Close'].ewm(span=self.ema_short, adjust=False).mean()
            data['EMA_long'] = data['Close'].ewm(span=self.ema_long, adjust=False).mean()
            data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()  # Add 50-period EMA
            data['Bullish'] = data['Close'] > data['Open']
            
            # Calculate Average True Range (ATR) - 14 period
            atr_indicator = AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close'], window=14)
            data['ATR_14'] = atr_indicator.average_true_range()
            
            # Calculate Half and Full ATR values from last ATR
            last_atr = data['ATR_14'].iloc[-1]
            if isinstance(last_atr, pd.Series):
                last_atr = last_atr.item()
            
            # Store ATR levels for plotting
            current_close = data['Close'].iloc[-1]
            if isinstance(current_close, pd.Series):
                current_close = current_close.item()
                
            # Store ATR levels dictionary for this ticker
            self.atr_levels = getattr(self, 'atr_levels', {})
            self.atr_levels[ticker] = {
                'atr': last_atr,
                'atr_half_up': current_close + (last_atr * 0.5),
                'atr_full_up': current_close + last_atr,
                'atr_half_down': current_close - (last_atr * 0.5),
                'atr_full_down': current_close - last_atr,
            }
            
            logging.debug(f"Calculated 14-day ATR for {ticker}: {last_atr:.2f}")
            
            # Detect technical resistance levels based on price history
            logging.debug(f"Detecting technical resistance levels for {ticker}")
            resistance_levels = self._detect_resistance_levels(data)
            
            # Make sure we have some resistance levels for demo purposes
            if not resistance_levels:
                current_close = data['Close'].iloc[-1]
                if isinstance(current_close, pd.Series):
                    current_close = current_close.item()
                
                hardcoded_resistance_levels = [
                    current_close * 1.05,  # 5% above current price 
                    current_close * 1.10,  # 10% above current price
                    current_close * 1.15   # 15% above current price
                ]
                resistance_levels.extend(hardcoded_resistance_levels)
                logging.debug(f"Added default resistance levels for {ticker}")
            
            # Store resistance levels as a property of the class instance
            # This is more reliable than DataFrame attributes which can be lost when slicing
            self.resistance_levels[ticker] = resistance_levels
            
            logging.debug(f"Stored {len(resistance_levels)} resistance levels for {ticker}")
            
            # Detect technical support levels based on price lows
            logging.debug(f"Detecting technical support levels for {ticker}")
            support_levels = self._detect_support_levels(data)
            
            # Make sure we have some support levels for demo purposes
            if not support_levels:
                current_close = data['Close'].iloc[-1]
                if isinstance(current_close, pd.Series):
                    current_close = current_close.item()
                
                hardcoded_support_levels = [
                    current_close * 0.95,  # 5% below current price 
                    current_close * 0.90,  # 10% below current price
                    current_close * 0.85   # 15% below current price
                ]
                support_levels.extend(hardcoded_support_levels)
                logging.debug(f"Added default support levels for {ticker}")
            
            # Store support levels as a property of the class instance
            self.support_levels[ticker] = support_levels
            
            logging.debug(f"Stored {len(support_levels)} support levels for {ticker}")
            
            # Initialize EMA_Crossover as boolean column
            data['EMA_Crossover'] = False
            
            # Calculate crossovers row-by-row
            for i in range(1, len(data)):
                prev_short = data['EMA_short'].iloc[i-1]
                prev_long = data['EMA_long'].iloc[i-1]
                curr_short = data['EMA_short'].iloc[i]
                curr_long = data['EMA_long'].iloc[i]
                
                if prev_short <= prev_long and curr_short > curr_long:
                    data.iloc[i, data.columns.get_loc('EMA_Crossover')] = True
            
            # Force boolean type
            data['EMA_Crossover'] = data['EMA_Crossover'].astype(bool)
            
            # Debug information about indicators
            logging.debug(f"Calculated indicators for {ticker}: {len(data)} data points")
            if resistance_levels:
                logging.info(f"Detected {len(resistance_levels)} resistance levels for {ticker}: {resistance_levels}")
            
            self.stock_data[ticker] = data
            return True
            
        except Exception as e:
            logging.error(f"Error calculating indicators for {ticker}: {e}")
            return False
    
    def calculate_indicators(self):
        """Calculate technical indicators for all tickers"""
        for ticker in self.stock_data:
            self._calculate_indicators_for_ticker(ticker)

    def generate_signals(self, lookback_days=5, post_crossover_window=3):
        current_signals = []
        
        for ticker, data in self.stock_data.items():
            if len(data) < lookback_days:
                logging.warning(f"Not enough data for {ticker} in last {lookback_days} days")
                continue
                
            recent_data = data.iloc[-lookback_days:].copy()  # Explicit copy to avoid view issues
            last_crossover_date = None  # Store the date of the most recent crossover
            
            # Check if EMA_short is above EMA_long throughout the entire lookback period
            ema_consistently_above = True
            for idx, row in recent_data.iterrows():
                short_ema = row['EMA_short']
                long_ema = row['EMA_long']
                if isinstance(short_ema, pd.Series):
                    short_ema = short_ema.item()
                if isinstance(long_ema, pd.Series):
                    long_ema = long_ema.item()
                
                if short_ema <= long_ema:
                    ema_consistently_above = False
                    break
            
            for date, row in recent_data.iterrows():
                try:
                    # Extract EMA_Crossover as a scalar boolean
                    ema_crossover = row['EMA_Crossover'].item() if isinstance(row['EMA_Crossover'], pd.Series) else row['EMA_Crossover']
                    if not isinstance(ema_crossover, (bool, np.bool_)):
                        raise ValueError(f"EMA_Crossover is not a scalar boolean, type: {type(ema_crossover)}, value: {ema_crossover}")
                    
                    # If a crossover event occurred on this day, update last_crossover_date
                    if ema_crossover:
                        last_crossover_date = date
                    
                    # Extract the bullish condition as a scalar boolean
                    bullish = row['Bullish'].item() if isinstance(row['Bullish'], pd.Series) else row['Bullish']
                    
                    # Extract EMA values
                    short_ema = row['EMA_short']
                    if isinstance(short_ema, pd.Series):
                        short_ema = short_ema.item()
                        
                    long_ema = row['EMA_long']
                    if isinstance(long_ema, pd.Series):
                        long_ema = long_ema.item()
                    
                    # Check buy signal conditions:
                    # 1. Either a recent crossover OR EMA_short consistently above EMA_long in the entire period
                    # 2. Volume, bullish, and price conditions are met
                    if (((last_crossover_date is not None and (date - last_crossover_date).days <= post_crossover_window) or 
                        (ema_consistently_above and short_ema > long_ema)) and 
                        row['Volume'] >= self.min_volume and 
                        bullish and 
                        row['Close'] > row['EMA_short'] and
                        row['Close'] > row['EMA_50']):  # Added condition: close price must be above 50-period EMA
                        
                        signal = self._create_signal_dict(ticker, date, 'BUY', row)
                        current_signals.append(signal)
                        logging.info(f"BUY signal for {ticker} on {date.date()}")
                        
                    # Sell signal conditions remain as before.
                    elif (not bullish and row['Close'] < row['EMA_short']):
                        signal = self._create_signal_dict(ticker, date, 'SELL', row)
                        current_signals.append(signal)
                        logging.info(f"SELL signal for {ticker} on {date.date()}")
                        
                except Exception as e:
                    logging.error(f"Error generating signal for {ticker} on {date}: {str(e)}")
                    continue
        
        self.signals.extend(current_signals)
        return current_signals

    def _create_signal_dict(self, ticker, date, signal_type, row):
        """Helper method to create signal dictionary with scalar values."""
        # Extract scalar values for each metric
        close_price = row['Close']
        if isinstance(close_price, pd.Series):
            close_price = close_price.iloc[0]
            
        volume = row['Volume']
        if isinstance(volume, pd.Series):
            volume = volume.iloc[0]
            
        ema_short = row['EMA_short']
        if isinstance(ema_short, pd.Series):
            ema_short = ema_short.iloc[0]
            
        ema_long = row['EMA_long']
        if isinstance(ema_long, pd.Series):
            ema_long = ema_long.iloc[0]
            
        return {
            'ticker': ticker,
            'date': date,
            'type': signal_type,
            'price': close_price,
            'volume': volume,
            'ema_short': ema_short,
            'ema_long': ema_long
        }

    def visualize_signals(self, ticker, days=30):
        if ticker not in self.stock_data:
            logging.warning(f"No data available for {ticker}")
            return
            
        try:
            data = self.stock_data[ticker].iloc[-days:]
            if data.empty:
                logging.warning(f"No recent data available for {ticker}")
                return
                
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                            gridspec_kw={'height_ratios': [3, 1]})
            fig.suptitle(f"{ticker} - {self.ema_short}/{self.ema_long} EMA Signals")
            
            # Plot price and EMAs
            ax1.plot(data.index, data['Close'], label='Close', color='black', alpha=0.75)
            ax1.plot(data.index, data['EMA_short'], label=f'{self.ema_short} EMA', color='blue')
            ax1.plot(data.index, data['EMA_long'], label=f'{self.ema_long} EMA', color='red')
            ax1.plot(data.index, data['EMA_50'], label='50 EMA', color='green', linestyle='--')
            
            # Plot signals
            for signal_type, marker, color in [('BUY', '^', 'green'), ('SELL', 'v', 'red')]:
                signals = [s for s in self.signals if s['ticker'] == ticker and 
                            s['type'] == signal_type and s['date'] in data.index]
                for s in signals:
                    ax1.scatter(s['date'], s['price'], marker=marker, color=color, s=100)
                    ax1.annotate(signal_type, (s['date'], s['price']),
                                xytext=(0, 20 if signal_type == 'BUY' else -20),
                                textcoords='offset points',
                                arrowprops=dict(arrowstyle='->', color=color))

            # Volume plot
            ax2.bar(list(data.index), list(data['Volume']), color='blue', alpha=0.5)
            ax2.axhline(y=self.min_volume, color='red', linestyle='--',
                        label=f'Min Vol ({self.min_volume/1000000}M)')
            
            # Format plots
            for ax in (ax1, ax2):
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.setp(ax.get_xticklabels(), rotation=45)
                
            ax1.set_ylabel('Price ($)')
            ax2.set_ylabel('Volume')
            plt.tight_layout()
            plt.show()
        
        except Exception as e:
            logging.error(f"Error visualizing signals for {ticker}: {e}")

    def create_dashboard(self):
        if not self.signals:
            logging.info("No signals to display")
            return
            
        signals_df = pd.DataFrame(self.signals)
        signals_df['date'] = signals_df['date'].dt.date
        signals_df = signals_df.sort_values('date', ascending=False)
        dashboard = signals_df[['ticker', 'type', 'date', 'price', 'volume']]
        dashboard['volume'] = (dashboard['volume'] / 1000000).round(2).astype(str) + 'M'
        
        logging.info("\n==== TRADING SIGNALS DASHBOARD ====")
        logging.info(dashboard.head(10).to_string(index=False))
        logging.info(f"\nTotal signals: {len(signals_df)}")
        logging.info(f"Buy signals: {len(signals_df[signals_df['type'] == 'BUY'])}")
        logging.info(f"Sell signals: {len(signals_df[signals_df['type'] == 'SELL'])}")

    def send_notification(self, signal):
        if not self.email_notifications or not self.email_settings:
            return
            
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_settings['from_email']
            msg['To'] = self.email_settings['to_email']
            msg['Subject'] = f"{signal['type']} Signal: {signal['ticker']}"
            
            body = (f"Ticker: {signal['ticker']}\n"
                   f"Signal: {signal['type']}\n"
                   f"Date: {signal['date'].strftime('%Y-%m-%d')}\n"
                   f"Price: ${signal['price']:.2f}\n"
                   f"Volume: {signal['volume']/1000000:.2f}M\n"
                   f"{self.ema_short} EMA: ${signal['ema_short']:.2f}\n"
                   f"{self.ema_long} EMA: ${signal['ema_long']:.2f}")
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.email_settings['smtp_server'], 
                            self.email_settings['smtp_port']) as server:
                server.starttls()
                server.login(self.email_settings['username'], 
                           self.email_settings['password'])
                server.send_message(msg)
                
            logging.info(f"Notification sent for {signal['ticker']} {signal['type']}")
            
        except Exception as e:
            logging.error(f"Failed to send notification: {e}")

    def backtest_strategy(self, ticker, start_date, end_date):
        try:
            data = yf.download(ticker, start=start_date, end=end_date, 
                             interval="1d", progress=False)
            if data.empty:
                logging.warning(f"No data for {ticker} from {start_date} to {end_date}")
                return None
                
            # Calculate indicators
            data['EMA_short'] = data['Close'].ewm(span=self.ema_short, adjust=False).mean()
            data['EMA_long'] = data['Close'].ewm(span=self.ema_long, adjust=False).mean()
            data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()  # Add 50-period EMA
            data['Bullish'] = data['Close'] > data['Open']
            data['EMA_Crossover'] = (
                (data['EMA_short'] > data['EMA_long']) & 
                (data['EMA_short'].shift(1) <= data['EMA_long'].shift(1))
            )
            
            # Calculate ATR
            atr_indicator = AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close'], window=14)
            data['ATR_14'] = atr_indicator.average_true_range()
            
            position, entry_price, trades = 0, 0, []
            
            for date, row in data.iterrows():
                # Extract scalar values from each relevant column
                ema_crossover = row['EMA_Crossover']
                if isinstance(ema_crossover, pd.Series):
                    ema_crossover = ema_crossover.item()

                volume = row['Volume']
                if isinstance(volume, pd.Series):
                    volume = volume.item()

                close_price = row['Close']
                if isinstance(close_price, pd.Series):
                    close_price = close_price.item()

                ema_short = row['EMA_short']
                if isinstance(ema_short, pd.Series):
                    ema_short = ema_short.item()

                bullish = row['Bullish']
                if isinstance(bullish, pd.Series):
                    bullish = bullish.item()

                # Get the 50 EMA value
                ema_50 = row['EMA_50']
                if isinstance(ema_50, pd.Series):
                    ema_50 = ema_50.item()
                    
                # Get ATR value
                atr = row['ATR_14']
                if isinstance(atr, pd.Series):
                    atr = atr.item()

                # Check if we're in a consistently bullish EMA trend (short above long)
                # In backtest we can check the past N days directly
                consistently_bullish_ema = False
                if position == 0 and idx >= 5:  # Need at least 5 previous bars to check
                    consistently_bullish_ema = True
                    for i in range(1, 6):  # Check past 5 days
                        prev_idx = idx - i
                        if prev_idx < 0:
                            break
                        prev_short = data['EMA_short'].iloc[prev_idx]
                        prev_long = data['EMA_long'].iloc[prev_idx]
                        if isinstance(prev_short, pd.Series):
                            prev_short = prev_short.item()
                        if isinstance(prev_long, pd.Series):
                            prev_long = prev_long.item()
                        if prev_short <= prev_long:
                            consistently_bullish_ema = False
                            break
                
                # Now use these scalar values in your condition
                # Buy if either a crossover event or a consistently bullish EMA trend
                if (position == 0 and 
                    (ema_crossover or (consistently_bullish_ema and ema_short > ema_long)) and
                    volume >= self.min_volume and 
                    bullish and 
                    close_price > ema_short and
                    close_price > ema_50):  # Added condition: close price must be above 50-period EMA
                    
                    position = 1
                    entry_price = close_price
                    # Store ATR value at entry for potential stop loss calculation
                    entry_atr = atr
                    stop_loss = entry_price - entry_atr  # Full ATR as stop loss
                    
                    trades.append({
                        'date': date, 
                        'action': 'BUY', 
                        'price': entry_price,
                        'atr': entry_atr,
                        'stop_loss': stop_loss
                    })
                    
                elif position == 1:
                    # Check for stop loss hit (price dropped by full ATR from entry)
                    stop_loss = trades[-1]['stop_loss']
                    
                    if close_price <= stop_loss:
                        # Stop loss hit
                        position = 0
                        exit_price = close_price
                        profit = (exit_price - entry_price) / entry_price * 100
                        trades.append({
                            'date': date,
                            'action': 'STOP LOSS',
                            'price': exit_price,
                            'profit_pct': profit
                        })
                    elif (not bullish and close_price < ema_short):
                        # Regular sell signal
                        position = 0
                        exit_price = close_price
                        profit = (exit_price - entry_price) / entry_price * 100
                        trades.append({
                            'date': date,
                            'action': 'SELL',
                            'price': exit_price,
                            'profit_pct': profit
                        })
            
            # Close open position
            if position == 1:
                exit_price = data['Close'].iloc[-1]
                profit = (exit_price - entry_price) / entry_price * 100
                trades.append({
                    'date': data.index[-1],
                    'action': 'SELL (EOT)',
                    'price': exit_price,
                    'profit_pct': profit
                })
            
            trades_df = pd.DataFrame(trades)
            if trades_df.empty:
                logging.info("No trades generated during backtest")
                return self._empty_backtest_results(ticker, start_date, end_date)
                
            sell_trades = trades_df[trades_df['action'].str.contains('SELL|STOP')]
            total_return = ((sell_trades['profit_pct'] / 100 + 1).prod() - 1) * 100
            buy_hold = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100
            
            # Calculate statistics about ATR usage
            stop_loss_hits = len(trades_df[trades_df['action'] == 'STOP LOSS'])
            total_trades = len(sell_trades)
            stop_loss_percentage = (stop_loss_hits / total_trades * 100) if total_trades > 0 else 0
            
            # Calculate average ATR at entry
            buy_trades = trades_df[trades_df['action'] == 'BUY']
            avg_atr = buy_trades['atr'].mean() if 'atr' in buy_trades.columns and not buy_trades.empty else 0
            
            return {
                'ticker': ticker,
                'period': f"{start_date} to {end_date}",
                'trades': len(sell_trades),
                'win_rate': (len(sell_trades[sell_trades['profit_pct'] > 0]) / 
                           len(sell_trades) * 100 if len(sell_trades) > 0 else 0),
                'avg_profit': sell_trades['profit_pct'].mean() if len(sell_trades) > 0 else 0,
                'total_return': total_return,
                'buy_hold_return': buy_hold,
                'stop_loss_hits': stop_loss_hits,
                'stop_loss_percentage': stop_loss_percentage,
                'avg_atr': avg_atr,
                'trade_list': trades
            }
            
        except Exception as e:
            logging.error(f"Backtest error for {ticker}: {e}")
            return None

    def _empty_backtest_results(self, ticker, start_date, end_date):
        """Return empty backtest results dictionary."""
        return {
            'ticker': ticker,
            'period': f"{start_date} to {end_date}",
            'trades': 0,
            'win_rate': 0,
            'avg_profit': 0,
            'total_return': 0,
            'buy_hold_return': 0,
            'stop_loss_hits': 0,
            'stop_loss_percentage': 0,
            'avg_atr': 0
        }

    def run(self, mode='once', interval=None):
        try:
            if mode == 'once':
                self.fetch_data()
                self.calculate_indicators()
                signals = self.generate_signals()
                self.create_dashboard()
                if self.email_notifications:
                    for signal in signals:
                        self.send_notification(signal)
                return signals
                
            elif mode == 'continuous':
                interval = interval or 3600
                logging.info(f"Starting continuous monitoring every {interval}s")
                while True:
                    try:
                        self.fetch_data()
                        self.calculate_indicators()
                        signals = self.generate_signals(lookback_days=1)
                        if signals:
                            self.create_dashboard()
                            if self.email_notifications:
                                for signal in signals:
                                    self.send_notification(signal)
                        time.sleep(interval)
                    except Exception as e:
                        logging.error(f"Continuous mode error: {e}")
                        time.sleep(interval)
                        
            else:
                raise ValueError(f"Unknown mode: {mode}")
                
        except KeyboardInterrupt:
            logging.info("Monitoring stopped by user")
        except Exception as e:
            logging.error(f"Run error: {e}")

if __name__ == "__main__":
    # stock_list = ["AAPL"]
    stock_list = ["AAPL", "MSFT", "AMZN"]
    system = StockSignalSystem(tickers=stock_list)
    system.run()
    system.visualize_signals("AAPL")
    
    results = system.backtest_strategy("AAPL", "2023-01-01", "2023-12-31")
    if results:
        logging.info("\n==== BACKTEST RESULTS ====")
        for k, v in results.items():
            if k != 'trade_list':
                if k in ['avg_atr', 'stop_loss_percentage']:
                    logging.info(f"{k}: {v:.2f}")
                else:
                    logging.info(f"{k}: {v}")
        
        # Display ATR-specific statistics
        logging.info("\n==== ATR ANALYSIS ====")
        logging.info(f"Average 14-day ATR: ${results['avg_atr']:.2f}")
        logging.info(f"Stop losses triggered: {results['stop_loss_hits']} trades ({results['stop_loss_percentage']:.1f}% of all trades)")