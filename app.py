import matplotlib
matplotlib.use('TkAgg')

from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
from stock_signal_system import StockSignalSystem
import plotly.graph_objs as go
from plotly.offline import plot
from plotly.subplots import make_subplots
import datetime
import os
import json
import logging

app = Flask(__name__)
app.secret_key = 'stockpredictor_secret_key'  # Required for flash messages

# Config files paths
WATCHLIST_FILE = 'watchlist.json'
SETTINGS_FILE = 'settings.json'

# Default settings
DEFAULT_SETTINGS = {
    'ema_short': 10,
    'ema_long': 20,
    'min_volume': 20000000,
    'lookback_days': 5,
    'post_crossover_window': 3,
    'data_resolution': '1d',
    'data_period': 100  # Number of periods to load
}

# Function to load watchlist from file
def load_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE, 'r') as file:
            return json.load(file)
    else:
        # Default watchlist if file doesn't exist
        default_watchlist = ["AAPL", "BABA", "AMZN", "GOOG", "META", "NVDA"]
        save_watchlist(default_watchlist)
        return default_watchlist

# Function to save watchlist to file
def save_watchlist(watchlist):
    with open(WATCHLIST_FILE, 'w') as file:
        json.dump(watchlist, file)

# Function to load settings from file
def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as file:
            return json.load(file)
    else:
        # Use default settings if file doesn't exist
        save_settings(DEFAULT_SETTINGS)
        return DEFAULT_SETTINGS.copy()

# Function to save settings to file
def save_settings(settings):
    with open(SETTINGS_FILE, 'w') as file:
        json.dump(settings, file)

# Load settings
settings = load_settings()

# Initialize and populate your stock signal system
stock_list = load_watchlist()
system = StockSignalSystem(
    tickers=stock_list,
    ema_short=settings['ema_short'],
    ema_long=settings['ema_long'],
    min_volume=settings['min_volume'],
    data_resolution=settings['data_resolution'],
    data_period=settings['data_period']
)
system.fetch_data()
system.calculate_indicators()
system.generate_signals(
    lookback_days=settings['lookback_days'],
    post_crossover_window=settings['post_crossover_window']
)

@app.route('/')
def index():
    # Render a dashboard with just the most recent signal for each ticker
    signals_df = pd.DataFrame(system.signals)
    if not signals_df.empty and 'date' in signals_df.columns:
        # Convert date to string format for display
        signals_df['date'] = signals_df['date'].dt.strftime('%Y-%m-%d')
        
        # Format volume as millions with 2 decimal places
        # Convert to numeric first to ensure proper calculation
        signals_df['volume'] = pd.to_numeric(signals_df['volume'], errors='coerce')
        signals_df['volume'] = signals_df['volume'].apply(lambda x: f"{x/1000000:.2f}M" if pd.notna(x) else "N/A")
        
        # Format price with 2 decimal places
        signals_df['price'] = pd.to_numeric(signals_df['price'], errors='coerce')
        signals_df['price'] = signals_df['price'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
        
        # Sort by date (newest first)
        signals_df = signals_df.sort_values('date', ascending=False)
        
        # Keep only the most recent signal for each ticker
        latest_signals_df = signals_df.loc[signals_df.groupby('ticker')['date'].idxmax()]
    else:
        latest_signals_df = signals_df
    
    signals = latest_signals_df.to_dict(orient='records')
    watchlist = load_watchlist()
    current_settings = load_settings()
    return render_template('index.html', signals=signals, tickers=system.tickers, 
                          watchlist=watchlist, settings=current_settings)
                          
@app.route('/settings/update', methods=['POST'])
def update_settings():
    # Get current settings and update with form values
    current_settings = load_settings()
    
    try:
        current_settings['ema_short'] = int(request.form.get('ema_short', current_settings['ema_short']))
        current_settings['ema_long'] = int(request.form.get('ema_long', current_settings['ema_long']))
        current_settings['min_volume'] = int(request.form.get('min_volume', current_settings['min_volume']))
        current_settings['lookback_days'] = int(request.form.get('lookback_days', current_settings['lookback_days']))
        current_settings['post_crossover_window'] = int(request.form.get('post_crossover_window', current_settings['post_crossover_window']))
        current_settings['data_resolution'] = request.form.get('data_resolution', current_settings['data_resolution'])
        current_settings['data_period'] = int(request.form.get('data_period', current_settings['data_period']))
        
        # Validate settings
        if current_settings['ema_short'] >= current_settings['ema_long']:
            flash('Short EMA period must be less than Long EMA period', 'error')
            return redirect(url_for('index'))
            
        if current_settings['min_volume'] <= 0:
            flash('Minimum volume must be positive', 'error')
            return redirect(url_for('index'))
            
        if current_settings['lookback_days'] <= 0:
            flash('Lookback days must be positive', 'error')
            return redirect(url_for('index'))
            
        if current_settings['post_crossover_window'] <= 0:
            flash('Post-crossover window must be positive', 'error')
            return redirect(url_for('index'))
            
        if current_settings['data_period'] < 10:
            flash('Data period must be at least 10', 'error')
            return redirect(url_for('index'))
            
        # Validate data resolution
        valid_resolutions = ['1d', '4h', '1h', '30m', '15m']
        if current_settings['data_resolution'] not in valid_resolutions:
            flash('Invalid data resolution', 'error')
            return redirect(url_for('index'))
        
        # Save settings
        save_settings(current_settings)
        
        # Reinitialize the system with new settings
        global system
        watchlist = load_watchlist()
        system = StockSignalSystem(
            tickers=watchlist,
            ema_short=current_settings['ema_short'], 
            ema_long=current_settings['ema_long'],
            min_volume=current_settings['min_volume'],
            data_resolution=current_settings['data_resolution'],
            data_period=current_settings['data_period']
        )
        system.fetch_data()
        system.calculate_indicators()
        system.generate_signals(
            lookback_days=current_settings['lookback_days'],
            post_crossover_window=current_settings['post_crossover_window']
        )
        
        flash('Settings updated successfully', 'success')
    except ValueError:
        flash('Invalid settings values. Please enter valid numbers.', 'error')
    
    return redirect(url_for('index'))

@app.route('/watchlist/add', methods=['POST'])
def add_to_watchlist():
    ticker = request.form.get('ticker', '').strip().upper()
    if not ticker:
        flash('Please enter a valid ticker symbol', 'error')
        return redirect(url_for('index'))
        
    watchlist = load_watchlist()
    
    if ticker in watchlist:
        flash(f'{ticker} is already in your watchlist', 'info')
    else:
        watchlist.append(ticker)
        save_watchlist(watchlist)
        
        # Load current settings
        current_settings = load_settings()
        
        # Update the system with the new ticker
        global system
        system = StockSignalSystem(
            tickers=watchlist,
            ema_short=current_settings['ema_short'],
            ema_long=current_settings['ema_long'],
            min_volume=current_settings['min_volume'],
            data_resolution=current_settings['data_resolution'],
            data_period=current_settings['data_period']
        )
        system.fetch_data()
        system.calculate_indicators()
        system.generate_signals(
            lookback_days=current_settings['lookback_days'],
            post_crossover_window=current_settings['post_crossover_window']
        )
        
        flash(f'{ticker} has been added to your watchlist', 'success')
    
    return redirect(url_for('index'))

@app.route('/watchlist/remove/<ticker>', methods=['GET', 'POST'])
def remove_from_watchlist(ticker):
    watchlist = load_watchlist()
    
    if ticker in watchlist:
        watchlist.remove(ticker)
        save_watchlist(watchlist)
        
        # Load current settings
        current_settings = load_settings()
        
        # Update the system with the updated watchlist
        global system
        system = StockSignalSystem(
            tickers=watchlist,
            ema_short=current_settings['ema_short'],
            ema_long=current_settings['ema_long'],
            min_volume=current_settings['min_volume'],
            data_resolution=current_settings['data_resolution'],
            data_period=current_settings['data_period']
        )
        system.fetch_data()
        system.calculate_indicators()
        system.generate_signals(
            lookback_days=current_settings['lookback_days'],
            post_crossover_window=current_settings['post_crossover_window']
        )
        
        flash(f'{ticker} has been removed from your watchlist', 'success')
    else:
        flash(f'{ticker} is not in your watchlist', 'error')
    
    return redirect(url_for('index'))

@app.route('/watchlist/refresh', methods=['GET'])
def refresh_watchlist():
    watchlist = load_watchlist()
    current_settings = load_settings()
    
    # Refresh data for all tickers in the watchlist
    global system
    system = StockSignalSystem(
        tickers=watchlist,
        ema_short=current_settings['ema_short'],
        ema_long=current_settings['ema_long'],
        min_volume=current_settings['min_volume'],
        data_resolution=current_settings['data_resolution'],
        data_period=current_settings['data_period']
    )
    system.fetch_data()
    system.calculate_indicators()
    system.generate_signals(
        lookback_days=current_settings['lookback_days'],
        post_crossover_window=current_settings['post_crossover_window']
    )
    
    flash('Watchlist data has been refreshed', 'success')

    # Save CSV summary of tickers, resistance, support, ATR, volume, last signal
    try:
        summary_data = []

        for ticker in system.tickers:
            res_levels = system.resistance_levels.get(ticker, [])
            sup_levels = system.support_levels.get(ticker, [])
            atrs = getattr(system, 'atr_levels', {}).get(ticker, {})
            data = system.stock_data.get(ticker)

            last_resistance = res_levels[0] if res_levels else None
            last_support = sup_levels[0] if sup_levels else None

            atr_value = atrs.get('atr', None)
            atr_full_up = atrs.get('atr_full_up', None)
            atr_full_down = atrs.get('atr_full_down', None)

            most_recent_volume = data['Volume'].iloc[-1] if data is not None and not data.empty else None

            ticker_signals = [s for s in system.signals if s['ticker'] == ticker]
            last_signal = ticker_signals[-1]['type'] if ticker_signals else None

            summary_data.append({
                'Ticker': ticker,
                'LastResistance': last_resistance,
                'LastSupport': last_support,
                'ATR': atr_value,
                'ATR_Full_Up': atr_full_up,
                'ATR_Full_Down': atr_full_down,
                'MostRecentVolume': most_recent_volume,
                'LastSignal': last_signal
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv("system_snapshot.csv", index=False)
        logging.info("System snapshot saved to system_snapshot.csv")
    except Exception as e:
        logging.error(f"Error saving system snapshot CSV: {e}", exc_info=True)

    return redirect(url_for('index'))


@app.route('/plot_single', methods=['GET', 'POST'])
def plot_single():
    """Display single stock chart using TradingView Lightweight Charts"""
    # Determine the selected ticker from GET parameters or form submission
    watchlist = load_watchlist()
    selected_ticker = request.values.get('ticker')
    if not selected_ticker or selected_ticker not in watchlist:
        # Default to the first ticker if none is selected or invalid
        selected_ticker = watchlist[0] if watchlist else None
        
    if not selected_ticker:
        flash('No tickers in your watchlist. Please add some first.', 'error')
        return redirect(url_for('index'))
    
    # Get number of days to show
    days = int(request.values.get('days', 100))
    
    # Ensure data is available for the selected ticker
    if selected_ticker not in system.stock_data or system.stock_data[selected_ticker].empty:
        # Try to fetch data specifically for this ticker if it's not available
        current_settings = load_settings()
        
        try:
            # Create a temporary system just for this ticker
            temp_system = StockSignalSystem(
                tickers=[selected_ticker],
                ema_short=current_settings['ema_short'],
                ema_long=current_settings['ema_long'],
                min_volume=current_settings['min_volume'],
                data_resolution=current_settings['data_resolution'],
                data_period=current_settings['data_period']
            )
            temp_system.fetch_data()
            
            if temp_system.stock_data and selected_ticker in temp_system.stock_data and not temp_system.stock_data[selected_ticker].empty:
                # Use the newly fetched data
                system.stock_data[selected_ticker] = temp_system.stock_data[selected_ticker]
                # Calculate indicators for this ticker
                if not system._calculate_indicators_for_ticker(selected_ticker):
                    flash(f'Could not calculate indicators for {selected_ticker}. Ticker might not have required price data.', 'error')
                    return redirect(url_for('index'))
            else:
                flash(f'No data available for {selected_ticker}. This ticker might not be supported or might not have data for the selected resolution.', 'error')
                return redirect(url_for('index'))
        except Exception as e:
            logging.error(f"Error loading {selected_ticker}: {str(e)}")
            flash(f'Error loading {selected_ticker}: {str(e)}', 'error')
            return redirect(url_for('index'))
    
    # Use the TradingView template instead of Plotly
    return render_template('tradingview_chart.html',
                          selected_ticker=selected_ticker,
                          tickers=watchlist,
                          days=days,
                          system=system)


@app.route('/plot', methods=['GET'])
def plot_combined():
    """Display multiple stocks in a combined view for comparison"""
    # Get the watchlist
    watchlist = load_watchlist()
    if not watchlist:
        flash('No tickers in your watchlist. Please add some first.', 'error')
        return redirect(url_for('index'))
    
    # Get selected tickers from query parameters, or use all if none specified
    selected_tickers = request.args.getlist('tickers')
    if not selected_tickers:
        selected_tickers = watchlist[:3] if len(watchlist) >= 3 else watchlist  # Default to first 3 tickers or all if less
    else:
        # Filter to ensure only valid tickers are used
        selected_tickers = [t for t in selected_tickers if t in watchlist]
        if not selected_tickers:
            selected_tickers = [watchlist[0]]
    
    return render_template('plot.html',
                          tickers=watchlist,
                          selected_tickers=selected_tickers,
                          system=system)


@app.route('/tradingview_chart')
def tradingview_chart():
    """Professional TradingView-style chart view"""
    watchlist = load_watchlist()
    selected_ticker = request.args.get('ticker')
    if not selected_ticker or selected_ticker not in watchlist:
        selected_ticker = watchlist[0] if watchlist else 'AAPL'
    
    return render_template('tradingview_chart.html', 
                          selected_ticker=selected_ticker, 
                          tickers=watchlist)


@app.route('/api/chart_data/<ticker>')
def api_chart_data(ticker):
    """API endpoint to provide chart data in TradingView format"""
    try:
        # Get timeframe and period from query parameters
        timeframe = request.args.get('timeframe', '1d')
        period = int(request.args.get('period', 100))
        
        # Check if ticker data exists in system
        if ticker not in system.stock_data or system.stock_data[ticker].empty:
            # Fetch data for this ticker if not available
            current_settings = load_settings()
            temp_system = StockSignalSystem(
                tickers=[ticker],
                ema_short=current_settings['ema_short'],
                ema_long=current_settings['ema_long'],
                min_volume=current_settings['min_volume'],
                data_resolution=timeframe,
                data_period=period
            )
            temp_system.fetch_data()
            
            if temp_system.stock_data and ticker in temp_system.stock_data:
                system.stock_data[ticker] = temp_system.stock_data[ticker]
                # Calculate indicators for this ticker
                system._calculate_indicators_for_ticker(ticker)
            else:
                return {"error": "Failed to fetch data for ticker"}, 404
        
        data = system.stock_data[ticker].tail(period)
        
        # Convert to TradingView Lightweight Charts format
        candlesticks = []
        volume_data = []
        ema20_data = []
        ema50_data = []
        
        for index, row in data.iterrows():
            # Convert timestamp to Unix timestamp
            timestamp = int(index.timestamp())
            
            candlesticks.append({
                'time': timestamp,
                'open': float(row['Open']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'close': float(row['Close'])
            })
            
            volume_data.append({
                'time': timestamp,
                'value': float(row['Volume']),
                'color': '#26a69a' if row['Close'] >= row['Open'] else '#ef5350'
            })
            
            # Add EMA data if available
            if 'EMA_short' in row and pd.notna(row['EMA_short']):
                ema20_data.append({
                    'time': timestamp,
                    'value': float(row['EMA_short'])
                })
            
            if 'EMA_50' in row and pd.notna(row['EMA_50']):
                ema50_data.append({
                    'time': timestamp,
                    'value': float(row['EMA_50'])
                })
        
        # Get latest price info
        latest = data.iloc[-1]
        latest_info = {
            'symbol': ticker,
            'price': float(latest['Close']),
            'change': float(latest['Close'] - data.iloc[-2]['Close']) if len(data) > 1 else 0,
            'change_percent': float((latest['Close'] - data.iloc[-2]['Close']) / data.iloc[-2]['Close'] * 100) if len(data) > 1 else 0,
            'volume': float(latest['Volume']),
            'high_24h': float(data['High'].max()),
            'low_24h': float(data['Low'].min())
        }
        
        # Get support and resistance levels
        support_levels = []
        resistance_levels = []
        atr_levels = []
        
        # Get the latest price for ATR calculation
        if len(data) > 0:
            latest_price = float(latest['Close'])
            
            # Calculate ATR levels if available
            if 'ATR' in data.columns:
                atr_value = data['ATR'].iloc[-1]
                if pd.notna(atr_value):
                    atr_levels = [
                        {'price': latest_price + atr_value, 'type': 'atr_up'},
                        {'price': latest_price - atr_value, 'type': 'atr_down'}
                    ]
            
            # Get support and resistance from system if available
            if hasattr(system, 'support_levels') and ticker in system.support_levels:
                support_levels = [{'price': level} for level in system.support_levels[ticker]]
            if hasattr(system, 'resistance_levels') and ticker in system.resistance_levels:
                resistance_levels = [{'price': level} for level in system.resistance_levels[ticker]]

        return {
            'candlesticks': candlesticks,
            'volume': volume_data,
            'ema20': ema20_data,
            'ema50': ema50_data,
            'info': latest_info,
            'signals': [s for s in system.signals if s['ticker'] == ticker],
            'support_levels': support_levels,
            'resistance_levels': resistance_levels,
            'atr_levels': atr_levels
        }
        
    except Exception as e:
        logging.error(f"Error fetching chart data for {ticker}: {str(e)}")
        return {"error": str(e)}, 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
