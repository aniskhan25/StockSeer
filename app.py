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
    return redirect(url_for('index'))


@app.route('/plot_single', methods=['GET', 'POST'])
def plot_single():
    # Determine the selected ticker from GET parameters or form submission
    watchlist = load_watchlist()
    selected_ticker = request.values.get('ticker')
    if not selected_ticker or selected_ticker not in watchlist:
        # Default to the first ticker if none is selected or invalid
        selected_ticker = watchlist[0] if watchlist else None
        
    if not selected_ticker:
        flash('No tickers in your watchlist. Please add some first.', 'error')
        return redirect(url_for('index'))
    
    # Ensure data is available for the selected ticker
    data_available = True
    
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
                    data_available = False
                    flash(f'Could not calculate indicators for {selected_ticker}. Ticker might not have required price data.', 'error')
            else:
                data_available = False
                flash(f'No data available for {selected_ticker}. This ticker might not be supported or might not have data for the selected resolution.', 'error')
        except Exception as e:
            data_available = False
            logging.error(f"Error loading {selected_ticker}: {str(e)}")
            flash(f'Error loading {selected_ticker}: {str(e)}', 'error')
    
    if not data_available:
        return redirect(url_for('index'))
    
    # Use the most recent 50 days of data
    days_to_show = int(request.values.get('days', 50))
    data = system.stock_data[selected_ticker].iloc[-days_to_show:]
    
    # Create subplot with two rows (price/EMA and volume)
    fig = make_subplots(
        rows=2, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=(
            f"{selected_ticker} Price with {system.ema_short}/{system.ema_long} EMA", 
            "Volume"
        )
    )
    
    # Clear any existing traces and layout to ensure no residual elements
    fig.data = []
    
    # Create candlestick chart on top subplot
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=f'{selected_ticker}',
            increasing_line_color='#26a69a',
            increasing_fillcolor='#26a69a',
            decreasing_line_color='#ef5350',
            decreasing_fillcolor='#ef5350',
            line=dict(width=1),
            whiskerwidth=0.5
        ),
        row=1, col=1
    )
    
    # Add trace for 50 EMA with a gray transparent line (add this first so it's in the background)
    if 'EMA_50' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['EMA_50'],
                mode='lines',
                name='50 EMA',
                line=dict(color='#9E9E9E', width=1.5),
                opacity=0.3,
                hoverinfo='text',
                hovertext=[f"50 EMA: ${val:.2f}" for val in data['EMA_50']]
            ),
            row=1, col=1
        )
    
    # Add trace for short EMA with a blue line
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['EMA_short'],
            mode='lines',
            name=f'{system.ema_short} EMA',
            line=dict(color='#2196F3', width=2),
            opacity=0.8
        ),
        row=1, col=1
    )
    
    # Add trace for long EMA with a red line
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['EMA_long'],
            mode='lines',
            name=f'{system.ema_long} EMA',
            line=dict(color='#FF5722', width=2),
            opacity=0.8
        ),
        row=1, col=1
    )
    
    # Add markers for signals specific to the selected ticker
    ticker_signals = [s for s in system.signals if s['ticker'] == selected_ticker]
    
    # Get the date range for our data
    start_date = data.index[0]
    end_date = data.index[-1]
    
    # Filter signals in our date range without using 'in' operator
    visible_signals = []
    for signal in ticker_signals:
        signal_date = signal['date']
        if start_date <= signal_date <= end_date:
            visible_signals.append(signal)
    
    logging.debug(f"Found {len(visible_signals)} visible signals for {selected_ticker} in date range {start_date} to {end_date}")
    
    # Add each signal marker
    for signal in visible_signals:
        marker_color = '#4CAF50' if signal['type'] == 'BUY' else '#F44336'
        marker_symbol = 'triangle-up' if signal['type'] == 'BUY' else 'triangle-down'
        
        # Get price as scalar value
        price = signal['price']
        if isinstance(price, pd.Series):
            price = price.item()
            
        fig.add_trace(
            go.Scatter(
                x=[signal['date']],
                y=[price],
                mode='markers+text',
                marker=dict(
                    color=marker_color, 
                    size=20, 
                    symbol=marker_symbol,
                    line=dict(width=2, color='black')
                ),
                text=[signal['type']],
                textposition='top center',
                name=f"{signal['type']} @ {price:.2f}",
                hoverinfo='text',
                hovertext=f"{signal['type']} Signal on {signal['date'].strftime('%Y-%m-%d')}<br>Price: ${price:.2f}",
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Add stop loss marker for BUY signals (3% below close price)
        if signal['type'] == 'BUY':
            stop_loss_price = price * 0.97  # 3% below the buy price
            
            # Add the stop loss marker
            fig.add_trace(
                go.Scatter(
                    x=[signal['date']],
                    y=[stop_loss_price],
                    mode='markers+text',
                    marker=dict(
                        color='#9C27B0',  # Purple color for stop loss
                        size=15, 
                        symbol='circle',
                        line=dict(width=2, color='black')
                    ),
                    text=['STOP'],
                    textposition='bottom center',
                    name=f"Stop Loss @ {stop_loss_price:.2f}",
                    hoverinfo='text',
                    hovertext=f"Stop Loss: ${stop_loss_price:.2f} (3% below entry)",
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Add dashed line connecting the buy signal to the stop loss
            fig.add_trace(
                go.Scatter(
                    x=[signal['date'], signal['date']],
                    y=[price, stop_loss_price],
                    mode='lines',
                    line=dict(color='#9C27B0', width=1, dash='dash'),
                    showlegend=False,
                    hoverinfo='none'
                ),
                row=1, col=1
            )
    
    # Add volume as a bar chart
    # Fix: Compare scalar values from each row
    colors = []
    normalized_volumes = []
    max_volume = data['Volume'].max()
    
    for i, row in data.iterrows():
        close_val = row['Close'].item() if isinstance(row['Close'], pd.Series) else row['Close']
        open_val = row['Open'].item() if isinstance(row['Open'], pd.Series) else row['Open']
        colors.append('#26a69a' if close_val >= open_val else '#ef5350')
        
        # Get volume
        vol = row['Volume']
        if isinstance(vol, pd.Series):
            vol = vol.item()
        normalized_volumes.append(vol)
    
    # Add volume bars to bottom subplot
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=normalized_volumes,
            name='Volume',
            marker_color=colors,
            opacity=0.8,
            hoverinfo='text',
            hovertext=[f"Volume: {int(vol):,}" for vol in normalized_volumes],
        ),
        row=2, col=1
    )
    
    # Add a horizontal line for the minimum volume threshold
    fig.add_shape(
        type="line",
        x0=data.index[0],
        x1=data.index[-1],
        y0=system.min_volume,
        y1=system.min_volume,
        line=dict(
            color="#FF9800",
            width=1.5,
            dash="dot",
        ),
        row=2, col=1
    )
    
    # Add annotation for min volume threshold
    fig.add_annotation(
        x=data.index[-1],
        y=system.min_volume,
        text=f"Min Volume: {system.min_volume/1000000:.0f}M",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        xshift=0,
        yshift=5,
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="#FF9800",
        borderwidth=1,
        borderpad=4,
        font=dict(color="#FF9800", size=10),
        row=2, col=1
    )
    
    # Add resistance and support levels if available
    try:
        logging.debug(f"Adding resistance and support levels for {selected_ticker}")
        
        # Resistance levels - default values if none detected
        hardcoded_resistance = [
            data['Close'].max() * 1.02,  # 2% above max close
            data['Close'].max() * 1.05,  # 5% above max close
            data['Close'].max() * 1.10   # 10% above max close
        ]
        
        # Support levels - default values if none detected
        hardcoded_support = [
            data['Close'].min() * 0.98,  # 2% below min close
            data['Close'].min() * 0.95,  # 5% below min close
            data['Close'].min() * 0.90   # 10% below min close
        ]
        
        # Combine algorithm-detected levels with hardcoded ones
        all_resistance_levels = []
        all_support_levels = []
        
        # Get resistance levels
        if hasattr(system, 'resistance_levels') and selected_ticker in system.resistance_levels:
            algorithm_resistance = system.resistance_levels[selected_ticker]
            all_resistance_levels.extend(algorithm_resistance)
            logging.debug(f"Using {len(algorithm_resistance)} algorithm-detected resistance levels")
        
        if not all_resistance_levels:
            all_resistance_levels.extend(hardcoded_resistance)
            logging.debug("Using default resistance levels")
            
        # Get support levels
        if hasattr(system, 'support_levels') and selected_ticker in system.support_levels:
            algorithm_support = system.support_levels[selected_ticker]
            all_support_levels.extend(algorithm_support)
            logging.debug(f"Using {len(algorithm_support)} algorithm-detected support levels")
        
        if not all_support_levels:
            all_support_levels.extend(hardcoded_support)
            logging.debug("Using default support levels")
            
        # Add ATR levels if available
        if hasattr(system, 'atr_levels') and selected_ticker in system.atr_levels:
            atr_data = system.atr_levels[selected_ticker]
            
            # Add ATR full value upside line (purple solid)
            fig.add_shape(
                type="line",
                x0=data.index[0],
                x1=data.index[-1],
                y0=atr_data['atr_full_up'],
                y1=atr_data['atr_full_up'],
                line=dict(
                    color="#673AB7",  # Purple
                    width=1.5,
                ),
                opacity=0.7,
                row=1, col=1
            )
            
            # Add ATR half value upside line (purple dotted)
            fig.add_shape(
                type="line",
                x0=data.index[0],
                x1=data.index[-1],
                y0=atr_data['atr_half_up'],
                y1=atr_data['atr_half_up'],
                line=dict(
                    color="#673AB7",  # Purple
                    width=1.5,
                    dash="dot",
                ),
                opacity=0.7,
                row=1, col=1
            )
            
            # Add ATR full value downside line (orange solid)
            fig.add_shape(
                type="line",
                x0=data.index[0],
                x1=data.index[-1],
                y0=atr_data['atr_full_down'],
                y1=atr_data['atr_full_down'],
                line=dict(
                    color="#FF9800",  # Orange
                    width=1.5,
                ),
                opacity=0.7,
                row=1, col=1
            )
            
            # Add ATR half value downside line (orange dotted)
            fig.add_shape(
                type="line",
                x0=data.index[0],
                x1=data.index[-1],
                y0=atr_data['atr_half_down'],
                y1=atr_data['atr_half_down'],
                line=dict(
                    color="#FF9800",  # Orange
                    width=1.5,
                    dash="dot",
                ),
                opacity=0.7,
                row=1, col=1
            )
            
            # Add annotations for ATR levels
            fig.add_annotation(
                x=data.index[-1],
                y=atr_data['atr_full_up'],
                text=f"ATR Full: ${atr_data['atr_full_up']:.2f}",
                showarrow=False,
                xanchor="right",
                yanchor="bottom",
                font=dict(color="#673AB7", size=8),
                bgcolor="rgba(103, 58, 183, 0.1)",
                bordercolor="#673AB7",
                borderwidth=1,
                borderpad=2,
                opacity=0.7,
                row=1, col=1
            )
            
            fig.add_annotation(
                x=data.index[-1],
                y=atr_data['atr_half_up'],
                text=f"ATR Half: ${atr_data['atr_half_up']:.2f}",
                showarrow=False,
                xanchor="right",
                yanchor="bottom",
                font=dict(color="#673AB7", size=8),
                bgcolor="rgba(103, 58, 183, 0.1)",
                bordercolor="#673AB7",
                borderwidth=1,
                borderpad=2,
                opacity=0.7,
                row=1, col=1
            )
            
            fig.add_annotation(
                x=data.index[-1],
                y=atr_data['atr_half_down'],
                text=f"ATR Half: ${atr_data['atr_half_down']:.2f}",
                showarrow=False,
                xanchor="right",
                yanchor="top",
                font=dict(color="#FF9800", size=8),
                bgcolor="rgba(255, 152, 0, 0.1)",
                bordercolor="#FF9800",
                borderwidth=1,
                borderpad=2,
                opacity=0.7,
                row=1, col=1
            )
            
            fig.add_annotation(
                x=data.index[-1],
                y=atr_data['atr_full_down'],
                text=f"ATR Full: ${atr_data['atr_full_down']:.2f}",
                showarrow=False,
                xanchor="right",
                yanchor="top",
                font=dict(color="#FF9800", size=8),
                bgcolor="rgba(255, 152, 0, 0.1)",
                bordercolor="#FF9800",
                borderwidth=1,
                borderpad=2,
                opacity=0.7,
                row=1, col=1
            )
            
            # Add ATR value indicator
            fig.add_annotation(
                x=data.index[0],
                y=data['High'].max(),
                text=f"14-Day ATR: ${atr_data['atr']:.2f}",
                showarrow=False,
                xanchor="left",
                yanchor="top",
                font=dict(color="#333333", size=10, weight="bold"),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="#333333",
                borderwidth=1,
                borderpad=3,
                opacity=0.9,
                row=1, col=1
            )
            
        # Add horizontal lines for resistance levels
        displayed_count = 0
        for i, level in enumerate(all_resistance_levels):
            # Limit the number of resistance levels to display (maximum 3)
            if displayed_count >= 3:
                continue
                
            # Filter significant resistance levels by spread (at least 2% apart)
            if i > 0 and all_resistance_levels[i-1] > 0:
                # Skip if too close to previous level
                prev_level = all_resistance_levels[i-1]
                percent_diff = abs(level - prev_level) / prev_level
                if percent_diff < 0.02:
                    continue
            
            # Standard technical resistance - red dashed lines
            line_color = "#FF0000"  # Red
            line_width = 1  # Thinner line
            line_dash = "dash"  # Dashed line
            line_opacity = 0.4  # More transparent
            label_prefix = "R"  # Resistance
            label_color = "#FF0000"
            bg_color = "rgba(255, 0, 0, 0.1)"
            border_color = "#FF0000"
            
            # Add the horizontal line
            fig.add_shape(
                type="line",
                x0=data.index[0],
                x1=data.index[-1],
                y0=level,
                y1=level,
                line=dict(
                    color=line_color,
                    width=line_width,
                    dash=line_dash,
                ),
                opacity=line_opacity,
                row=1, col=1
            )
            
            # Add annotation for the resistance level
            fig.add_annotation(
                x=data.index[-1],
                y=level,
                text=f"{label_prefix}{displayed_count+1}: ${level:.2f}",
                showarrow=False,
                xanchor="right",
                yanchor="bottom",
                font=dict(color=label_color, size=8),
                bgcolor=bg_color,
                bordercolor=border_color,
                borderwidth=1,
                borderpad=2,
                opacity=0.7,
                row=1, col=1
            )
            
            displayed_count += 1
            
        # Add horizontal lines for support levels
        displayed_count = 0
        for i, level in enumerate(all_support_levels):
            # Limit the number of support levels to display (maximum 3)
            if displayed_count >= 3:
                continue
                
            # Filter significant support levels by spread (at least 2% apart)
            if i > 0 and all_support_levels[i-1] > 0:
                # Skip if too close to previous level
                prev_level = all_support_levels[i-1]
                percent_diff = abs(level - prev_level) / prev_level
                if percent_diff < 0.02:
                    continue
            
            # Standard technical support - green dashed lines
            line_color = "#4CAF50"  # Green
            line_width = 1  # Thinner line
            line_dash = "dash"  # Dashed line
            line_opacity = 0.4  # More transparent
            label_prefix = "S"  # Support
            label_color = "#4CAF50"
            bg_color = "rgba(76, 175, 80, 0.1)"
            border_color = "#4CAF50"
            
            # Add the horizontal line
            fig.add_shape(
                type="line",
                x0=data.index[0],
                x1=data.index[-1],
                y0=level,
                y1=level,
                line=dict(
                    color=line_color,
                    width=line_width,
                    dash=line_dash,
                ),
                opacity=line_opacity,
                row=1, col=1
            )
            
            # Add annotation for the support level
            fig.add_annotation(
                x=data.index[-1],
                y=level,
                text=f"{label_prefix}{displayed_count+1}: ${level:.2f}",
                showarrow=False,
                xanchor="right",
                yanchor="top",  # Place below the line
                font=dict(color=label_color, size=8),
                bgcolor=bg_color,
                bordercolor=border_color,
                borderwidth=1,
                borderpad=2,
                opacity=0.7,
                row=1, col=1
            )
            
            displayed_count += 1
        
    except Exception as e:
        logging.error(f"Error adding resistance and support levels: {e}", exc_info=True)
    
    # Update layout with modern styling
    fig.update_layout(
        title=dict(
            text=f"{selected_ticker} - Technical Analysis",
            font=dict(size=24, color='#444', family="Arial, sans-serif"),
            x=0.5
        ),
        paper_bgcolor='#f8f9fa',
        plot_bgcolor='#ffffff',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='#E0E0E0',
            borderwidth=1
        ),
        hovermode='x unified',
        height=800,
        margin=dict(l=60, r=60, t=80, b=60)
    )
    
    # Configure X-axis (date)
    fig.update_xaxes(
        title='Date',
        showgrid=True,
        gridcolor='#f0f0f0',
        tickfont=dict(size=10),
        rangeslider=dict(visible=True, thickness=0.05),
        row=1, col=1
    )
    
    # Configure Y-axis (price)
    fig.update_yaxes(
        title='Price ($)',
        tickprefix='$',
        tickformat='.2f',
        showgrid=True,
        gridcolor='#f0f0f0',
        zeroline=False,
        row=1, col=1
    )
    
    # Configure Y-axis (volume)
    fig.update_yaxes(
        title='Volume',
        showgrid=True,
        gridcolor='#f0f0f0',
        zeroline=False,
        tickformat=',',
        row=2, col=1
    )
    
    # Generate HTML div with Plotly figure
    plot_div = plot(fig, output_type='div', include_plotlyjs=True)
    
    watchlist = load_watchlist()
    return render_template('plot_single.html', plot_div=plot_div, tickers=watchlist, 
                          selected_ticker=selected_ticker, days=days_to_show, 
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
    
    # Get the time period to display
    days_to_show = int(request.args.get('days', 30))
    
    # Create a figure with one subplot per ticker
    n_tickers = len(selected_tickers)
    fig = make_subplots(
        rows=n_tickers, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[f"{ticker} - {system.ema_short}/{system.ema_long} EMA" for ticker in selected_tickers]
    )
    
    # Add traces for each ticker
    for i, ticker in enumerate(selected_tickers):
        if ticker not in system.stock_data or system.stock_data[ticker].empty:
            # Try to fetch data for this ticker
            current_settings = load_settings()
            
            # Create a temporary system just for this ticker
            temp_system = StockSignalSystem(
                tickers=[ticker],
                ema_short=current_settings['ema_short'],
                ema_long=current_settings['ema_long'],
                min_volume=current_settings['min_volume'],
                data_resolution=current_settings['data_resolution'],
                data_period=current_settings['data_period']
            )
            temp_system.fetch_data()
            
            if temp_system.stock_data and ticker in temp_system.stock_data and not temp_system.stock_data[ticker].empty:
                # Use the newly fetched data
                system.stock_data[ticker] = temp_system.stock_data[ticker]
                # Calculate indicators for this ticker
                if not system._calculate_indicators_for_ticker(ticker):
                    # Skip this ticker if indicators couldn't be calculated
                    logging.warning(f"Failed to calculate indicators for {ticker}")
                    continue
            else:
                # Skip this ticker if we can't get data for it
                logging.warning(f"Could not fetch data for {ticker}")
                continue
            
        # Use the most recent data for this ticker
        data = system.stock_data[ticker].iloc[-days_to_show:]
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name=ticker,
                increasing_line_color='green',
                decreasing_line_color='red',
                showlegend=False
            ),
            row=i+1, col=1
        )
        
        # Add EMAs
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['EMA_short'],
                mode='lines',
                name=f'{system.ema_short} EMA',
                line=dict(color='blue', width=1.5),
                showlegend=(i == 0)  # Only show in legend for first ticker
            ),
            row=i+1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['EMA_long'],
                mode='lines',
                name=f'{system.ema_long} EMA',
                line=dict(color='red', width=1.5),
                showlegend=(i == 0)  # Only show in legend for first ticker
            ),
            row=i+1, col=1
        )
        
        # Add signals specific to this ticker
        ticker_signals = [s for s in system.signals if s['ticker'] == ticker and s['date'] in data.index]
        for signal in ticker_signals:
            marker_color = 'green' if signal['type'] == 'BUY' else 'red'
            marker_symbol = 'triangle-up' if signal['type'] == 'BUY' else 'triangle-down'
            fig.add_trace(
                go.Scatter(
                    x=[signal['date']],
                    y=[signal['price']],
                    mode='markers+text',
                    marker=dict(color=marker_color, size=12, symbol=marker_symbol),
                    text=[signal['type']],
                    textposition='top center',
                    name=f"{ticker} {signal['type']}",
                    showlegend=False
                ),
                row=i+1, col=1
            )
    
    # Update layout
    fig.update_layout(
        title='Stock Comparison - Technical Analysis',
        height=300 * n_tickers,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        xaxis=dict(
            rangeslider=dict(visible=False),
            type='date'
        )
    )
    
    # Update y-axes
    for i in range(1, n_tickers + 1):
        fig.update_yaxes(
            title='Price ($)',
            tickformat='.2f',
            row=i, col=1
        )
    
    # Create plot
    plot_div = plot(fig, output_type='div', include_plotlyjs=True)
    
    watchlist = load_watchlist()
    return render_template(
        'plot.html', 
        plot_div=plot_div, 
        tickers=watchlist,
        selected_tickers=selected_tickers,
        system=system
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)