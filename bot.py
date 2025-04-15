import os
import sys
import datetime
import pandas as pd

from binance.client import Client
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QGroupBox, QHBoxLayout
from PyQt5.QtCore import QTimer, Qt
from decimal import Decimal, ROUND_DOWN

# Global variable to hold the main window instance
window_instance = None

TESTNET = False

# Load environment variables for Binance API credentials
from dotenv import load_dotenv
if TESTNET:
    load_dotenv('testnet.env')
else:
    load_dotenv('binance.env')

API_KEY = os.environ.get("BINANCE_API_KEY")
API_SECRET = os.environ.get("BINANCE_API_SECRET")

print("API Key:", API_KEY)
print("API Secret:", API_SECRET)

client = Client(API_KEY, API_SECRET, testnet=TESTNET)

def get_historical_data(symbol="XRPUSDT", interval=Client.KLINE_INTERVAL_5MINUTE, lookback="1 day ago UTC"):
    """
    Fetch historical candlestick data from Binance.
    """
    klines = client.get_historical_klines(symbol, interval, lookback)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'num_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['close'] = pd.to_numeric(df['close'])
    return df

def calculate_emas(df, span1=10, span2=20, span3=50):
    """
    Calculate the 10 and 20 period EMAs on closing prices.
    """
    df['ema10'] = df['close'].ewm(span=span1, adjust=False).mean()
    df['ema20'] = df['close'].ewm(span=span2, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=span3, adjust=False).mean()
    return df

import numpy as np  # make sure numpy is imported at the top

def generate_signal(df, coin, bought_price=None, lookback=3, slope_angle_threshold=0.04):
    """
    Generate a trading signal based on the following strategy:
    - Buy when EMA10 > EMA20, the most recent close is above EMA10, and the 
      EMA10 trend (slope over the last 5 values) is upward by at least 10°.
    - Sell when the most recent close is below EMA10.
    - Hold in all other cases.
    """
    # Need at least 5 data points to calculate a meaningful slope.
    if df.empty or len(df) < lookback:
        return 'hold'
    
    # Get the last 5 EMA10 values and compute the slope using linear regression.
    recent = df.tail(lookback)
    try:
        ema10_vals = recent['ema10'].astype(float).values
    except Exception as e:
        print("Error extracting EMA10 values:", e)
        return 'hold'
    
    # Convert ema10_vals to a NumPy array for element-wise operations
    ema10_vals = np.array(ema10_vals)

    # Compute percentage changes from the first EMA10 value in this window.
    base = ema10_vals[0]
    pct_changes = (ema10_vals - base) / base  # relative changes

    # Use indices as x values (0, 1, ..., 4)
    x = np.arange(len(pct_changes))
    slope = np.polyfit(x, pct_changes, 1)[0]  # slope in % change per time unit
    # Convert the slope to an angle (in degrees)
    slope_angle = np.degrees(np.arctan(slope))
    
    latest = df.iloc[-1]
    close = latest.get('close')
    ema10 = latest.get('ema10')
    ema20 = latest.get('ema20')
    ema50 = latest.get('ema50')

    if close is None or ema10 is None or ema20 is None or ema50 is None or pd.isna(close) or pd.isna(ema10) or pd.isna(ema20):
        return 'hold'
    
    close = float(close)
    ema10 = float(ema10)
    ema20 = float(ema20)
    ema50 = float(ema50)
    
    if bought_price is not None:
        print(f"{coin}: Bought Price: {bought_price:.3f}, Close: {close:.3f}, EMA10: {ema10:.3f}, EMA20: {ema20:.3f}, EMA50: {ema50:.3f}, Slope Angle: {slope_angle:.3f}")
    else:
        print(f"{coin}: Close: {close:.3f}, EMA10: {ema10:.3f}, EMA20: {ema20:.3f}, EMA50: {ema50:.3f}, Slope Angle: {slope_angle:.3f}")

    # Strategy:
    # Buy if EMA10 > EMA20, close > EMA10, and EMA10 trend angle > 5 degrees.
    if ema10 > ema20 and ema20 > ema50 and close > ema10 and slope_angle > slope_angle_threshold:
        return 'buy'
    # Sell if the most recent close is below EMA10.
    elif close < ema20:
        return 'sell'
    elif bought_price is not None and close < bought_price * 0.99:
        return 'sell'
    else:
        return 'hold'
    
def adjust_for_fee(amount, fee_rate=0.005):
    """
    Adjust the trade amount to account for a 0.5% transaction fee.
    """
    return amount * (1 - fee_rate)

def get_current_price(symbol):
    """
    Get the current price for a given symbol from Binance.
    """
    try:
        ticker = client.get_symbol_ticker(symbol=symbol)
        if ticker and 'price' in ticker:
            return float(ticker['price'])
    except Exception as e:
        print(f"Error fetching price for {symbol}: {e}")
    return 0.0  # Always return a float

def adjust_quantity_to_lot_size(symbol, quantity):
    """
    Adjust the given quantity to conform to the LOT_SIZE filter for the symbol.
    This version uses the symbol's 'quantityPrecision' if available, and falls back
    to using the normalized stepSize to determine allowed decimals.
    """
    info = client.get_symbol_info(symbol)
    if 'quantityPrecision' in info:
        precision = info['quantityPrecision']
        quantizer = Decimal('1e-{}'.format(precision))
        adjusted_quantity = Decimal(str(quantity)).quantize(quantizer, rounding=ROUND_DOWN)
        for f in info['filters']:
            if f['filterType'] == 'LOT_SIZE':
                min_qty = Decimal(f['minQty'])
                if adjusted_quantity < min_qty:
                    adjusted_quantity = min_qty
                break
        return format(adjusted_quantity, 'f')
    else:
        step_size = None
        min_qty = None
        for f in info['filters']:
            if f['filterType'] == 'LOT_SIZE':
                step_size = f['stepSize']
                min_qty = f['minQty']
                break
        if step_size is None:
            return str(quantity)
        normalized_step = Decimal(step_size).normalize()
        allowed_decimals = abs(normalized_step.as_tuple().exponent)
        quantizer = Decimal('1e-{}'.format(allowed_decimals))
        adjusted_quantity = Decimal(str(quantity)).quantize(quantizer, rounding=ROUND_DOWN)
        min_qty_decimal = Decimal(min_qty)
        if adjusted_quantity < min_qty_decimal:
            adjusted_quantity = min_qty_decimal
        fmt = "{:0." + str(allowed_decimals) + "f}"
        return fmt.format(adjusted_quantity)

def execute_trade(signal, trade_amount, coin, fee_rate=0.005):
    """
    Execute a market trade for the specified coin based on the signal.
    For buy orders, trade_amount is the USDT allocation.
    For sell orders, trade_amount is the coin quantity.
    Ensures that the trade meets Binance's minimum notional requirement,
    using a fallback minimum if the API returns 0.0.
    Uses the global window_instance to update portfolio and coin labels.
    Returns True if the trade was executed successfully, False otherwise.
    """
    symbol = coin + "USDT"
    info = client.get_symbol_info(symbol)
    min_notional = None
    for f in info['filters']:
        if f['filterType'] == 'MIN_NOTIONAL':
            min_notional = float(f['minNotional'])
            break
    # Use fallback minimum if API returns None or 0.0.
    fallback_min_notional = min_notional if min_notional and min_notional > 0.0 else 10.0

    current_price = get_current_price(symbol)
    if current_price == 0.0:
        print(f"Skipping trade for {coin} due to invalid price.")
        return False

    if signal == 'buy':
        # For buy orders, trade_amount is in USDT allocation.
        coin_qty = (trade_amount / current_price) * (1 - fee_rate)  # fee adjustment
        notional_value = trade_amount  # USDT allocation
        # Check if the allocation is below the fallback minimum.
        if trade_amount < fallback_min_notional:
            print(f"Skipping BUY trade for {coin}: Allocation {trade_amount:.2f} USDT is below fallback minimum notional {fallback_min_notional:.2f} USDT.")
            return False
    else:
        # For sell orders, trade_amount is the coin quantity.
        coin_qty = trade_amount
        notional_value = coin_qty * current_price

    if notional_value < fallback_min_notional:
        print(f"Skipping {signal.upper()} trade for {coin}: Notional value {notional_value:.2f} USDT is below fallback minimum notional {fallback_min_notional:.2f} USDT.")
        return False

    try:
        if signal == 'buy':
            coin_qty_str = adjust_quantity_to_lot_size(symbol, coin_qty)
            client.order_market_buy(symbol=symbol, quantity=coin_qty_str)
            window_instance.coin_labels[coin].setStyleSheet("color: red;")
        elif signal == 'sell':
            quantity_str = adjust_quantity_to_lot_size(symbol, coin_qty)
            print(quantity_str)
            client.order_market_sell(symbol=symbol, quantity=quantity_str)
            window_instance.coin_labels[coin].setStyleSheet("color: green;")
        else:
            print("No trade executed.")
            return False
    except Exception as e:
        print(f"Error executing trade for {coin}: {e}")
        return False

    if signal == 'buy':
        trade_entry = f"{signal.upper()} {coin}: {trade_amount:.1f} USDT allocated, price {current_price:.4f}"
    else:
        trade_entry = f"{signal.upper()} {coin}: {coin_qty:.4f} coins sold, price {current_price:.4f}"
    window_instance.trade_history.insert(0, trade_entry)
    if len(window_instance.trade_history) > 10:
        window_instance.trade_history.pop()
    for i, trade in enumerate(window_instance.trade_history):
        window_instance.history_labels[i].setText(trade)
    return True

class TradingBotWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        global window_instance
        window_instance = self  # Set the global instance reference

        self.trade_history = []  # Initialize trade history list

        self.setWindowTitle("Crypto Trading Bot")
        self.setWindowModality(Qt.ApplicationModal)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        self.current_state = "USDT"  # Default state
        self.TARGET_COINS = ["ETH", "ADA", "SOL", "XRP", "XLM", "BTC", "BNB", "DOGE", "TRX", "WBTC", "LINK", "AVAX", "SUI", "HBAR"]
        self.portfolio = {}
        self.bought_price = {coin: None for coin in self.TARGET_COINS}
        
        # Load actual balances for target coins
        for coin in self.TARGET_COINS:
            try:
                data = client.get_asset_balance(asset=coin)
                self.portfolio[coin] = float(data['free']) if data and 'free' in data else 0.0
            except Exception as e:
                print(f"Error fetching balance for {coin}: {e}")
                self.portfolio[coin] = 0.0

        # Load USDT balance
        try:
            data = client.get_asset_balance(asset="USDT")
            self.portfolio["USDT"] = float(data['free']) if data and 'free' in data else 1000.0
        except Exception as e:
            print("Error fetching balance for USDT:", e)
            self.portfolio["USDT"] = 1000.0

        # Set baseline as the current total portfolio value
        self.initial_investment = self.calculate_portfolio_value()

        # Initialize last buy price and last sell time for each coin
        self.last_buy_price = {coin: None for coin in self.TARGET_COINS}
        self.last_sell_time = {coin: None for coin in self.TARGET_COINS}

        # Overview Box: Display Total Asset and Total Gain
        self.overview_box = QGroupBox("Overview")
        # overview_header = QLabel("Overview")
        # overview_header.setStyleSheet("font-weight: bold; font-size: 18px;")
        self.total_asset_label = QLabel("Total Asset: 0 USDT", self)
        self.gain_label = QLabel("Total Gain: 0 USDT", self)
        overview_layout = QVBoxLayout()
        # overview_layout.addWidget(overview_header)
        overview_layout.addWidget(self.total_asset_label)
        overview_layout.addWidget(self.gain_label)
        overview_layout.setAlignment(Qt.AlignTop)
        self.overview_box.setLayout(overview_layout)

        # Portfolio Box: Display each coin's status
        self.portfolio_box = QGroupBox("Portfolio")
        # portfolio_header = QLabel("Portfolio")
        # portfolio_header.setStyleSheet("font-weight: bold; font-size: 18px;")
        self.coin_labels = {coin: QLabel(f"{coin}: {self.portfolio[coin]:.4f}") for coin in self.TARGET_COINS}
        portfolio_layout = QVBoxLayout()
        # portfolio_layout.addWidget(portfolio_header)
        for coin, label in self.coin_labels.items():
            portfolio_layout.addWidget(label)
        portfolio_layout.setAlignment(Qt.AlignTop)
        self.portfolio_box.setLayout(portfolio_layout)

        # Trade History Box: Display last 10 executed trades with a header
        self.history_box = QGroupBox("Trade History")
        # history_header = QLabel("Trade History")
        # history_header.setStyleSheet("font-weight: bold; font-size: 18px;")
        self.history_labels = [QLabel("-") for _ in range(10)]
        history_layout = QVBoxLayout()
        # history_layout.addWidget(history_header)
        for label in self.history_labels:
            history_layout.insertWidget(0, label)
        history_layout.setAlignment(Qt.AlignTop)
        self.history_box.setLayout(history_layout)

        # Main Layout: Arrange boxes horizontally
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.overview_box)
        main_layout.addWidget(self.portfolio_box)
        main_layout.addWidget(self.history_box)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.setStyleSheet("""
            QMainWindow {
                background-color: #f4f4f4;
            }
            QGroupBox {
                background-color: #ffffff;
                border: 2px solid #ccc;
                border-radius: 8px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
                font-size: 18px;
                font-weight: bold;
                color: #444;
            }
            QLabel {
                font-family: 'Arial', 'Helvetica', sans-serif;
                font-size: 16px;
                color: #333;
                padding: 4px;
            }
        """)

        # Use the current portfolio value as the baseline for gain calculation.
        # self.initial_investment is already set above.

        # Setup a QTimer to update every 5 minutes (300000 ms); adjust for testing if needed.
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_bot)
        self.timer.start(600000)

        # Initial update
        self.update_bot()

    def calculate_portfolio_value(self):
        total_value = self.portfolio["USDT"]
        for coin in self.TARGET_COINS:
            try:
                price = get_current_price(coin + "USDT")
                total_value += self.portfolio[coin] * price if price is not None else 0
            except Exception as e:
                print(f"Error fetching price for {coin}: ", e)
        return total_value

    def update_bot(self, target_allocation_limit=0.4, fee_rate=0.005):
        try:
            total_value = self.portfolio["USDT"]

            # Build a list of (coin, coin_signal, df_coin) for each coin.
            coin_signal_data = []
            for coin in self.TARGET_COINS:
                symbol = coin + "USDT"
                df_coin = get_historical_data(symbol=symbol, interval=Client.KLINE_INTERVAL_5MINUTE, lookback="1 day ago UTC")
                df_coin = calculate_emas(df_coin, span1=10, span2=20)
                coin_signal = generate_signal(df_coin, coin, bought_price=self.bought_price[coin])
                coin_signal_data.append((coin, coin_signal, df_coin))
            
            # Define sort order: sell signals first, then hold, then buy.
            def sort_key(item):
                coin, signal, _ = item
                if signal == "sell":
                    return 0
                elif signal == "hold":
                    return 1
                elif signal == "buy":
                    return 2
                return 3
            coin_signal_data.sort(key=sort_key)
            
            # Process coins in the sorted order.
            for coin, coin_signal, df_coin in coin_signal_data:
                symbol = coin + "USDT"
                # Print coin signal info
                if not df_coin.empty:
                    ema10_val = float(df_coin.iloc[-1]['ema10'])
                    ema20_val = float(df_coin.iloc[-1]['ema20'])
                    ema50_val = float(df_coin.iloc[-1]['ema50'])
                    close_val = float(df_coin.iloc[-1]['close'])
                    print(f"{coin} Signal: {coin_signal} | EMA10: {ema10_val:.4f} | EMA20: {ema20_val:.4f} | EMA50: {ema50_val:.4f} | Close: {close_val:.4f}")
                else:
                    print(f"{coin} - No data available; Signal: {coin_signal}")
                
                price = get_current_price(symbol)
                if price is None or price == 0.0:
                    print(f"Skipping {coin} due to invalid price.")
                    continue

                total_value += self.portfolio[coin] * price
                self.coin_labels[coin].setText(f"{coin}: {self.portfolio[coin]:.4f}")
                
                # Extra condition: If the coin signal is "buy" but the most recent candle's close is below the 10 EMA, skip buy.
                if coin_signal == "buy":
                    latest_close = float(df_coin.iloc[-1]['close'])
                    latest_ema10 = float(df_coin.iloc[-1]['ema10'])
                    if latest_close < latest_ema10:
                        print(f"Skipping buy for {coin} because the latest close ({latest_close:.4f}) is below the 10 EMA ({latest_ema10:.4f}).")
                        continue

                # Check signals for the coin:
                if coin_signal == "buy":
                    # Check if 15 minutes have passed since the last sell for this coin.
                    now = datetime.datetime.now()
                    if self.last_sell_time.get(coin) is not None:
                        elapsed = now - self.last_sell_time[coin]
                        if elapsed.total_seconds() < 15 * 60:
                            print(f"Buy cooldown active for {coin}: Only {elapsed.total_seconds()/60:.2f} minutes since last sell. Skipping buy.")
                            continue
                    # Then, proceed with your existing buy logic...
                    # Calculate current value of the coin holding in USDT
                    current_coin_value = self.portfolio[coin] * price
                    if total_value > 0 and (current_coin_value / total_value) >= target_allocation_limit:
                        print(f"{coin} already meets the target allocation ({(current_coin_value/total_value)*100:.2f}%). Skipping buy.")
                    else:
                        target_allocation = target_allocation_limit * total_value
                        additional_needed = target_allocation - current_coin_value
                        allocation = min(additional_needed, self.portfolio["USDT"])
                        allocation = round(allocation)  # Round allocation to whole number
                        # Check if allocation meets the minimum notional requirement.
                        info = client.get_symbol_info(coin + "USDT")
                        min_notional = 0.0
                        for f in info['filters']:
                            if f['filterType'] == 'MIN_NOTIONAL':
                                min_notional = float(f['minNotional'])
                                break
                        if allocation < min_notional:
                            print(f"Skipping buy for {coin}: allocation {allocation} USDT is below minimum notional {min_notional:.2f} USDT")
                            continue
                        if allocation > 0:
                            usdt_data = client.get_asset_balance(asset="USDT")
                            actual_usdt_balance = float(usdt_data["free"]) if usdt_data and "free" in usdt_data else 0.0
                            if allocation > actual_usdt_balance:
                                print(f"Insufficient USDT for buying {coin}: allocation {allocation} USDT exceeds actual balance {actual_usdt_balance} USDT. Skipping trade.")
                            else:
                                success = execute_trade("buy", allocation, coin)
                                if success:
                                    coin_amount = allocation / price * (1 - fee_rate)
                                    self.portfolio[coin] += coin_amount
                                    self.bought_price[coin] = price
                                    self.portfolio["USDT"] -= allocation
                                    # Optionally update last_buy_time as well if needed.
                                    print(f"Buy executed for {coin}: Allocated {allocation} USDT, coin amount: {coin_amount:.4f}")
                                else:
                                    print(f"Buy trade for {coin} failed.")
                        else:
                            print(f"No available USDT for additional buy of {coin}.")
                
                elif coin_signal == "sell" and self.portfolio[coin] > 0:
                    coin_amount = self.portfolio[coin] * 0.99
                    price = get_current_price(coin + "USDT")
                    if price is None or price == 0.0:
                        print(f"Skipping {coin} due to invalid price.")
                        continue

                    # Check if the value of the holding is less than 25 USDT; if so, skip selling.
                    if coin_amount * price < 25:
                        print(f"Skipping sell for {coin} because holding value {coin_amount * price:.2f} USDT is below 25 USDT.")
                        continue

                    # Retrieve the minimum notional requirement for this coin pair.
                    info = client.get_symbol_info(coin + "USDT")
                    min_notional = 0.0
                    for f in info['filters']:
                        if f['filterType'] == 'MIN_NOTIONAL':
                            min_notional = float(f['minNotional'])
                            break

                    # Adjust the coin amount to meet lot size requirements.
                    quantity_str = adjust_quantity_to_lot_size(coin + "USDT", coin_amount)
                    adjusted_qty = float(quantity_str)
                    # Ensure adjusted quantity does not exceed your actual holding.
                    if adjusted_qty > coin_amount:
                        adjusted_qty = coin_amount
                    # Reformat the adjusted quantity (optional – you may call adjust_quantity_to_lot_size again)
                    quantity_str = adjust_quantity_to_lot_size(coin + "USDT", adjusted_qty)
                    notional_value_adjusted = float(quantity_str) * price
                    if notional_value_adjusted < min_notional:
                        print(f"Not selling {coin} because adjusted notional value {notional_value_adjusted:.2f} USDT is below minimum {min_notional:.2f} USDT")
                        continue

                    # Execute sell trade using the adjusted quantity.
                    print(coin_amount, quantity_str, adjusted_qty)
                    success = execute_trade("sell", adjusted_qty, coin)
                    if success:
                        usdt_received = adjusted_qty * price * (1 - fee_rate)
                        self.portfolio["USDT"] += usdt_received
                        self.portfolio[coin] = max(0.0, self.portfolio[coin] - adjusted_qty)
                        self.bought_price[coin] = None
                        self.last_sell_time[coin] = datetime.datetime.now()  # update sell timestamp
                        print(f"Sell executed for {coin}: Received {usdt_received:.2f} USDT")
                    else:
                        print(f"Sell trade for {coin} failed. Portfolio remains unchanged.")

                # Set label color: red if coin is held, green if not.
                if self.portfolio[coin] > 0:
                    self.coin_labels[coin].setStyleSheet("color: red;")
                else:
                    self.coin_labels[coin].setStyleSheet("color: green;")

            self.total_asset_label.setText(f"Total Asset: {total_value:.2f} USDT")
            gain = total_value - self.initial_investment
            self.gain_label.setText(f"Total Gain: {gain:.2f} USDT")

            print(f"{datetime.datetime.now()}: Total Asset = {total_value:.2f} USDT, Gain = {gain:.2f} USDT")
        except Exception as e:
            print("Error in update_bot:", e)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TradingBotWindow()
    window.show()
    sys.exit(app.exec_())
