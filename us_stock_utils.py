import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import requests
import sqlite3
import os

# Path to SQLite database file
DB_PATH = os.path.join(os.path.dirname(__file__), 'investdb.db')

def fetch_stock_data(symbol, start_date, end_date, max_retries=3):
    """Fetch stock data with improved error handling and debugging"""
    db = None
    try:
        db = sqlite3.connect(DB_PATH)
        cursor = db.cursor()
        cursor.execute("""
    CREATE TABLE IF NOT EXISTS us_stock_data (
        symbol VARCHAR(10) NOT NULL,
        Date DATE NOT NULL,
        Close FLOAT,
        Prev_ratio FLOAT,
        Open FLOAT,
        High FLOAT,
        Low FLOAT,
        Volume BIGINT,
        PRIMARY KEY(symbol, Date)
    )
    """
    )
    db.commit()
    cursor.execute("SELECT MAX(Date) FROM us_stock_data WHERE symbol=?", (symbol,))
    last = cursor.fetchone()[0]
    last_date = pd.to_datetime(last) if last else None

    for retry_count in range(max_retries):
        try:
            # Add 1 day to end_date to ensure we include the last day
            adjusted_end_date = end_date + timedelta(days=1)
            
            print(f"Attempt {retry_count+1}: Fetching data for {symbol} from {start_date.date()} to {adjusted_end_date.date()}")
            data = yf.download(
                tickers=symbol,
                start=start_date,
                end=adjusted_end_date,
                progress=False,
                ignore_tz=True,
                prepost=False
            )
            
            if not data.empty and len(data) > 0:
                print(f"Successfully retrieved {len(data)} days of data")
                return data
                
            ticker = yf.Ticker(symbol)
            hist_data = ticker.history(
                start=start_date,
                end=adjusted_end_date,
                interval="1d"
            )
            
            if not hist_data.empty and len(hist_data) > 0:
                print(f"Successfully retrieved {len(hist_data)} days of data via Ticker.history")
                return hist_data
                
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {
                'period1': int(start_date.timestamp()),
                'period2': int(adjusted_end_date.timestamp()),
                'interval': '1d',
                'includePrePost': 'false'
            }
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'application/json',
                'Accept-Encoding': 'gzip, deflate, br'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=15)
            
            if response.status_code == 200:
                try:
                    json_data = response.json()
                    
                    if 'chart' in json_data and 'result' in json_data['chart'] and json_data['chart']['result']:
                        chart_data = json_data['chart']['result'][0]
                        timestamps = pd.to_datetime([datetime.fromtimestamp(ts) for ts in chart_data['timestamp']])
                        quotes = chart_data['indicators']['quote'][0]
                        
                        df = pd.DataFrame({
                            'Open': quotes['open'],
                            'High': quotes['high'],
                            'Low': quotes['low'],
                            'Close': quotes['close'],
                            'Volume': quotes['volume']
                        }, index=timestamps)
                        
                        df = df.dropna()
                        
                        if not df.empty and len(df) > 0:
                            print(f"Successfully retrieved {len(df)} days of data via direct API")
                            return df

                        if not df.empty: # Changed from 'if df:' to 'if not df.empty:'
                            for _, row in df.iterrows():
                                cursor.execute(
                                    "INSERT OR IGNORE INTO us_stock_data (symbol, Date, Close, Prev_ratio, Open, High, Low, Volume) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                    (
                                        symbol, row.name.strftime('%Y-%m-%d'), row['Close'], row.get('Prev_ratio'), # .name for index
                                        row['Open'], row['High'], row['Low'], row['Volume']
                                    )
                                )
                            db.commit()
                            full_df = pd.read_sql_query(
                                "SELECT Date, Close, Prev_ratio, Open, High, Low, Volume FROM us_stock_data WHERE symbol=? ORDER BY Date", 
                                con=db, params=(symbol,)
                            )
                            full_df['Date'] = pd.to_datetime(full_df['Date'])
                            full_df.set_index('Date', inplace=True)
                            return full_df
                        else:
                            if last_date:
                                full_df = pd.read_sql_query(
                                    "SELECT Date, Close, Prev_ratio, Open, High, Low, Volume FROM us_stock_data WHERE symbol=? ORDER BY Date",
                                    con=db, params=(symbol,)
                                )
                                full_df['Date'] = pd.to_datetime(full_df['Date'])
                                full_df.set_index('Date', inplace=True)
                                return full_df # Return existing data if API fails
                            return pd.DataFrame()
                except Exception as api_error:
                    print(f"API parsing error: {str(api_error)}")
                    
        except Exception as e:
            if retry_count == max_retries - 1:
                print(f"Failed to fetch data after {max_retries} attempts: {str(e)}")
                return pd.DataFrame()
            
            wait_time = 2 * (retry_count + 1)
            print(f"Attempt {retry_count+1} failed: {str(e)}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    
    print(f"Could not retrieve data for {symbol} after multiple attempts")
    return pd.DataFrame()
    finally:
        if db:
            db.close()

def get_us_stock_list():
    """us_stock_list 테이블에서 symbol, name을 읽어 리스트로 반환"""
    # This function will be more useful when UI for adding stocks is implemented in Flask.
    # For now, it can return a default list or connect to DB if needed.
    db = None
    try:
        db = sqlite3.connect(DB_PATH)
        cursor = db.cursor()
        cursor.execute("SELECT symbol, name FROM us_stock_list")
        rows = cursor.fetchall()
        if not rows: # Provide a default if empty, useful for initial setup
            return [('AAPL', 'Apple Inc.'), ('TSLA', 'Tesla Inc.')]
        return rows
    except sqlite3.Error as e:
        print(f"Database error in get_us_stock_list: {e}")
        # Fallback to a default list in case of DB error
        return [('AAPL', 'Apple Inc.'), ('TSLA', 'Tesla Inc.'), ('MSFT', 'Microsoft Corp.')]
    finally:
        if db:
            db.close()


def add_us_stock(symbol, name):
    """us_stock_list 테이블에 symbol, name을 추가"""
    # This function will be more useful when UI for adding stocks is implemented in Flask.
    db = None
    try:
        db = sqlite3.connect(DB_PATH)
        cursor = db.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO us_stock_list (symbol, name) VALUES (?, ?)",
            (symbol, name)
        )
        db.commit()
        print(f"Added/updated stock: {symbol} - {name}")
        return True
    except sqlite3.Error as e:
        print(f"Database error in add_us_stock: {e}")
        return False
    finally:
        if db:
            db.close()

def calculate_moving_averages(data):
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    data['Volume_MA20'] = data['Volume'].rolling(window=20).mean()
    data['Volume_MA50'] = data['Volume'].rolling(window=50).mean()
    return data

def calculate_bollinger_bands(data, window=20, num_std=2):
    data['SMA'] = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    data['Upper_Band'] = data['SMA'] + (rolling_std * num_std)
    data['Lower_Band'] = data['SMA'] - (rolling_std * num_std)
    data['BB_Width'] = (data['Upper_Band'] - data['Lower_Band']) / data['MA20'] # Ensure MA20 is calculated first
    return data

def calculate_rsi(data, window=14):
    if data.empty:
        return data
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean() # Added min_periods=1
    avg_loss = loss.rolling(window=window, min_periods=1).mean() # Added min_periods=1
    
    # Prevent division by zero if avg_loss is 0
    rs = avg_gain / avg_loss.replace(0, pd.NA) # Replace 0 with NA to avoid inf, then handle NA
    
    data['RSI'] = 100 - (100 / (1 + rs))
    data['RSI'] = data['RSI'].fillna(50) # Fill NA RSI values (e.g. if avg_loss was 0) with 50 (neutral)

    return data

def calculate_stochastic(data, k_period=14, d_period=3, slowing=3):
    if data.empty:
        return data
    lowest_low = data['Low'].rolling(window=k_period).min()
    highest_high = data['High'].rolling(window=k_period).max()
    data['%K'] = 100 * ((data['Close'] - lowest_low) / (highest_high - lowest_low))
    if slowing > 1:
        data['%K'] = data['%K'].rolling(window=slowing).mean()
    data['%D'] = data['%K'].rolling(window=d_period).mean()
    return data

def calculate_daily_change(data):
    if data.empty:
        return data
    data['Daily_Change'] = data['Close'] - data['Open']
    data['Prev_ratio'] = data['Close'].pct_change()
    data['Daily_Change_MA20'] = data['Daily_Change'].rolling(window=20).mean()
    return data

def create_stock_visualization(stock_data, symbol, display_options):
    num_rows = 2
    subplot_titles = ['Price with Indicators', 'Volume']
    indicator_rows = {}

    if display_options.get('bb_width', False):
        num_rows += 1
        indicator_rows['bb_width'] = num_rows
        subplot_titles.append('Bollinger Band Width')
    if display_options.get('daily_change', False):
        num_rows += 1
        indicator_rows['daily_change'] = num_rows
        subplot_titles.append('Daily Change')
    if display_options.get('rsi', False):
        num_rows += 1
        indicator_rows['rsi'] = num_rows
        subplot_titles.append('RSI (14)')
    if display_options.get('stochastic', False):
        num_rows += 1
        indicator_rows['stochastic'] = num_rows
        subplot_titles.append('Stochastic (14,3)')

    # Add row for monthly difference graph - this was in the original, keeping for now
    # num_rows += 1
    # diff_row = num_rows # This variable was not used later, commenting out for now.
    # subplot_titles.append('Monthly Difference')

    row_heights_map = {
        2: [0.7, 0.3], 3: [0.6, 0.2, 0.2], 4: [0.5, 0.17, 0.17, 0.16],
        5: [0.4, 0.15, 0.15, 0.15, 0.15], 6: [0.35, 0.13, 0.13, 0.13, 0.13, 0.13]
    }
    row_heights = row_heights_map.get(num_rows, [0.7/num_rows]*num_rows)
    if num_rows > 6 : row_heights[0] = 0.3


    fig = make_subplots(
        rows=num_rows, cols=1, shared_xaxes=True, vertical_spacing=0.1,
        subplot_titles=subplot_titles, row_heights=row_heights
    )

    fig.add_trace(
        go.Candlestick(
            x=stock_data.index, open=stock_data['Open'], high=stock_data['High'],
            low=stock_data['Low'], close=stock_data['Close'],
            increasing=dict(line=dict(color='red'), fillcolor='red'),
            decreasing=dict(line=dict(color='blue'), fillcolor='blue'),
            name='Price'
        ), row=1, col=1
    )
    
    # Monthly min/max markers (optional, can be controlled by display_options if needed)
    # monthly = stock_data.resample('M') # 'M' is deprecated, use 'ME' for month end or 'MS' for month start
    monthly_resample = stock_data.resample('ME')

    if not monthly_resample.empty:
        min_idx = monthly_resample['Low'].idxmin()
        max_idx = monthly_resample['High'].idxmax()
        # Ensure indices are valid and present in stock_data
        valid_min_idx = min_idx.dropna()
        valid_max_idx = max_idx.dropna()

        if not valid_min_idx.empty:
            monthly_low = stock_data.loc[stock_data.index.intersection(valid_min_idx)]
            if not monthly_low.empty:
                fig.add_trace(go.Scatter(
                    x=monthly_low.index, y=monthly_low['Low'], mode='markers+text',
                    marker=dict(color='green', size=8), text=monthly_low.index.strftime('%m-%d'),
                    textposition='bottom center', name='Monthly Low'
                ), row=1, col=1)
        
        if not valid_max_idx.empty:
            monthly_high = stock_data.loc[stock_data.index.intersection(valid_max_idx)]
            if not monthly_high.empty:
                fig.add_trace(go.Scatter(
                    x=monthly_high.index, y=monthly_high['High'], mode='markers+text',
                    marker=dict(color='black', size=8), text=monthly_high.index.strftime('%m-%d'),
                    textposition='top center', name='Monthly High'
                ), row=1, col=1)

    ma_colors = {
        'MA5': 'rgba(255, 0, 0, 0.7)', 'MA20': 'rgba(255, 165, 0, 0.7)',
        'MA50': 'rgba(0, 0, 255, 0.7)', 'MA200': 'rgba(128, 0, 128, 0.7)'
    }
    for ma_type, color in ma_colors.items():
        if ma_type in stock_data.columns and display_options.get(ma_type, False):
            fig.add_trace(
                go.Scatter(x=stock_data.index, y=stock_data[ma_type], line=dict(color=color, width=1), name=ma_type),
                row=1, col=1
            )

    if display_options.get('bollinger', False) and 'Upper_Band' in stock_data.columns and 'Lower_Band' in stock_data.columns:
        fig.add_trace(
            go.Scatter(x=stock_data.index, y=stock_data['Upper_Band'], line=dict(color='rgba(34, 139, 34, 0.5)', width=1), name='Upper Band'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=stock_data.index, y=stock_data['Lower_Band'], line=dict(color='rgba(34, 139, 34, 0.5)', width=1),
                         fill='tonexty', fillcolor='rgba(34, 139, 34, 0.1)', name='Lower Band'),
            row=1, col=1
        )

    vol_colors = ['red' if c >= o else 'blue' for o, c in zip(stock_data['Open'], stock_data['Close'])]
    fig.add_trace(
        go.Bar(x=stock_data.index, y=stock_data['Volume'], marker=dict(color=vol_colors), name='Volume'),
        row=2, col=1
    )

    volume_ma_colors = {'Volume_MA20': 'rgba(255, 165, 0, 0.9)', 'Volume_MA50': 'rgba(0, 0, 255, 0.9)'}
    for ma_type, color in volume_ma_colors.items():
        if ma_type in stock_data.columns and display_options.get(ma_type, False) : # Added display_options check
            fig.add_trace(
                go.Scatter(x=stock_data.index, y=stock_data[ma_type], line=dict(color=color, width=2, dash='dot'), mode='lines', name=ma_type),
                row=2, col=1
            )
    
    if display_options.get('bb_width', False) and 'BB_Width' in stock_data.columns:
        fig.add_trace(
            go.Scatter(x=stock_data.index, y=stock_data['BB_Width'], line=dict(color='rgba(75, 0, 130, 0.8)', width=1.5),
                         name='BB Width', fill='tozeroy', fillcolor='rgba(75, 0, 130, 0.1)'),
            row=indicator_rows['bb_width'], col=1
        )

    if display_options.get('rsi', False) and 'RSI' in stock_data.columns:
        rsi_row = indicator_rows['rsi']
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], line=dict(color='blue', width=1.5), name='RSI (14)'), row=rsi_row, col=1)
        fig.add_shape(type="line", x0=stock_data.index[0], y0=70, x1=stock_data.index[-1], y1=70, line=dict(color="red", width=1, dash="dash"), row=rsi_row, col=1)
        fig.add_shape(type="line", x0=stock_data.index[0], y0=30, x1=stock_data.index[-1], y1=30, line=dict(color="green", width=1, dash="dash"), row=rsi_row, col=1)
        fig.add_shape(type="line", x0=stock_data.index[0], y0=50, x1=stock_data.index[-1], y1=50, line=dict(color="gray", width=1, dash="dot"), row=rsi_row, col=1)
        fig.update_yaxes(range=[0, 100], row=rsi_row, col=1)


    if display_options.get('daily_change', False) and 'Daily_Change' in stock_data.columns:
        daily_change_row = indicator_rows['daily_change']
        colors = ['green' if x >= 0 else 'red' for x in stock_data['Daily_Change']]
        fig.add_trace(go.Bar(x=stock_data.index, y=stock_data['Daily_Change'], marker_color=colors, name='Daily Change'), row=daily_change_row, col=1)
        if 'Daily_Change_MA20' in stock_data.columns:
             fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Daily_Change_MA20'], line=dict(color='blue', width=1, dash='dash'), name='Daily Change MA(20)'), row=daily_change_row, col=1)
        fig.add_shape(type="line", x0=stock_data.index[0], y0=0, x1=stock_data.index[-1], y1=0, line=dict(color="black", width=1), row=daily_change_row, col=1)

    if display_options.get('stochastic', False) and '%K' in stock_data.columns and '%D' in stock_data.columns:
        stochastic_row = indicator_rows['stochastic']
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['%K'], line=dict(color='blue', width=1.5), name='%K'), row=stochastic_row, col=1)
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['%D'], line=dict(color='red', width=1.5, dash='dash'), name='%D'), row=stochastic_row, col=1)
        fig.add_shape(type="line", x0=stock_data.index[0], y0=80, x1=stock_data.index[-1], y1=80, line=dict(color="red", width=1, dash="dash"), row=stochastic_row, col=1)
        fig.add_shape(type="line", x0=stock_data.index[0], y0=20, x1=stock_data.index[-1], y1=20, line=dict(color="green", width=1, dash="dash"), row=stochastic_row, col=1)
        fig.add_shape(type="line", x0=stock_data.index[0], y0=50, x1=stock_data.index[-1], y1=50, line=dict(color="gray", width=1, dash="dot"), row=stochastic_row, col=1)

    fig.update_layout(
        title=f'{symbol} Stock Data with Technical Indicators',
        xaxis_rangeslider_visible=False,
        height=max(600, num_rows * 250), # Dynamic height
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', dtick='M1', tickformat='%b\n%Y'),
        xaxis2=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', dtick='M1', tickformat='%b\n%Y')
    )
    if num_rows >= 3:
        fig.update_layout(xaxis3=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', dtick='M1', tickformat='%b\n%Y'))
    if num_rows >= 4: # Corrected from num_rows == 4
        fig.update_layout(xaxis4=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', dtick='M1', tickformat='%b\n%Y'))
    if num_rows >= 5:
        fig.update_layout(xaxis5=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', dtick='M1', tickformat='%b\n%Y'))
    if num_rows >= 6:
        fig.update_layout(xaxis6=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', dtick='M1', tickformat='%b\n%Y'))


    fig.update_xaxes(tickformat='%m-%d', tickangle=-45)

    # Dividend highlighting
    db_div = None
    try:
        db_div = sqlite3.connect(DB_PATH)
        cur_div = db_div.cursor()
        start_dt = stock_data.index.min().date()
        end_dt = stock_data.index.max().date()
        cur_div.execute(
            "SELECT ex_date, pay_date FROM dividends WHERE symbol=? AND ex_date BETWEEN ? AND ?",
            (symbol, start_dt, end_dt)
        )
        for ex_date_str, pay_date_str in cur_div.fetchall():
            # Ensure dates are valid before adding vrect
            try:
                ex_date = pd.to_datetime(ex_date_str).date()
                pay_date = pd.to_datetime(pay_date_str).date()
                if ex_date < pay_date : # Basic validation
                    fig.add_vrect(
                        x0=ex_date, x1=pay_date,
                        fillcolor='yellow', opacity=0.3,
                        layer='below', line_width=0,
                        row=1, col=1
                    )
            except Exception as e_date:
                print(f"Error processing dividend date: {e_date} for {ex_date_str}, {pay_date_str}")
    except Exception as e:
        print(f"Could not load dividend ranges: {e}")
    finally:
        if db_div:
            db_div.close()

    return fig.to_html(full_html=False, include_plotlyjs='cdn')
