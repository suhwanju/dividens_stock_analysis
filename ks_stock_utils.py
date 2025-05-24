import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import requests
from bs4 import BeautifulSoup
import sqlite3
import os

# Path to SQLite database file
DB_PATH = os.path.join(os.path.dirname(__file__), 'investdb.db')

def fetch_korea_stock_data(stock_code, max_pages=30): # Added max_pages to limit scraping
    """
    Fetches Korean stock data from Naver Finance using scraping.
    Uses SQLite for incremental updates.
    Returns a Pandas DataFrame or an empty DataFrame on error.
    """
    print(f"Fetching Korean stock data for: {stock_code}")
    db = None
    try:
        db = sqlite3.connect(DB_PATH)
        cursor = db.cursor()
        cursor.execute("""
    CREATE TABLE IF NOT EXISTS ks_stock_data (
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
    """)
    db.commit()
    
    cursor.execute("SELECT MAX(Date) FROM ks_stock_data WHERE symbol=?", (stock_code,))
    last_db_date_str = cursor.fetchone()[0]
    last_db_date = pd.to_datetime(last_db_date_str) if last_db_date_str else None
    print(f"Last date in DB for {stock_code}: {last_db_date}")

    base_url = f'https://finance.naver.com/item/sise_day.nhn?code={stock_code}'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    df_list = []
    
    # Determine how many pages to scrape. If last_db_date is recent, we might not need many.
    # Naver shows 10 items per page.
    # If last_db_date is None, scrape more pages.
    # If last_db_date is very old, scrape more pages.
    # If last_db_date is recent, scrape fewer.
    
    # For simplicity, still using max_pages, but the break condition is important.
    
    pages_to_scrape = max_pages
    if last_db_date and (datetime.now() - last_db_date).days < 30 : # If data is less than 30 days old
        pages_to_scrape = 5 # Scrape fewer pages, e.g. enough for ~1 month of data + buffer
        print(f"Recent data found. Reducing pages to scrape to {pages_to_scrape} for {stock_code}.")


    for page in range(1, pages_to_scrape + 1):
        url = f'{base_url}&page={page}'
        try:
            response = requests.get(url, headers=headers, timeout=10) # Added timeout
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table')
            
            if table:
                # Important: Naver's HTML structure might require more specific parsing
                # For example, finding the correct table if multiple exist.
                # pd.read_html might fail if the table structure is not standard.
                temp_df_list = pd.read_html(str(table), header=0)
                if not temp_df_list or temp_df_list[0].empty:
                    print(f"No table data found on page {page} for {stock_code}. Might be last page.")
                    break 
                
                df = temp_df_list[0].dropna()
                
                # Check for expected columns, adjust if Naver changes format
                expected_cols = ['날짜', '종가', '전일비', '시가', '고가', '저가', '거래량']
                if not all(col in df.columns for col in ['날짜', '종가', '시가', '고가', '저가', '거래량']):
                    print(f"Page {page} for {stock_code}: Table format unexpected. Columns: {df.columns.tolist()}")
                    # Potentially skip this page or try to map columns if possible
                    continue

                df.rename(columns={'날짜': 'Date', '종가': 'Close', '전일비': 'Prev_ratio', 
                                   '시가': 'Open', '고가': 'High', '저가': 'Low', '거래량': 'Volume'}, inplace=True)
                
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df.dropna(subset=['Date'], inplace=True) # Remove rows where date conversion failed

                # Filter out data already in DB or older
                if last_db_date:
                    df = df[df['Date'] > last_db_date]
                
                if df.empty:
                    print(f"No new data on page {page} for {stock_code} after filtering by date {last_db_date}.")
                    # If we are on page 1 and no new data, means DB is up to date.
                    # If on later pages, means we've fetched all new data.
                    if page > 1 or (last_db_date and (datetime.now().date() == last_db_date.date() or (datetime.now().date() - last_db_date.date()).days == 1) ): # if db is current or yesterday
                        print(f"Breaking fetch for {stock_code} as no newer data found or DB is current.")
                        break 
                    # else, continue to next page if on page 1 and db is not current
                
                df_list.append(df)
                print(f"Fetched page {page} for {stock_code}, {len(df)} new rows.")

            else: # No table found
                print(f"No data table found on page {page} for {stock_code}. Assuming end of data.")
                break
        except requests.exceptions.RequestException as e_req:
            print(f"Request error on page {page} for {stock_code}: {e_req}")
            break # Stop if there's a network issue
        except Exception as e:
            print(f"Error processing page {page} for {stock_code}: {e}")
            # Decide if to break or continue based on error type
            break 
        
        if page < pages_to_scrape : # Don't sleep on the last iteration
            time.sleep(0.5) # Be respectful to Naver servers

    if df_list:
        new_df = pd.concat(df_list, ignore_index=True)
        # Ensure correct dtypes before inserting, esp. numeric ones
        cols_to_numeric = ['Close', 'Prev_ratio', 'Open', 'High', 'Low', 'Volume']
        for col in cols_to_numeric:
            new_df[col] = pd.to_numeric(new_df[col].astype(str).str.replace(',', ''), errors='coerce')
        new_df.dropna(subset=cols_to_numeric, inplace=True) # Drop rows if essential numeric data is missing

        if not new_df.empty:
            try:
                new_df.to_sql('ks_stock_data_temp', db, if_exists='replace', index=False)
                insert_query = """
                INSERT OR IGNORE INTO ks_stock_data (symbol, Date, Close, Prev_ratio, Open, High, Low, Volume)
                SELECT ?, Date, Close, Prev_ratio, Open, High, Low, Volume FROM ks_stock_data_temp
                """
                # Convert Date column to string for SQLite execution if it's datetime
                # df['Date_str'] = df['Date'].dt.strftime('%Y-%m-%d')
                # Then use Date_str in the executemany or loop
                
                # For simplicity, using loop here for clarity with date formatting
                for _, row in new_df.iterrows():
                    cursor.execute(
                        "INSERT OR IGNORE INTO ks_stock_data (symbol, Date, Close, Prev_ratio, Open, High, Low, Volume) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            stock_code, row['Date'].strftime('%Y-%m-%d'), row['Close'], row['Prev_ratio'],
                            row['Open'], row['High'], row['Low'], row['Volume']
                        )
                    )
                db.commit()
                print(f"Committed {len(new_df)} new rows to DB for {stock_code}.")
            except Exception as e_sql:
                 print(f"SQL Error during bulk insert for {stock_code}: {e_sql}")

    # Always read the full data from DB after update attempt
    try:
        full_df = pd.read_sql_query(
            "SELECT Date, Close, Prev_ratio, Open, High, Low, Volume FROM ks_stock_data WHERE symbol=? ORDER BY Date ASC", 
            con=db, params=(stock_code,)
        )
        if not full_df.empty:
            full_df['Date'] = pd.to_datetime(full_df['Date'])
            full_df.set_index('Date', inplace=True)
            # Ensure numeric types after reading from DB
            for col in ['Close', 'Prev_ratio', 'Open', 'High', 'Low', 'Volume']:
                if col in full_df.columns:
                    full_df[col] = pd.to_numeric(full_df[col], errors='coerce')
            full_df.dropna(subset=['Close', 'Open', 'High', 'Low', 'Volume'], inplace=True) # Critical data
            print(f"Returning {len(full_df)} total rows from DB for {stock_code}")
            db.close()
            return full_df
        else:
            print(f"No data found in DB for {stock_code} after fetch attempt.")
            db.close()
            return pd.DataFrame()
    except Exception as e_read:
        print(f"Error reading full data from DB for {stock_code}: {e_read}")
        return pd.DataFrame()
    finally:
        if db:
            db.close()


def get_korea_stock_list():
    """ks_stock_list 테이블에서 symbol, name을 읽어 리스트로 반환"""
    db = None
    try:
        db = sqlite3.connect(DB_PATH)
        cursor = db.cursor()
        # Ensure table exists
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS ks_stock_list (
            symbol VARCHAR(10) PRIMARY KEY NOT NULL,
            name VARCHAR(255) NOT NULL
        )""")
        db.commit()
        cursor.execute("SELECT symbol, name FROM ks_stock_list ORDER BY name")
        rows = cursor.fetchall()
        if not rows: # Provide a default if empty
            return [('005930', 'Samsung Electronics'), ('000660', 'SK Hynix')]
        return rows
    except sqlite3.Error as e:
        print(f"Database error in get_korea_stock_list: {e}")
        return [('005930', 'Samsung Electronics'), ('000660', 'SK Hynix')] # Fallback
    finally:
        if db:
            db.close()

def add_korea_stock(symbol, name):
    """ks_stock_list 테이블에 symbol, name을 추가"""
    db = None
    try:
        db = sqlite3.connect(DB_PATH)
        cursor = db.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS ks_stock_list (
            symbol VARCHAR(10) PRIMARY KEY NOT NULL,
            name VARCHAR(255) NOT NULL
        )""")
        cursor.execute("INSERT OR IGNORE INTO ks_stock_list (symbol, name) VALUES (?, ?)", (symbol, name))
        db.commit()
        print(f"Added/updated Korean stock: {symbol} - {name}")
        return True
    except sqlite3.Error as e:
        print(f"Database error in add_korea_stock: {e}")
        return False
    finally:
        if db:
            db.close()

# Technical indicator calculation functions (identical to us_stock_utils.py)
def calculate_moving_averages(data):
    data['MA5'] = data['Close'].rolling(window=5, min_periods=1).mean()
    data['MA20'] = data['Close'].rolling(window=20, min_periods=1).mean()
    data['MA50'] = data['Close'].rolling(window=50, min_periods=1).mean()
    data['MA200'] = data['Close'].rolling(window=200, min_periods=1).mean()
    data['Volume_MA20'] = data['Volume'].rolling(window=20, min_periods=1).mean()
    data['Volume_MA50'] = data['Volume'].rolling(window=50, min_periods=1).mean()
    return data

def calculate_bollinger_bands(data, window=20, num_std=2):
    if data.empty or 'Close' not in data.columns: return data
    data['SMA'] = data['Close'].rolling(window=window, min_periods=1).mean()
    rolling_std = data['Close'].rolling(window=window, min_periods=1).std()
    data['Upper_Band'] = data['SMA'] + (rolling_std * num_std)
    data['Lower_Band'] = data['SMA'] - (rolling_std * num_std)
    if 'MA20' in data.columns and not data['MA20'].empty: # Check if MA20 exists and is not all NaN
        data['BB_Width'] = (data['Upper_Band'] - data['Lower_Band']) / data['MA20'].replace(0, pd.NA) # Avoid division by zero
    else: # Fallback if MA20 is not available or all NaN
        data['BB_Width'] = (data['Upper_Band'] - data['Lower_Band']) / data['SMA'].replace(0, pd.NA)
    return data

def calculate_rsi(data, window=14):
    if data.empty or 'Close' not in data.columns: return data
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0.0) # Ensure 0.0 for calculations
    loss = -delta.where(delta < 0, 0.0) # Ensure 0.0 for calculations
    
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    
    rs = avg_gain / avg_loss.replace(0, pd.NA) # Avoid division by zero, replace 0 with NA
    data['RSI'] = 100 - (100 / (1 + rs))
    data['RSI'] = data['RSI'].fillna(50) # Fill NA RSI values (e.g. if avg_loss was 0) with 50
    return data

def calculate_stochastic(data, k_period=14, d_period=3, slowing=3):
    if data.empty or not all(col in data.columns for col in ['Low', 'High', 'Close']): return data
    lowest_low = data['Low'].rolling(window=k_period, min_periods=1).min()
    highest_high = data['High'].rolling(window=k_period, min_periods=1).max()
    
    # Avoid division by zero if highest_high == lowest_low
    denominator = (highest_high - lowest_low).replace(0, pd.NA)
    data['%K'] = 100 * ((data['Close'] - lowest_low) / denominator)
    data['%K'] = data['%K'].fillna(50) # Fill NA %K values (e.g. if denominator was 0)
    
    if slowing > 1:
        data['%K'] = data['%K'].rolling(window=slowing, min_periods=1).mean()
    data['%D'] = data['%K'].rolling(window=d_period, min_periods=1).mean()
    return data

def calculate_daily_change(data):
    if data.empty or not all(col in data.columns for col in ['Open', 'Close']): return data
    data['Daily_Change'] = data['Close'] - data['Open']
    # The 'Prev_ratio' from Naver is 전일비, which is the change amount, not ratio.
    # To calculate a daily change percentage (like US stocks' Prev_ratio):
    # data['Daily_Change_Percent'] = data['Close'].pct_change() * 100 
    data['Daily_Change_MA10'] = data['Daily_Change'].rolling(window=10, min_periods=1).mean() # As in original ks_stock_visualization
    return data

# Visualization function (copied from us_stock_utils.py, potentially refactor to common later)
def create_stock_visualization(stock_data, symbol, display_options):
    if stock_data.empty:
        return "<p>No data available to create chart.</p>"

    num_rows = 2
    subplot_titles = ['Price with Indicators', 'Volume']
    indicator_rows = {}

    # Dynamically add rows and titles for selected indicators
    if display_options.get('bb_width', False) and 'BB_Width' in stock_data.columns:
        num_rows += 1; indicator_rows['bb_width'] = num_rows; subplot_titles.append('Bollinger Band Width')
    if display_options.get('daily_change', False) and 'Daily_Change' in stock_data.columns:
        num_rows += 1; indicator_rows['daily_change'] = num_rows; subplot_titles.append('Daily Change')
    if display_options.get('rsi', False) and 'RSI' in stock_data.columns:
        num_rows += 1; indicator_rows['rsi'] = num_rows; subplot_titles.append('RSI (14)')
    if display_options.get('stochastic', False) and '%K' in stock_data.columns:
        num_rows += 1; indicator_rows['stochastic'] = num_rows; subplot_titles.append('Stochastic (14,3)')
    
    # Monthly difference chart row (as in original) - KS version has this, US version did not use `diff_row`
    # For consistency with the original KS script, let's add it if it was intended.
    # The original KS script had `diff_row` but didn't seem to use it in subplot creation if other indicators were off.
    # Re-evaluating: the original `ks_stock_visualization.py` *does* add a row for 'Monthly Difference' unconditionally.
    num_rows += 1
    monthly_diff_row = num_rows # This row will be for monthly difference
    subplot_titles.append('Monthly Difference')


    row_heights_map = {2: [0.7, 0.3], 3: [0.6, 0.2, 0.2], 4: [0.5, 0.17, 0.17, 0.16],
                       5: [0.4, 0.15, 0.15, 0.15, 0.15], 6: [0.35, 0.13, 0.13, 0.13, 0.13, 0.13],
                       7: [0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2]} # Adjusted for 7 rows
    row_heights = row_heights_map.get(num_rows, [1.0/num_rows]*num_rows) # Default to equal if more than 7
    if num_rows > 1 : row_heights[0] = max(0.3, row_heights[0]) # Ensure price chart has reasonable height


    fig = make_subplots(rows=num_rows, cols=1, shared_xaxes=True, vertical_spacing=0.05, # Reduced vertical_spacing
                        subplot_titles=subplot_titles, row_heights=row_heights)

    fig.add_trace(go.Candlestick(
        x=stock_data.index, open=stock_data['Open'], high=stock_data['High'],
        low=stock_data['Low'], close=stock_data['Close'],
        increasing=dict(line=dict(color='red'), fillcolor='red'),
        decreasing=dict(line=dict(color='blue'), fillcolor='blue'), name='Price'
    ), row=1, col=1)

    # Monthly Min/Max and Difference (as in original KS script)
    # Using 'ME' for month-end frequency for resampling
    monthly_resample = stock_data.resample('ME') 
    if not monthly_resample.empty:
        min_idx = monthly_resample['Low'].idxmin().dropna()
        max_idx = monthly_resample['High'].idxmax().dropna()

        if not min_idx.empty:
            monthly_low = stock_data.loc[stock_data.index.intersection(min_idx)]
            if not monthly_low.empty:
                fig.add_trace(go.Scatter(
                    x=monthly_low.index, y=monthly_low['Low'], mode='markers+text',
                    marker=dict(color='green', size=8), text=monthly_low.index.strftime('%m-%d'),
                    textposition='bottom center', name='Monthly Low', textfont=dict(size=8)
                ), row=1, col=1)
        
        if not max_idx.empty:
            monthly_high = stock_data.loc[stock_data.index.intersection(max_idx)]
            if not monthly_high.empty:
                fig.add_trace(go.Scatter(
                    x=monthly_high.index, y=monthly_high['High'], mode='markers+text',
                    marker=dict(color='black', size=8), text=monthly_high.index.strftime('%m-%d'),
                    textposition='top center', name='Monthly High', textfont=dict(size=8)
                ), row=1, col=1)
        
        # Monthly difference bar chart
        monthly_diff_data = (monthly_resample['High'].max() - monthly_resample['Low'].min()).dropna()
        if not monthly_diff_data.empty:
             fig.add_trace(go.Bar(
                x=monthly_diff_data.index, y=monthly_diff_data,
                text=monthly_diff_data.apply(lambda x: f'{x:,.0f}'), textposition='outside',
                name='Monthly Diff.'
            ), row=monthly_diff_row, col=1)


    ma_colors = {'MA5': 'rgba(255,0,0,0.7)','MA20': 'rgba(255,165,0,0.7)','MA50': 'rgba(0,0,255,0.7)','MA200': 'rgba(128,0,128,0.7)'}
    for ma, color in ma_colors.items():
        if display_options.get(ma, False) and ma in stock_data.columns:
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data[ma], line=dict(color=color, width=1), name=ma), row=1, col=1)

    if display_options.get('bollinger', False) and 'Upper_Band' in stock_data.columns:
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Upper_Band'], line=dict(color='rgba(34,139,34,0.5)', width=1), name='UpperBB'), row=1, col=1)
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Lower_Band'], line=dict(color='rgba(34,139,34,0.5)', width=1), fill='tonexty', fillcolor='rgba(34,139,34,0.1)', name='LowerBB'), row=1, col=1)

    vol_colors = ['red' if c >= o else 'blue' for o, c in zip(stock_data['Open'], stock_data['Close'])]
    fig.add_trace(go.Bar(x=stock_data.index, y=stock_data['Volume'], marker=dict(color=vol_colors), name='Volume'), row=2, col=1)
    
    vol_ma_colors = {'Volume_MA20': 'rgba(255,165,0,0.9)', 'Volume_MA50': 'rgba(0,0,255,0.9)'}
    for vol_ma, color in vol_ma_colors.items(): # Volume MA is not in display_options in original KS
        if vol_ma in stock_data.columns and display_options.get(vol_ma, True): # Default to True if not in display_options
             fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data[vol_ma], line=dict(color=color, width=1.5, dash='dot'), name=vol_ma), row=2, col=1)

    if display_options.get('bb_width', False) and 'BB_Width' in stock_data.columns:
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['BB_Width'], line=dict(color='rgba(75,0,130,0.8)',width=1.5), name='BB Width', fill='tozeroy', fillcolor='rgba(75,0,130,0.1)'), row=indicator_rows['bb_width'], col=1)
    
    if display_options.get('rsi', False) and 'RSI' in stock_data.columns:
        rsi_r = indicator_rows['rsi']
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], line=dict(color='blue',width=1.5), name='RSI'), row=rsi_r, col=1)
        fig.add_shape(type="line", x0=stock_data.index[0],y0=70,x1=stock_data.index[-1],y1=70,line=dict(color="red",width=1,dash="dash"), row=rsi_r, col=1)
        fig.add_shape(type="line", x0=stock_data.index[0],y0=30,x1=stock_data.index[-1],y1=30,line=dict(color="green",width=1,dash="dash"), row=rsi_r, col=1)
        fig.update_yaxes(range=[0,100], row=rsi_r, col=1)

    if display_options.get('daily_change', False) and 'Daily_Change' in stock_data.columns:
        dc_r = indicator_rows['daily_change']
        dc_colors = ['green' if x>=0 else 'red' for x in stock_data['Daily_Change']]
        fig.add_trace(go.Bar(x=stock_data.index, y=stock_data['Daily_Change'], marker_color=dc_colors, name='Daily Change'), row=dc_r, col=1)
        if 'Daily_Change_MA10' in stock_data.columns: # Original KS uses MA10 for daily change
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Daily_Change_MA10'], line=dict(color='blue',width=1,dash='dash'), name='DailyChangeMA10'), row=dc_r, col=1)
        fig.add_shape(type="line", x0=stock_data.index[0],y0=0,x1=stock_data.index[-1],y1=0,line=dict(color="black",width=1), row=dc_r, col=1)

    if display_options.get('stochastic', False) and '%K' in stock_data.columns and '%D' in stock_data.columns:
        sto_r = indicator_rows['stochastic']
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['%K'], line=dict(color='blue',width=1.5), name='%K'), row=sto_r, col=1)
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['%D'], line=dict(color='red',width=1.5,dash='dash'), name='%D'), row=sto_r, col=1)
        fig.add_shape(type="line", x0=stock_data.index[0],y0=80,x1=stock_data.index[-1],y1=80,line=dict(color="red",width=1,dash="dash"), row=sto_r, col=1)
        fig.add_shape(type="line", x0=stock_data.index[0],y0=20,x1=stock_data.index[-1],y1=20,line=dict(color="green",width=1,dash="dash"), row=sto_r, col=1)
    
    fig.update_layout(title=f'{symbol} Stock Chart', xaxis_rangeslider_visible=False, height=max(600, num_rows * 200), # Dynamic height
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
                      margin=dict(l=50, r=50, t=80, b=50)) # Adjusted margin

    # Apply monthly grid lines to all x-axes
    for i in range(1, num_rows + 1):
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(220,220,220,0.5)', dtick="M1", tickformat="%b\n%Y", row=i, col=1, tickangle=-45)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(220,220,220,0.5)', row=i, col=1)

    # Dividend Highlighting (from original KS script, though it was commented out or using US DB path)
    # Assuming dividends table might be shared or specific to KS stocks.
    # For now, using DB_PATH. If KS dividends are stored differently, this needs adjustment.
    db_div = None
    try:
        db_div = sqlite3.connect(DB_PATH) # uses local DB_PATH from this file
        cur_div = db_div.cursor()
        # Ensure table exists, if not, this will fail gracefully or create it if defined
        # cursor.execute("CREATE TABLE IF NOT EXISTS dividends (...)") 
        start_dt = stock_data.index.min().date()
        end_dt = stock_data.index.max().date()
        cur_div.execute("SELECT ex_date, pay_date FROM dividends WHERE symbol=? AND ex_date BETWEEN ? AND ?", (symbol, start_dt, end_dt))
        for ex_date_str, pay_date_str in cur_div.fetchall():
            try:
                ex_date = pd.to_datetime(ex_date_str).date()
                pay_date = pd.to_datetime(pay_date_str).date()
                if ex_date < pay_date:
                    fig.add_vrect(x0=ex_date, x1=pay_date, fillcolor='rgba(255,255,0,0.2)', layer='below', line_width=0, row=1, col=1)
            except Exception as e_date:
                print(f"Error processing dividend date for KS chart: {e_date}")
    except Exception as e_div:
        print(f"Could not load dividend ranges for KS chart: {e_div}")
    finally:
        if db_div:
            db_div.close()

    return fig.to_html(full_html=False, include_plotlyjs='cdn')
