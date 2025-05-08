import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import requests
from io import StringIO
from stock_db_manager import StockInvestmentDB
from exchange_rate_visualization import get_exchange_rates,get_exchange_rates_from_yahoo, visualize_exchange_rates, show_current_rates,visualize_combined_exchange_data
from bs4 import BeautifulSoup
import sqlite3
import os
from sqlite3 import Error

# Path to SQLite database file
DB_PATH = os.path.join(os.path.dirname(__file__), 'investdb.db')

def fetch_korea_stock_data(stock_code):
    # SQLite setup for incremental update
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
    """
    )
    db.commit()
    cursor.execute(
        "SELECT MAX(Date) FROM ks_stock_data WHERE symbol=?", (stock_code,)
    )
    last = cursor.fetchone()[0]
    last_date = pd.to_datetime(last) if last else None

    # Original scraping setup
    base_url = f'https://finance.naver.com/item/sise_day.nhn?code={stock_code}'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
    }

    df_list = []
    for page in range(1, 30):
        url = f'{base_url}&page={page}'
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table')
            if table:
                df = pd.read_html(str(table), header=0)[0].dropna()
                df.columns = ['Date','Close','Prev_ratio','Open','High','Low','Volume']
                df['Date'] = pd.to_datetime(df['Date'])
                if last_date:
                    df = df[df['Date'] > last_date]
                if df.empty:
                    break
                df_list.append(df)
        except Exception as e:
            st.write(f"Error: {e}")
        time.sleep(1)

    if df_list:
        new_df = pd.concat(df_list, ignore_index=True)
        for _, row in new_df.iterrows():
            cursor.execute(
                "INSERT OR IGNORE INTO ks_stock_data (symbol, Date, Close, Prev_ratio, Open, High, Low, Volume) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    stock_code, row['Date'].strftime('%Y-%m-%d'), row['Close'], row['Prev_ratio'],
                    row['Open'], row['High'], row['Low'], row['Volume']
                )
            )
        db.commit()
        full_df = pd.read_sql_query(
            "SELECT Date, Close, Prev_ratio, Open, High, Low, Volume FROM ks_stock_data WHERE symbol=? ORDER BY Date", 
            con=db, params=(stock_code,)
        )
        full_df['Date'] = pd.to_datetime(full_df['Date'])
        full_df.set_index('Date', inplace=True)
        return full_df
    else:
        if last_date:
            full_df = pd.read_sql_query(
                "SELECT Date, Close, Prev_ratio, Open, High, Low, Volume FROM ks_stock_data WHERE symbol=? ORDER BY Date",
                con=db, params=(stock_code,)
            )
            full_df['Date'] = pd.to_datetime(full_df['Date'])
            full_df.set_index('Date', inplace=True)
            return full_df
        return pd.DataFrame()

def get_korea_stock_list():
    """ks_stock_list 테이블에서 symbol, name을 읽어 리스트로 반환"""
    db = sqlite3.connect(DB_PATH)
    cursor = db.cursor()
    cursor.execute("SELECT symbol, name FROM ks_stock_list")
    rows = cursor.fetchall()  # [('005930','삼성전자'), ('000660','SK하이닉스'), ...]
    db.close()
    return rows

def add_korea_stock(symbol, name):
    """ks_stock_list 테이블에 symbol, name을 추가"""
    db = sqlite3.connect(DB_PATH)
    cursor = db.cursor()
    cursor.execute(
        "INSERT OR IGNORE INTO ks_stock_list (symbol, name) VALUES (?, ?)",
        (symbol, name)
    )
    db.commit()
    db.close()

def calculate_moving_averages(data):
    """Calculate various moving averages (5, 20, 50, 200 days)"""
    # Calculate price moving averages
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    
    # Calculate volume moving averages
    data['Volume_MA20'] = data['Volume'].rolling(window=20).mean()
    data['Volume_MA50'] = data['Volume'].rolling(window=50).mean()
    
    return data

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculate Bollinger Bands (20-day, 2 standard deviations)"""
    # Calculate the Simple Moving Average
    data['SMA'] = data['Close'].rolling(window=window).mean()
    
    # Calculate the standard deviation
    rolling_std = data['Close'].rolling(window=window).std()
    
    # Calculate upper and lower bands
    data['Upper_Band'] = data['SMA'] + (rolling_std * num_std)
    data['Lower_Band'] = data['SMA'] - (rolling_std * num_std)

    # Calculate Bollinger Band Width
    data['BB_Width'] = (data['Upper_Band'] - data['Lower_Band']) / data['MA20']
       
    return data

def calculate_rsi(data, window=14):
    """Calculate RSI (Relative Strength Index)"""
    if data.empty:
        return data
    
    # Calculate price changes
    delta = data['Close'].diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and loss over the specified window
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    # Calculate RS (Relative Strength)
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    data['RSI'] = 100 - (100 / (1 + rs))
    
    return data

def calculate_stochastic(data, k_period=14, d_period=3, slowing=3):
    """Calculate Stochastic Oscillator"""
    if data.empty:
        return data
    
    # Calculate %K
    lowest_low = data['Low'].rolling(window=k_period).min()
    highest_high = data['High'].rolling(window=k_period).max()
    data['%K'] = 100 * ((data['Close'] - lowest_low) / (highest_high - lowest_low))
    
    # Calculate %K Slowing (if slowing > 1)
    if slowing > 1:
        data['%K'] = data['%K'].rolling(window=slowing).mean()
    
    # Calculate %D (moving average of %K)
    data['%D'] = data['%K'].rolling(window=d_period).mean()
    
    return data

def calculate_daily_change(data):
    """Calculate the daily price change and percentage change"""
    if data.empty:
        return data
    
    # Calculate absolute price change
    data['Daily_Change'] = data['Close'] - data['Open']
    
    
    # Calculate 20-day moving average of daily change
    data['Daily_Change_MA10'] = data['Daily_Change'].rolling(window=10).mean()

    return data

def create_stock_visualization(stock_data, symbol, display_options):
    """Create and display stock visualization with Bollinger Bands and Moving Averages"""
    # Determine how many rows we need and track their positions
    num_rows = 2  # Start with price and volume
    subplot_titles = ['Price with Indicators', 'Volume']
    
    # Track which row each indicator will use
    indicator_rows = {}
    
    # Calculate total rows needed
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

    # Add to the row calculation section
    if display_options.get('stochastic', False):
        num_rows += 1
        indicator_rows['stochastic'] = num_rows
        subplot_titles.append('Stochastic (14,3)')

    # Add row for monthly difference graph
    num_rows += 1
    diff_row = num_rows
    subplot_titles.append('Monthly Difference')
    
    # Adjust row heights based on total number of rows
    if num_rows == 2:
        row_heights = [0.7, 0.3]
    elif num_rows == 3:
        row_heights = [0.6, 0.2, 0.2]
    elif num_rows == 4:
        row_heights = [0.5, 0.17, 0.17, 0.16]
    elif num_rows == 5:
        row_heights = [0.4, 0.15, 0.15, 0.15, 0.15]
    elif num_rows == 6:
        row_heights = [0.35, 0.13, 0.13, 0.13, 0.13, 0.13]
    else:
        # Fallback for any other case
        row_heights = [0.7/num_rows] * num_rows
        row_heights[0] = 0.3  # Give more space to the price chart
    
    # Create visualization
    fig = make_subplots(
        rows=num_rows, 
        cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.1,
        subplot_titles=subplot_titles,
        row_heights=row_heights
    )

    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=stock_data.index,
            open=stock_data['Open'],
            high=stock_data['High'],
            low=stock_data['Low'],
            close=stock_data['Close'],
            increasing=dict(line=dict(color='red'), fillcolor='red'),
            decreasing=dict(line=dict(color='blue'), fillcolor='blue'),
            name='Price'
        ),
        row=1, col=1
    )

    # Add monthly min/max red markers with date labels
    monthly = stock_data.resample('M')
    min_idx = monthly['Low'].idxmin()
    max_idx = monthly['High'].idxmax()
    monthly_low = stock_data.loc[min_idx]
    monthly_high = stock_data.loc[max_idx]
    fig.add_trace(go.Scatter(
        x=monthly_low.index, y=monthly_low['Low'],
        mode='markers+text',
        marker=dict(color='green', size=8),
        text=monthly_low.index.strftime('%m-%d'),
        textposition='bottom center',
        textfont=dict(size=8, color='black'),
        name='Monthly Low'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=monthly_high.index, y=monthly_high['High'],
        mode='markers+text',
        marker=dict(color='black', size=8),
        text=monthly_high.index.strftime('%m-%d'),
        textposition='top center',
        textfont=dict(size=8, color='black'),
        name='Monthly High'
    ), row=1, col=1)

    # Add monthly difference bar chart
    monthly = stock_data.resample('M')
    monthly_diff = monthly['High'].max() - monthly['Low'].min()
    
    fig.add_trace(go.Bar(
        x=monthly_diff.index, y=monthly_diff,
        text=monthly_diff,
        textposition='outside',
        name='Monthly Difference'
    ), row=diff_row, col=1)

    # Add Moving Averages based on display options
    ma_colors = {
        'MA5': 'rgba(255, 0, 0, 0.7)',     # Red
        'MA20': 'rgba(255, 165, 0, 0.7)',  # Orange
        'MA50': 'rgba(0, 0, 255, 0.7)',    # Blue
        'MA200': 'rgba(128, 0, 128, 0.7)'  # Purple
    }
    
    for ma_type, color in ma_colors.items():
        if ma_type in stock_data.columns and display_options.get(ma_type, False):
            fig.add_trace(
                go.Scatter(
                    x=stock_data.index,
                    y=stock_data[ma_type],
                    line=dict(color=color, width=1),
                    name=f'{ma_type}'
                ),
                row=1, col=1
            )

    # Add Bollinger Bands if enabled
    if display_options.get('bollinger', False):
        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data['Upper_Band'],
                line=dict(color='rgba(34, 139, 34, 0.5)', width=1),
                name='Upper Band (+2σ)'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data['Lower_Band'],
                line=dict(color='rgba(34, 139, 34, 0.5)', width=1),
                fill='tonexty',
                fillcolor='rgba(34, 139, 34, 0.1)',
                name='Lower Band (-2σ)'
            ),
            row=1, col=1
        )

    # Add volume chart with colors based on price movement
    vol_colors = ['red' if c >= o else 'blue' for o, c in zip(stock_data['Open'], stock_data['Close'])]
    fig.add_trace(
        go.Bar(
            x=stock_data.index,
            y=stock_data['Volume'],
            marker=dict(color=vol_colors),
            name='Volume'
        ),
        row=2, col=1
    )
    
    # Add Volume Moving Averages
    volume_ma_colors = {
        'Volume_MA20': 'rgba(255, 165, 0, 0.9)',  # Orange
        'Volume_MA50': 'rgba(0, 0, 255, 0.9)'     # Blue
    }

    for ma_type, color in volume_ma_colors.items():
        if ma_type in stock_data.columns:
            line_style = 'dot' 
            
            fig.add_trace(
                go.Scatter(
                    x=stock_data.index,
                    y=stock_data[ma_type],
                    line=dict(
                        color=color, 
                        width=2,
                        dash=line_style  # Set the line style
                    ),
                    mode='lines',
                    name=f'{ma_type}'
                ),
                row=2, col=1
            )
    
    # Add Bollinger Band Width chart if enabled
    if display_options.get('bb_width', False):
        bb_width_row = indicator_rows['bb_width']
        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data['BB_Width'],
                line=dict(color='rgba(75, 0, 130, 0.8)', width=1.5),
                name='BB Width',
                fill='tozeroy',
                fillcolor='rgba(75, 0, 130, 0.1)'
            ),
            row=bb_width_row, col=1
        )

    # Add RSI chart if enabled
    if display_options.get('rsi', False):
        rsi_row = indicator_rows['rsi']
        
        # Add colored backgrounds for overbought/oversold regions
        # Add overbought area (above 70) - red
        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=[70] * len(stock_data),
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo='none',
                name="Overbought Threshold"
            ),
            row=rsi_row, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data['RSI'].apply(lambda x: 70 if x < 70 else x),
                line=dict(color="rgba(0,0,0,0)"),
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.2)',
                showlegend=False,
                hoverinfo='none',
                name="Overbought"
            ),
            row=rsi_row, col=1
        )
        
        # Annotate overbought region boundaries with small font
        rsi_series = stock_data['RSI']
        mask_high = rsi_series >= 70
        start_dates_high = mask_high[(~mask_high.shift(1, fill_value=False)) & mask_high].index
        end_dates_high = mask_high[(~mask_high.shift(-1, fill_value=False)) & mask_high].index
        for sd, ed in zip(start_dates_high, end_dates_high):
            fig.add_trace(go.Scatter(
                x=[sd], y=[70],
                mode='text',
                text=[sd.strftime('%m-%d')],
                textposition='bottom center',
                textfont=dict(size=8, color='black'),
                showlegend=False
            ), row=rsi_row, col=1)
            fig.add_trace(go.Scatter(
                x=[ed], y=[70],
                mode='text',
                text=[ed.strftime('%m-%d')],
                textposition='top center',
                textfont=dict(size=8, color='black'),
                showlegend=False
            ), row=rsi_row, col=1)

        # Add oversold area (below 30) - green
        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=[30] * len(stock_data),
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo='none',
                name="Oversold Threshold"
            ),
            row=rsi_row, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data['RSI'].apply(lambda x: x if x < 30 else 30),
                line=dict(color="rgba(0,0,0,0)"),
                fill='tonexty',
                fillcolor='rgba(0,255,0,0.2)',
                showlegend=False,
                hoverinfo='none',
                name="Oversold"
            ),
            row=rsi_row, col=1
        )

        # Annotate RSI<=30 region boundaries
        rsi_series = stock_data['RSI']
        mask = rsi_series <= 30
        start_dates = mask[(~mask.shift(1, fill_value=False)) & mask].index
        end_dates = mask[(~mask.shift(-1, fill_value=False)) & mask].index
        for sd, ed in zip(start_dates, end_dates):
            fig.add_trace(go.Scatter(
                x=[sd], y=[30],
                mode='text',
                text=[sd.strftime('%m-%d')],
                textposition='bottom center',
                textfont=dict(size=9, color='black'),
                showlegend=False
            ), row=rsi_row, col=1)
            fig.add_trace(go.Scatter(
                x=[ed], y=[30],
                mode='text',
                text=[ed.strftime('%m-%d')],
                textposition='top center',
                textfont=dict(size=9, color='black'),
                showlegend=False
            ), row=rsi_row, col=1)
        
        # Add RSI line (must come after the fill areas)
        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data['RSI'],
                line=dict(color='blue', width=1.5),
                name='RSI (14)'
            ),
            row=rsi_row, col=1
        )
        
        # Add reference lines
        fig.add_shape(
            type="line",
            x0=stock_data.index[0],
            y0=70,
            x1=stock_data.index[-1],
            y1=70,
            line=dict(color="red", width=1, dash="dash"),
            row=rsi_row, col=1
        )
        
        fig.add_shape(
            type="line",
            x0=stock_data.index[0],
            y0=30,
            x1=stock_data.index[-1],
            y1=30,
            line=dict(color="green", width=1, dash="dash"),
            row=rsi_row, col=1
        )
        
        # Add midline
        fig.add_shape(
            type="line",
            x0=stock_data.index[0],
            y0=50,
            x1=stock_data.index[-1],
            y1=50,
            line=dict(color="gray", width=1, dash="dot"),
            row=rsi_row, col=1
        )
    
    if display_options.get('daily_change', False):
        # Find the correct row for daily change
        daily_change_row = indicator_rows['daily_change']
        
        # Add daily change line with color variation
        colors = ['green' if x >= 0 else 'red' for x in stock_data['Daily_Change']]
        
        fig.add_trace(
            go.Bar(  # Change to a bar chart for better visualization
                x=stock_data.index,
                y=stock_data['Daily_Change'],
                marker_color=colors,
                name='Daily Change'
            ),
            row=daily_change_row, col=1
        )

        # Add 20-day moving average line
        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data['Daily_Change_MA10'],
                line=dict(color='blue', width=1, dash='dash'),  # Changed to dashed line
                name='Daily Change MA(20)',
                showlegend=True 
            ),
            row=daily_change_row, col=1
        )
        
        # Add zero line reference
        fig.add_shape(
            type="line",
            x0=stock_data.index[0],
            y0=0,
            x1=stock_data.index[-1],
            y1=0,
            line=dict(color="black", width=1),
            row=daily_change_row, col=1
        )
    # Add Stochastic chart if enabled
    if display_options.get('stochastic', False):
        stochastic_row = indicator_rows['stochastic']
        
        # Add %K line (faster)
        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data['%K'],
                line=dict(color='blue', width=1.5),
                name='%K'
            ),
            row=stochastic_row, col=1
        )
        
        # Add %D line (slower)
        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data['%D'],
                line=dict(color='red', width=1.5, dash='dash'),
                name='%D'
            ),
            row=stochastic_row, col=1
        )
        
        # Add overbought/oversold lines
        fig.add_shape(
            type="line",
            x0=stock_data.index[0],
            y0=80,
            x1=stock_data.index[-1],
            y1=80,
            line=dict(color="red", width=1, dash="dash"),
            row=stochastic_row, col=1
        )
        
        fig.add_shape(
            type="line",
            x0=stock_data.index[0],
            y0=20,
            x1=stock_data.index[-1],
            y1=20,
            line=dict(color="green", width=1, dash="dash"),
            row=stochastic_row, col=1
        )
        
        # Add midline
        fig.add_shape(
            type="line",
            x0=stock_data.index[0],
            y0=50,
            x1=stock_data.index[-1],
            y1=50,
            line=dict(color="gray", width=1, dash="dot"),
            row=stochastic_row, col=1
        )

    # Update layout with monthly grid lines
    fig.update_layout(
        title=f'{symbol} Stock Data with Technical Indicators',
        xaxis_rangeslider_visible=False,
        height=1170 if display_options.get('bb_width', False) else 1040, 
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            font=dict(size=7),
            bordercolor="LightGrey",
            borderwidth=1,
            bgcolor="rgba(255, 255, 255, 0.7)"
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            dtick='M1',
            tickformat='%b\n%Y'
        ),
        xaxis2=dict(
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            dtick='M1',
            tickformat='%b\n%Y'
        )
    )
    
    # Configure additional x-axes if needed
    if num_rows >= 3:
        fig.update_layout(
            xaxis3=dict(
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)',
                dtick='M1',
                tickformat='%b\n%Y'
            )
        )
    
    if num_rows == 4:
        fig.update_layout(
            xaxis4=dict(
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)',
                dtick='M1',
                tickformat='%b\n%Y'
            )
        )
    
    # Set y-axis range for RSI
    if display_options.get('rsi', False):
        fig.update_yaxes(range=[0, 100], row=rsi_row, col=1)
    
    # Update x-axis date formatting
    fig.update_xaxes(tickformat='%m-%d', tickangle=-45)
    
    # Highlight dividend periods
    try:
        db_div = sqlite3.connect(DB_PATH)
        cur_div = db_div.cursor()
        start_dt = stock_data.index.min().date()
        end_dt = stock_data.index.max().date()
        cur_div.execute(
            "SELECT ex_date, pay_date FROM dividends WHERE symbol=? AND ex_date BETWEEN ? AND ?",
            (symbol, start_dt, end_dt)
        )
        for ex_date, pay_date in cur_div.fetchall():
            fig.add_vrect(
                x0=ex_date, x1=pay_date,
                fillcolor='yellow', opacity=0.3,
                layer='below', line_width=0,
                row=1, col=1
            )
        db_div.close()
    except Exception as e:
        st.warning(f"Could not load dividend ranges: {e}")
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)

# Then add this function to your file
def show_investment_data():
    """Display investment data from the database"""
    
    st.subheader("Investment Portfolio")
    
    # Database connection setup
    db_config = {
        "host": "localhost",
        "user": "your_username", 
        "password": "your_password",
        "database": "your_database_name"
    }
    
    # Create database connection
    db = StockInvestmentDB(**db_config)
    
    # Show connection status
    if db.connect():
        #st.success("Database connection successful!")
        
        # Create tabs for different views
        invest_tab1, invest_tab2, invest_tab3, invest_tab4 = st.tabs([
            "All Investments", "Active Investments", "Sold Investments", "Summary"
        ])
        
        with invest_tab1:
            st.subheader("All Investments")
            all_data = db.get_all_investments()
            if not all_data.empty:
                st.dataframe(all_data, use_container_width=True)
            else:
                st.info("No investment data found.")
        
        with invest_tab2:
            st.subheader("Active Investments")
            active_data = db.get_active_investments()
            if not active_data.empty:
                # Calculate current values if possible
                st.dataframe(active_data, use_container_width=True)
            else:
                st.info("No active investments found.")
        
        with invest_tab3:
            st.subheader("Sold Investments")
            sold_data = db.get_sold_investments()
            if not sold_data.empty:
                # Add profit/loss metrics
                total_profit = sold_data["sell_profit_amount"].sum()
                st.metric("Total Realized Profit/Loss", f"${total_profit:,.2f}")
                st.dataframe(sold_data, use_container_width=True)
            else:
                st.info("No sold investments found.")
        
        with invest_tab4:
            st.subheader("Investment Summary")
            summary_data = db.get_investment_summary()
            if not summary_data.empty:
                # Add overall metrics
                total_active = summary_data["active_investment"].sum()
                total_profit = summary_data["realized_profit"].sum()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Active Investment", f"${total_active:,.2f}")
                with col2:
                    st.metric("Total Realized Profit", f"${total_profit:,.2f}")
                
                st.dataframe(summary_data, use_container_width=True)
            else:
                st.info("No investment summary available.")
        
        # Close the connection
        db.disconnect()
    else:
        st.error("Failed to connect to the database. Check your credentials.")

def show_investment_data():
    """Display investment data from the database"""
    
    st.subheader("Investment Portfolio")
    
    # Database connection setup
    db_config = {
        "host": "localhost",
        "user": "root",
        "password": "Shju6256#$", 
        "database": "investdb"
    }
    
    # Create database connection
    db = StockInvestmentDB(**db_config)
    
    # Show connection status
    if db.connect():
        #st.success("Database connection successful!")
        
        # Get all investment data
        all_data = db.get_all_investments()
        
        if all_data.empty:
            st.error("No investment data found.")
            return
        
        # Get unique tickers
        if 'stock_ticker' in all_data.columns:
            # Filter out None/NaN values and convert to list
            unique_tickers = [ticker for ticker in all_data['stock_ticker'].unique() if ticker and str(ticker) != 'nan']
            
            if unique_tickers:  # Check if the list is not empty
                # Initialize session state for selected_ticker if it doesn't exist
                if 'selected_ticker' not in st.session_state:
                    st.session_state.selected_ticker = unique_tickers[0]
                
                # Create a tab for each ticker and track which one is selected
                ticker_index = unique_tickers.index(st.session_state.selected_ticker) if st.session_state.selected_ticker in unique_tickers else 0
                ticker_tabs = st.tabs(unique_tickers)
                
                # For each ticker, show its specific data
                for i, ticker in enumerate(unique_tickers):
                    with ticker_tabs[i]:
                        # Update the selected ticker in session state when this tab is active
                        if i == ticker_index or ticker_tabs[i].active:
                            st.session_state.selected_ticker = ticker
                        
                        ticker_data = all_data[all_data['stock_ticker'] == ticker]
                        
                        # Show summary for this ticker
                        active_ticker_data = ticker_data[ticker_data['status'] == 'H']
                        sold_ticker_data = ticker_data[ticker_data['status'] == 'S']
                        
                        # Calculate ticker metrics
                        total_invested = ticker_data['invest_amount'].sum()
                        active_invested = active_ticker_data['invest_amount'].sum() if not active_ticker_data.empty else 0
                        realized_profit = sold_ticker_data['selling_profit_amount'].sum() if not sold_ticker_data.empty else 0
                        
                        # Display ticker metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Invested", f"${total_invested:,.2f}")
                        with col2:
                            st.metric("Active Investment", f"${active_invested:,.2f}")
                        with col3:
                            st.metric("Realized Profit", f"${realized_profit:,.2f}")
                        
                        # Show active investments for this ticker
                        st.subheader(f"Active {ticker} Investments")
                        if not active_ticker_data.empty:
                            total_active_amount = active_ticker_data['invest_amount'].sum()
                            st.metric(f"Total {ticker} Active Investment", f"${total_active_amount:,.2f}")
                            st.dataframe(active_ticker_data, use_container_width=True)
                        else:
                            st.info(f"No active {ticker} investments found.")
                        
                        if not sold_ticker_data.empty:
                            st.subheader(f"Sold {ticker} Investments")
                            st.dataframe(sold_ticker_data, use_container_width=True)

                        #stock_chart(ticker)
            else:
                st.info("No valid ticker information found in the data.")
        else:
            st.error("The database doesn't contain a 'stock_ticker' column.")
        
        # Close the connection
        db.disconnect()
    else:
        st.error("Failed to connect to the database. Check your credentials.")

def kor_stock_chart(ticker='466940'):
    # App title
    #st.title('국내 주식 Data Visualization')

    # 신규 종목 추가 UI
    st.subheader('종목 리스트에 신규 종목 추가')
    col1, col2, col3 = st.columns([2,3,1])
    with col1:
        new_ks_symbol = st.text_input('심볼 입력', '', key='kr_new_symbol').upper().strip()
    with col2:
        new_ks_name = st.text_input('종목명 입력', '', key='kr_new_name').strip()
    with col3:
        if st.button('추가', key='kr_add_button', type='primary'):
            if new_ks_symbol and new_ks_name:
                add_korea_stock(new_ks_symbol, new_ks_name)
                st.success(f'{new_ks_symbol} - {new_ks_name} 추가되었습니다.')
                st.experimental_rerun()
            else:
                st.warning('심볼과 종목명을 모두 입력하세요')

    # ① 테이블에서 불러온 종목 리스트
    stock_list = get_korea_stock_list()
    # ['005930 - 삼성전자', '000660 - SK하이닉스', ...]
    options = [f"{sym} - {name}" for sym, name in stock_list]

    # ② selectbox 에서만 선택 가능
    selection = st.selectbox('종목 선택', options)
    symbol = selection.split(' - ')[0]  # 앞부분 symbol만 추출

    # Update session state if user changes the symbol
    if symbol != st.session_state.get('selected_ticker', None):
        st.session_state.selected_ticker = symbol
    # Time period selection for DISPLAY only
    time_periods = {
        '1 Week': 7,
        '1 Month': 30,
        '3 Months': 90,
        '6 Months': 180,
        '1 Year': 365,
        '3 Year': 1095,
        '5 Year': 1825
    }
    options = list(time_periods.keys())
    default_index = options.index('6 Months')
    selected_period = st.selectbox('Select Display Time Period', options, index=default_index, key='kr_stock_period')
    
    # Indicator toggles in a horizontal layout
    st.write("### Display Options")
    # Add another column for RSI toggle
    # Add another column for Daily Change toggle
    col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)
        
    with col1:
        show_ma5 = st.checkbox('KMA5', value=True)
    with col2:
        show_ma20 = st.checkbox('KMA20', value=True)
    with col3:
        show_ma50 = st.checkbox('KMA50', value=True)
    with col4:
        show_ma200 = st.checkbox('KMA200', value=True)
    with col5:
        show_bollinger = st.checkbox('KBollinger Bands', value=True)
    with col6:
        show_bb_width = st.checkbox('KBB Width', value=False)
    with col7:
        show_rsi = st.checkbox('KRSI', value=True)
    with col8:
        show_daily_change = st.checkbox('KDaily Change', value=False)
    with col9:
        show_stochastic = st.checkbox('KStochastic', value=False)

    # Store display preferences in a dictionary
    display_options = {
        'MA5': show_ma5,
        'MA20': show_ma20,
        'MA50': show_ma50,
        'MA200': show_ma200,
        'bollinger': show_bollinger,
        'bb_width': show_bb_width,
        'rsi': show_rsi,
        'daily_change': show_daily_change,
        'stochastic': show_stochastic
    }

    try:
        if not symbol:
            st.warning('Please enter a valid stock symbol.')
        else:
            # Always fetch 5 years of data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1825)  # 5 years
            
            with st.spinner(f'Loading data for {symbol}...'):
                # Fetch the full 5-year dataset
                full_stock_data = fetch_korea_stock_data(symbol)
                
                # Calculate indicators if data is available
                # In the main function, update the indicator calculation part:
            if not full_stock_data.empty:
                full_stock_data = calculate_moving_averages(full_stock_data)
                full_stock_data = calculate_bollinger_bands(full_stock_data)
                #full_stock_data = calculate_volume_mas(full_stock_data)
                full_stock_data = calculate_rsi(full_stock_data)
                full_stock_data = calculate_daily_change(full_stock_data)
                full_stock_data = calculate_stochastic(full_stock_data)
            if full_stock_data.empty:
                st.error(f"No data found for {symbol}. Please check if the stock symbol is correct.")
            else:
                # Filter data based on selected display period
                display_days = time_periods[selected_period]
                display_start_date = end_date - timedelta(days=display_days)
                display_data = full_stock_data[full_stock_data.index >= display_start_date]
                
                # Visualize the filtered data
                create_stock_visualization(display_data, symbol, display_options)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
