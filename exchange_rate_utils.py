import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import requests # yfinance might need it, and good to have for potential future use

def get_exchange_rates_from_yahoo(currencies=None, max_points=30):
    """
    Yahoo Finance API를 사용하여 환율 데이터를 가져옵니다.
    Returns a dictionary of DataFrames or an empty dict if an error occurs.
    Example: {'USD': {'dates': ['2023-01-01', ...], 'rates': [1200.0, ...]}, ...}
    """
    all_data = pd.DataFrame()
    
    end_date = datetime.now()
    # Ensure start_date is calculated based on max_points, not fixed 30 days
    start_date = end_date - timedelta(days=max_points) 

    if currencies is None:
        currencies = ["USD", "EUR", "JPY", "CNY", "GBP"] # Default if none provided

    # Yahoo Finance uses pairs like "USDKRW=X"
    # The function expects base currencies like "USD", "EUR"
    # We need to fetch against KRW. So, if "USD" is in currencies, we fetch "USDKRW=X"
    currency_pairs = []
    for curr in currencies:
        if curr != "KRW": # KRW itself is the base, so no pair needed
            currency_pairs.append(f"{curr}KRW=X")
        # If KRW is requested, it's implicitly handled or can be added as 1.0 later if needed

    results = {}

    for pair in currency_pairs:
        try:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    ticker = yf.Ticker(pair)
                    data = ticker.history(start=start_date, end=end_date, interval="1d")
                    if not data.empty:
                        # Extract the currency code (e.g., USD from USDKRW=X)
                        currency_code = pair.replace("KRW=X", "")
                        
                        # Store dates and rates
                        dates = [d.strftime('%Y-%m-%d') for d in data.index]
                        rates = data['Close'].tolist()
                        
                        results[currency_code] = {"dates": dates, "rates": rates}
                        print(f"Successfully fetched data for {pair}")
                        break 
                    else:
                        print(f"No data for {pair}, attempt {attempt+1}")
                        if attempt < max_retries - 1:
                            time.sleep(2 ** attempt)
                        else:
                            print(f"Failed to fetch data for {pair} after {max_retries} attempts (empty data).")
                except Exception as e_fetch:
                    print(f"Exception fetching data for {pair}, attempt {attempt+1}: {str(e_fetch)}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    else:
                        print(f"Failed to fetch data for {pair} after {max_retries} attempts: {str(e_fetch)}")
                        # Store empty data for this currency to indicate failure but not break others
                        currency_code = pair.replace("KRW=X", "")
                        results[currency_code] = {"dates": [], "rates": []}
        except Exception as e_outer:
            # This handles errors in the outer loop logic for a pair
            print(f"Error processing pair {pair}: {str(e_outer)}")
            currency_code = pair.replace("KRW=X", "")
            results[currency_code] = {"dates": [], "rates": []}
            
    return results


def prepare_current_rates_for_template(exchange_data):
    """
    Formats the latest rates from exchange_data for HTML template display.
    Returns a list of dictionaries.
    Example: [{'currency': 'USD', 'rate': '1200.50', 'delta': '+0.50 (0.04%)'}, ...]
    """
    display_data = []
    if not exchange_data:
        return display_data

    for currency, data in exchange_data.items():
        if data.get("rates") and len(data["rates"]) > 0:
            current_rate = data["rates"][-1]
            prev_rate = data["rates"][-2] if len(data["rates"]) > 1 else current_rate
            
            delta = current_rate - prev_rate
            delta_percent = (delta / prev_rate) * 100 if prev_rate and prev_rate != 0 else 0
            
            rate_str = f"{current_rate:.2f}"
            delta_str = f"{delta:+.2f} ({delta_percent:+.2f}%)" # Added + for positive changes

            display_data.append({
                "currency": currency,
                "rate": rate_str,
                "delta": delta_str
            })
        else:
            # Handle cases where a currency might have no data
             display_data.append({
                "currency": currency,
                "rate": "N/A",
                "delta": "N/A"
            })
    return display_data

def visualize_combined_exchange_data_html(exchange_data):
    """
    Generates a Plotly HTML string for combined exchange rate trends and volatility.
    Returns HTML string or None if no data.
    """
    if not exchange_data or all(not v.get('rates') for v in exchange_data.values()):
        print("No data available to visualize.")
        return None
    
    fig = make_subplots(rows=2, cols=1, 
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        subplot_titles=("Exchange Rate Trend (vs KRW)", "Daily Volatility (%)"))
    
    colors = {
        "USD": "#1f77b4", "EUR": "#ff7f0e", "JPY": "#2ca02c",
        "CNY": "#d62728", "GBP": "#9467bd", "AUD": "#8c564b",
        "CAD": "#e377c2", "CHF": "#7f7f7f", "HKD": "#bcbd22",
        "SGD": "#17becf"
    }
    
    # Collect all dates to determine overall range for 0% line
    all_processed_dates = set()

    for currency, data_dict in exchange_data.items():
        # Ensure data_dict is a dictionary and has 'dates' and 'rates'
        if not isinstance(data_dict, dict) or 'dates' not in data_dict or 'rates' not in data_dict:
            print(f"Skipping {currency}: Data is not in the expected format or missing keys.")
            continue
        
        dates = data_dict['dates']
        rates = data_dict['rates']

        if not dates or not rates or len(dates) != len(rates):
            print(f"Skipping {currency}: Dates or rates are empty or have mismatched lengths.")
            continue
        
        # Ensure dates are strings and sort
        try:
            # Convert to datetime for sorting if not already, then back to string if necessary for Plotly
            # Plotly generally handles datetime objects well for x-axis
            date_objects = [pd.to_datetime(d) for d in dates]
            date_rate_pairs = sorted(zip(date_objects, rates), key=lambda x: x[0])
            sorted_dates_dt = [pair[0] for pair in date_rate_pairs]
            # sorted_dates_str = [dt.strftime('%Y-%m-%d') for dt in sorted_dates_dt] # if string format is strictly needed by a part of plotly
            sorted_rates = [pair[1] for pair in date_rate_pairs]
        except Exception as e:
            print(f"Error processing dates for {currency}: {e}")
            continue


        # 1. Exchange Rate Trend
        fig.add_trace(
            go.Scatter(
                x=sorted_dates_dt, y=sorted_rates, mode='lines',
                name=f'{currency}/KRW (Trend)',
                line=dict(color=colors.get(currency, "#000000")),
                hovertemplate='%{y:.2f} KRW<extra></extra>'
            ), row=1, col=1
        )
        
        # 2. Volatility
        if len(sorted_rates) > 1:
            daily_changes = []
            volatility_dates_dt = []
            for i in range(1, len(sorted_rates)):
                prev_rate = sorted_rates[i-1]
                curr_rate = sorted_rates[i]
                if prev_rate != 0:
                    daily_change = ((curr_rate - prev_rate) / prev_rate) * 100
                    daily_changes.append(daily_change)
                    volatility_dates_dt.append(sorted_dates_dt[i]) # Use the date of the current rate
            
            if volatility_dates_dt:
                fig.add_trace(
                    go.Scatter(
                        x=volatility_dates_dt, y=daily_changes, mode='lines',
                        name=f'{currency}/KRW (Volatility)',
                        line=dict(color=colors.get(currency, "#000000"), dash='dot'),
                        hovertemplate='Volatility: %{y:.2f}%<extra></extra>'
                    ), row=2, col=1
                )
                all_processed_dates.update(volatility_dates_dt)

    if all_processed_dates:
        min_date = min(all_processed_dates)
        max_date = max(all_processed_dates)
        fig.add_shape(
            type="line", x0=min_date, x1=max_date, y0=0, y1=0,
            line=dict(color="black", width=1, dash="dash"),
            row=2, col=1
        )
    
    fig.update_layout(
        height=700, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', griddash='dot')
    fig.update_yaxes(gridcolor='lightgray', griddash='dot')
    fig.update_yaxes(title_text="Exchange Rate (KRW)", row=1, col=1)
    fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1) # Shared X-axis, so title on the last one is fine
    
    return fig.to_html(full_html=False, include_plotlyjs='cdn')
