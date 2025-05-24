from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
from datetime import datetime, timedelta
from us_stock_utils import (
    fetch_stock_data,
    calculate_moving_averages,
    calculate_bollinger_bands,
    calculate_rsi,
    calculate_stochastic,
    calculate_daily_change,
    create_stock_visualization,
    get_us_stock_list, # Added for future use
    add_us_stock # Added for future use
)

app = Flask(__name__)
app.secret_key = os.urandom(24) # For session management

# Define a simple root route for initial testing
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/us-stocks', methods=['GET', 'POST'])
def us_stocks_page():
    chart_html = None
    # Default values
    selected_symbol = request.values.get('symbol', 'TSLA') # Use request.values to get from GET or POST
    
    time_periods = {
        '1 Week': 7, '1 Month': 30, '3 Months': 90,
        '6 Months': 180, '1 Year': 365, '3 Years': 1095, '5 Years': 1825
    }
    selected_period_key = request.values.get('time_period', '1 Year')
    display_days = time_periods.get(selected_period_key, 365)

    # Default display options
    current_display_options = {
        'MA5': request.values.get('show_ma5') == 'true',
        'MA20': request.values.get('show_ma20') == 'true',
        'MA50': request.values.get('show_ma50') == 'true',
        'MA200': request.values.get('show_ma200') == 'true',
        'bollinger': request.values.get('show_bollinger') == 'true',
        'bb_width': request.values.get('show_bb_width') == 'true',
        'rsi': request.values.get('show_rsi') == 'true',
        'daily_change': request.values.get('show_daily_change') == 'true',
        'stochastic': request.values.get('show_stochastic') == 'true',
        'Volume_MA20': request.values.get('show_vol_ma20') == 'true',
        'Volume_MA50': request.values.get('show_vol_ma50') == 'true',
    }
    # On initial GET, set some defaults to True if no form submitted yet
    if request.method == 'GET':
        current_display_options['MA5'] = True
        current_display_options['MA20'] = True
        current_display_options['MA50'] = True
        current_display_options['bollinger'] = True
        current_display_options['rsi'] = True


    us_stock_list = get_us_stock_list() # Fetch for dropdown

    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'add_stock':
            new_symbol = request.form.get('new_symbol', '').upper().strip()
            new_name = request.form.get('new_name', '').strip()
            if new_symbol and new_name:
                if add_us_stock(new_symbol, new_name):
                    flash(f'Stock {new_symbol} - {new_name} added successfully!', 'success')
                    # Update stock list for the current response
                    us_stock_list = get_us_stock_list()
                    selected_symbol = new_symbol # Select the newly added stock
                else:
                    flash(f'Failed to add stock {new_symbol}. It might already exist or there was a database error.', 'error')
            else:
                flash('New symbol and name cannot be empty.', 'error')
            # Return the page with updated list and potentially selected new stock, but no chart yet for add_stock
            return render_template("us_stocks.html",
                                   stock_list=us_stock_list,
                                   selected_symbol=selected_symbol,
                                   time_periods=time_periods.keys(),
                                   selected_period=selected_period_key,
                                   display_options=current_display_options,
                                   chart_html=None)

        elif action == 'get_chart':
            selected_symbol = request.form.get('symbol')
            selected_period_key = request.form.get('time_period')
            display_days = time_periods.get(selected_period_key, 365)

            if not selected_symbol:
                flash('Please select a stock symbol.', 'error')
            else:
                # Fetch full data (e.g., 5 years)
                fetch_end_date = datetime.now()
                fetch_start_date = fetch_end_date - timedelta(days=1825) # 5 years of data
                
                full_data = fetch_stock_data(selected_symbol, fetch_start_date, fetch_end_date)

                if not full_data.empty:
                    full_data = calculate_moving_averages(full_data)
                    full_data = calculate_bollinger_bands(full_data)
                    full_data = calculate_rsi(full_data)
                    full_data = calculate_stochastic(full_data)
                    full_data = calculate_daily_change(full_data)
                    
                    # Filter data *after* calculations
                    display_start_date = fetch_end_date - timedelta(days=display_days)
                    display_data = full_data[full_data.index >= display_start_date]
                    
                    chart_html = create_stock_visualization(display_data, selected_symbol, current_display_options)
                else:
                    flash(f"Could not retrieve data for symbol: {selected_symbol}.", 'error')
    
    # For GET request or after POST actions that lead to chart display
    return render_template("us_stocks.html",
                           stock_list=us_stock_list,
                           selected_symbol=selected_symbol,
                           time_periods=time_periods.keys(),
                           selected_period=selected_period_key,
                           display_options=current_display_options,
                           chart_html=chart_html)

@app.route('/exchange-rates', methods=['GET', 'POST'])
def exchange_rates_page():
    from exchange_rate_utils import get_exchange_rates_from_yahoo, prepare_current_rates_for_template, visualize_combined_exchange_data_html

    # Defaults
    default_currencies_str = "USD,EUR,JPY,CNY,GBP" # Comma-separated string for form input
    # Convert to list for processing, ensure no empty strings if input is empty
    selected_currencies_list = [c.strip() for c in request.values.get('currencies', default_currencies_str).split(',') if c.strip()]
    
    try:
        max_points = int(request.values.get('max_points', 30))
        if not (7 <= max_points <= 90): # Basic validation
             max_points = 30
             flash("Data points must be between 7 and 90. Defaulting to 30.", "warning")
    except ValueError:
        max_points = 30
        flash("Invalid input for data points. Defaulting to 30.", "warning")

    current_rates_data = []
    chart_html = None

    if not selected_currencies_list:
        flash("Please select at least one currency.", "warning")
    else:
        exchange_data = get_exchange_rates_from_yahoo(currencies=selected_currencies_list, max_points=max_points)
        
        if exchange_data and any(data.get('rates') for data in exchange_data.values()):
            current_rates_data = prepare_current_rates_for_template(exchange_data)
            chart_html = visualize_combined_exchange_data_html(exchange_data)
        else:
            flash(f"Could not retrieve sufficient exchange rate data for {', '.join(selected_currencies_list)}. Please try again later.", 'error')

    return render_template("exchange_rates.html",
                           current_rates=current_rates_data,
                           chart_html=chart_html,
                           # Pass back the current form values
                           selected_currencies_str=','.join(selected_currencies_list), # For the input field
                           max_points_value=max_points)

@app.route('/investment-portfolio', methods=['GET'])
def investment_portfolio_page():
    from investment_utils import get_unique_tickers_from_investments, get_investment_data_for_ticker, get_overall_investment_summary
    
    selected_ticker = request.args.get('ticker', None)
    unique_tickers = get_unique_tickers_from_investments()
    
    ticker_data = None
    overall_summary_df = None

    if not unique_tickers:
        flash("No investment data found in the database.", "warning")
    elif selected_ticker and selected_ticker not in unique_tickers:
        flash(f"Ticker {selected_ticker} not found in investments. Displaying first available ticker.", "warning")
        selected_ticker = unique_tickers[0] # Default to first ticker if invalid one is provided
    elif not selected_ticker and unique_tickers:
        selected_ticker = unique_tickers[0] # Default to first ticker if none selected

    if selected_ticker:
        ticker_data = get_investment_data_for_ticker(selected_ticker)
        # Convert DataFrames to HTML, or pass them directly if your template handles it
        if ticker_data:
            ticker_data["active_html"] = ticker_data["active_df"].to_html(classes="table table-sm table-striped", index=False, escape=True) if not ticker_data["active_df"].empty else "<p>No active investments for this ticker.</p>"
            ticker_data["sold_html"] = ticker_data["sold_df"].to_html(classes="table table-sm table-striped", index=False, escape=True) if not ticker_data["sold_df"].empty else "<p>No sold investments for this ticker.</p>"
    else:
        # Optionally, fetch and display overall summary if no tickers are available or none selected
        # For now, the logic above ensures a ticker is selected if available.
        # If truly no tickers, a flash message is already set.
        pass

    # Fetch overall summary for a potential summary display (optional, not in current ticker-focused view)
    # overall_summary_df = get_overall_investment_summary()
    # overall_summary_html = overall_summary_df.to_html(classes="table table-sm table-striped", index=False, escape=True) if not overall_summary_df.empty else ""


    return render_template("investment_portfolio.html",
                           unique_tickers=unique_tickers,
                           selected_ticker=selected_ticker,
                           ticker_data=ticker_data,
                           # overall_summary_html=overall_summary_html # If you want to display it
                           )

# Time periods dictionary - can be global or defined within routes
TIME_PERIODS = {
    '1 Week': 7, '1 Month': 30, '3 Months': 90,
    '6 Months': 180, '1 Year': 365, '3 Years': 1095, '5 Years': 1825
}

@app.route('/ks-stocks', methods=['GET', 'POST'])
def ks_stocks_page():
    from ks_stock_utils import (
        fetch_korea_stock_data, get_korea_stock_list, add_korea_stock,
        calculate_moving_averages, calculate_bollinger_bands, calculate_rsi,
        calculate_stochastic, calculate_daily_change, create_stock_visualization
    )

    chart_html = None
    # Default selected symbol: first from the list or a common one like Samsung Electronics
    ks_stock_list = get_korea_stock_list()
    default_ks_symbol = ks_stock_list[0][0] if ks_stock_list else '005930' # Default to Samsung if list is empty initially
    
    selected_symbol = request.values.get('symbol', default_ks_symbol)
    selected_period_key = request.values.get('time_period', '1 Year') # Default to '1 Year'
    display_days = TIME_PERIODS.get(selected_period_key, 365)

    # Indicator display options from form, default to False if not present
    current_display_options = {
        'MA5': request.values.get('show_ma5') == 'true',
        'MA20': request.values.get('show_ma20') == 'true',
        'MA50': request.values.get('show_ma50') == 'true',
        'MA200': request.values.get('show_ma200') == 'true',
        'bollinger': request.values.get('show_bollinger') == 'true',
        'bb_width': request.values.get('show_bb_width') == 'true',
        'rsi': request.values.get('show_rsi') == 'true',
        'daily_change': request.values.get('show_daily_change') == 'true',
        'stochastic': request.values.get('show_stochastic') == 'true',
        # KS chart in Streamlit didn't have Volume_MA toggles, but calculations exist
        'Volume_MA20': request.values.get('show_vol_ma20', 'false') == 'true', 
        'Volume_MA50': request.values.get('show_vol_ma50', 'false') == 'true',
    }

    if request.method == 'GET':
        # Set some common indicators to True by default on initial load
        current_display_options.update({
            'MA5': True, 'MA20': True, 'MA50': True, 'bollinger': True, 'rsi': True
        })


    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'add_stock':
            new_symbol = request.form.get('new_ks_symbol', '').strip()
            new_name = request.form.get('new_ks_name', '').strip()
            if new_symbol and new_name:
                if add_korea_stock(new_symbol, new_name):
                    flash(f'Korean Stock {new_symbol} - {new_name} added successfully!', 'success')
                    ks_stock_list = get_korea_stock_list() # Refresh list
                    selected_symbol = new_symbol # Select the newly added stock
                else:
                    flash(f'Failed to add Korean stock {new_symbol}. It might already exist or DB error.', 'error')
            else:
                flash('New Korean stock symbol and name cannot be empty.', 'error')
            # No chart generation needed for add_stock, just re-render with updated list
            return render_template("ks_stocks.html",
                                   stock_list=ks_stock_list,
                                   selected_symbol=selected_symbol,
                                   time_periods=TIME_PERIODS.keys(),
                                   selected_period=selected_period_key,
                                   display_options=current_display_options,
                                   chart_html=None)

        elif action == 'get_chart':
            selected_symbol = request.form.get('symbol') # Already captured by request.values above, but explicit for POST action
            if not selected_symbol:
                flash('Please select a Korean stock symbol.', 'error')
            else:
                # Fetch data (e.g., Naver scraping logic in fetch_korea_stock_data handles its own range)
                # The `fetch_korea_stock_data` will fetch incrementally based on DB.
                # For display, we filter *after* calculations on the full available dataset from DB.
                
                # Max pages for scraping can be passed if desired, e.g. fetch_korea_stock_data(selected_symbol, max_pages=10)
                # For now, using default max_pages in the util function.
                full_data = fetch_korea_stock_data(selected_symbol) 

                if not full_data.empty:
                    full_data = calculate_moving_averages(full_data)
                    full_data = calculate_bollinger_bands(full_data)
                    full_data = calculate_rsi(full_data)
                    full_data = calculate_stochastic(full_data)
                    full_data = calculate_daily_change(full_data)
                    
                    # Filter data for display *after* all calculations
                    # Ensure index is datetime for filtering
                    if not isinstance(full_data.index, pd.DatetimeIndex):
                         full_data.index = pd.to_datetime(full_data.index)

                    display_start_date = datetime.now() - timedelta(days=display_days)
                    # Ensure display_start_date is timezone naive if full_data.index is naive, or localize
                    if full_data.index.tz is not None and display_start_date.tzinfo is None:
                        display_start_date = display_start_date.replace(tzinfo=full_data.index.tz)
                    elif full_data.index.tz is None and display_start_date.tzinfo is not None:
                        display_start_date = display_start_date.replace(tzinfo=None)

                    display_data = full_data[full_data.index >= display_start_date]
                    
                    if display_data.empty:
                        flash(f"No data available for {selected_symbol} in the selected period: {selected_period_key}", "warning")
                        chart_html = None
                    else:
                        chart_html = create_stock_visualization(display_data, selected_symbol, current_display_options)
                else:
                    flash(f"Could not retrieve data for Korean stock symbol: {selected_symbol}.", 'error')
    
    # For GET request or after POST actions that lead to chart display
    # If it was a GET and a symbol is selected (e.g. default or from query param), try to load chart
    elif request.method == 'GET' and selected_symbol:
        full_data = fetch_korea_stock_data(selected_symbol)
        if not full_data.empty:
            full_data = calculate_moving_averages(full_data)
            full_data = calculate_bollinger_bands(full_data)
            full_data = calculate_rsi(full_data)
            full_data = calculate_stochastic(full_data)
            full_data = calculate_daily_change(full_data)
            
            if not isinstance(full_data.index, pd.DatetimeIndex):
                full_data.index = pd.to_datetime(full_data.index)

            display_start_date = datetime.now() - timedelta(days=display_days)
            if full_data.index.tz is not None and display_start_date.tzinfo is None:
                display_start_date = display_start_date.replace(tzinfo=full_data.index.tz)
            elif full_data.index.tz is None and display_start_date.tzinfo is not None:
                display_start_date = display_start_date.replace(tzinfo=None)
            
            display_data = full_data[full_data.index >= display_start_date]

            if display_data.empty:
                flash(f"No data found for {selected_symbol} for the default period ({selected_period_key}). Try fetching or selecting a different period.", "info")
            else:
                chart_html = create_stock_visualization(display_data, selected_symbol, current_display_options)
        else:
            flash(f"No data found for {selected_symbol}. You may need to add it or fetch its data via POST.", "info")


    return render_template("ks_stocks.html",
                           stock_list=ks_stock_list,
                           selected_symbol=selected_symbol,
                           time_periods=TIME_PERIODS.keys(),
                           selected_period=selected_period_key,
                           display_options=current_display_options,
                           chart_html=chart_html)


@app.route('/dividends', methods=['GET', 'POST'])
def dividends_page():
    from dividend_utils import get_all_stock_symbols, add_dividend, get_dividends
    from datetime import datetime # Ensure datetime is imported

    stock_symbols = get_all_stock_symbols()
    filter_symbol_selected = request.values.get('filter_symbol', None) # Use request.values for GET or POST fallback

    if request.method == 'POST':
        # Adding a new dividend
        symbol = request.form.get('symbol')
        ex_date_str = request.form.get('ex_date')
        pay_date_str = request.form.get('pay_date')
        dividend_str = request.form.get('dividend')

        if not all([symbol, ex_date_str, pay_date_str, dividend_str]):
            flash('All fields are required to add a dividend.', 'error')
        else:
            try:
                ex_date = datetime.strptime(ex_date_str, '%Y-%m-%d').date()
                pay_date = datetime.strptime(pay_date_str, '%Y-%m-%d').date()
                dividend_amount = float(dividend_str)

                if dividend_amount < 0:
                    flash('Dividend amount cannot be negative.', 'error')
                elif add_dividend(symbol, ex_date, pay_date, dividend_amount):
                    flash(f'Dividend for {symbol} on ex-date {ex_date_str} added successfully!', 'success')
                    # Update filter_symbol_selected to the one just added to show its data
                    filter_symbol_selected = symbol 
                else:
                    flash(f'Failed to add dividend for {symbol}. It might already exist or there was a database error.', 'error')
            except ValueError:
                flash('Invalid date format or dividend amount. Please use YYYY-MM-DD for dates and a valid number for amount.', 'error')
            
        # Redirect to GET to show the (potentially filtered) list and avoid form resubmission issues
        if filter_symbol_selected:
             return redirect(url_for('dividends_page', filter_symbol=filter_symbol_selected))
        return redirect(url_for('dividends_page'))

    # GET Request handling (also after POST redirect)
    dividends_df = get_dividends(filter_symbol=filter_symbol_selected if filter_symbol_selected else None)
    
    dividends_html = ""
    if not dividends_df.empty:
        dividends_html = dividends_df.to_html(classes='table table-sm table-striped', index=False, escape=True)
    elif filter_symbol_selected :
        flash(f"No dividend records found for symbol {filter_symbol_selected}.", 'info')
    # else: # No filter, and no dividends at all
        # flash("No dividend records found in the database.", 'info') # This might be too noisy on initial load

    return render_template("dividends.html",
                           stock_symbols=stock_symbols,
                           dividends_html=dividends_html,
                           filter_symbol=filter_symbol_selected) # filter_symbol is used to pre-select in dropdown

if __name__ == '__main__':
    app.run(debug=True)
