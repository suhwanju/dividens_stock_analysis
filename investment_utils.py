import pandas as pd
from stock_db_manager import StockInvestmentDB, DB_PATH # Import DB_PATH as well for direct use if needed
import sqlite3 # For direct queries if necessary, and type hinting

def get_db_connection():
    """Helper function to get a SQLite connection."""
    conn = sqlite3.connect(DB_PATH)
    return conn

def get_all_investment_data():
    """
    Fetches all investment data using StockInvestmentDB.
    Returns a Pandas DataFrame.
    """
    db = StockInvestmentDB()
    if db.connect():
        df = db.get_all_investments()
        db.disconnect()
        return df
    return pd.DataFrame()

def get_unique_tickers_from_investments():
    """
    Fetches all investment data and returns a list of unique stock tickers.
    """
    df = get_all_investment_data()
    if not df.empty and 'stock_ticker' in df.columns:
        # Filter out None/NaN values and get unique tickers
        unique_tickers = [ticker for ticker in df['stock_ticker'].unique() if pd.notna(ticker)]
        return unique_tickers
    return []

def get_investment_data_for_ticker(ticker_symbol):
    """
    Fetches all investment data for a specific ticker.
    Returns a dictionary with 'all', 'active', 'sold' DataFrames and 'summary_metrics'.
    """
    db = StockInvestmentDB()
    data = {
        "all_df": pd.DataFrame(),
        "active_df": pd.DataFrame(),
        "sold_df": pd.DataFrame(),
        "summary_metrics": {
            "total_invested": 0,
            "active_invested_value": 0, # Changed from active_investment to avoid confusion
            "realized_profit": 0,
            "total_transactions": 0 # Added from summary
        }
    }

    if not db.connect():
        print("Failed to connect to the database.")
        return data

    # Fetch all transactions for the ticker to calculate total_invested
    all_ticker_transactions_df = db.get_investments_by_ticker(ticker_symbol)
    if not all_ticker_transactions_df.empty:
        data["all_df"] = all_ticker_transactions_df
        # Calculate total invested from all transactions for this ticker
        # Assuming 'invest_amount' is the column for investment per transaction
        if 'invest_amount' in all_ticker_transactions_df.columns:
            data["summary_metrics"]["total_invested"] = all_ticker_transactions_df['invest_amount'].sum()
        data["summary_metrics"]["total_transactions"] = len(all_ticker_transactions_df)


    # Fetch active investments for the ticker
    active_df = all_ticker_transactions_df[all_ticker_transactions_df['status'] == 'active']
    data["active_df"] = active_df
    if not active_df.empty and 'invest_amount' in active_df.columns:
        data["summary_metrics"]["active_invested_value"] = active_df['invest_amount'].sum()

    # Fetch sold investments for the ticker
    sold_df = all_ticker_transactions_df[all_ticker_transactions_df['status'] == 'sold']
    data["sold_df"] = sold_df
    if not sold_df.empty and 'sell_profit_amount' in sold_df.columns: # Original used selling_profit_amount
        # Check for both column names found in the source code
        profit_col = 'sell_profit_amount' if 'sell_profit_amount' in sold_df.columns else 'selling_profit_amount'
        if profit_col in sold_df.columns:
            data["summary_metrics"]["realized_profit"] = sold_df[profit_col].sum()
        else:
            print(f"Warning: Profit column not found in sold_df for ticker {ticker_symbol}")


    # Disconnect from DB
    db.disconnect()
    return data

def get_overall_investment_summary():
    """
    Fetches a summary of all investments grouped by ticker.
    Returns a Pandas DataFrame.
    """
    db = StockInvestmentDB()
    if db.connect():
        summary_df = db.get_investment_summary() # This method already exists in StockInvestmentDB
        db.disconnect()
        return summary_df
    return pd.DataFrame()

# Example of how to convert DataFrames to HTML for Flask if needed directly in utils
# (though typically this is done in the app.py route or template)
def dataframe_to_html(df, table_id=None, classes=None):
    """Converts a Pandas DataFrame to an HTML table string."""
    if df is None or df.empty:
        return "<p>No data available.</p>"
    
    html_string = df.to_html(escape=True, index=False, table_id=table_id, classes=classes)
    return html_string

if __name__ == '__main__':
    # Test functions (optional)
    print("Fetching unique tickers...")
    tickers = get_unique_tickers_from_investments()
    print(f"Unique Tickers: {tickers}")

    if tickers:
        selected_ticker = tickers[0]
        print(f"\nFetching data for ticker: {selected_ticker}")
        ticker_data = get_investment_data_for_ticker(selected_ticker)
        print("\nSummary Metrics:")
        print(ticker_data["summary_metrics"])
        print("\nActive Investments DataFrame:")
        print(ticker_data["active_df"].head())
        # print("\nActive Investments HTML:")
        # print(dataframe_to_html(ticker_data["active_df"], classes="table table-sm"))
        print("\nSold Investments DataFrame:")
        print(ticker_data["sold_df"].head())

    print("\nFetching overall investment summary...")
    overall_summary = get_overall_investment_summary()
    print(overall_summary.head())
