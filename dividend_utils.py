import pandas as pd
import sqlite3
import os
from datetime import date # For type hinting and usage

# Assuming us_stock_utils.py and ks_stock_utils.py are in the same directory or accessible in PYTHONPATH
try:
    from us_stock_utils import get_us_stock_list
except ImportError:
    print("Warning: us_stock_utils.py not found or get_us_stock_list not available.")
    # Fallback or dummy function if direct import fails during development/testing
    def get_us_stock_list(): return [('AAPL', 'Apple Inc-Fallback'), ('MSFT', 'Microsoft-Fallback')]

try:
    from ks_stock_utils import get_korea_stock_list
except ImportError:
    print("Warning: ks_stock_utils.py not found or get_korea_stock_list not available.")
    # Fallback or dummy function
    def get_korea_stock_list(): return [('005930', 'Samsung Elec-Fallback'), ('000660', 'SK Hynix-Fallback')]


# Path to SQLite database file (consistent with other utils)
DB_PATH = os.path.join(os.path.dirname(__file__), 'investdb.db')

def _ensure_dividends_table():
    """Ensures the dividends table exists in the database."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS dividends (
            symbol VARCHAR(20) NOT NULL, -- Increased symbol length for safety
            ex_date DATE NOT NULL,
            pay_date DATE,
            dividend FLOAT,
            PRIMARY KEY(symbol, ex_date)
        )
        """)
        conn.commit()
    except sqlite3.Error as e:
        print(f"Database error while ensuring dividends table: {e}")
    finally:
        if conn:
            conn.close()


def add_dividend(symbol: str, ex_date: date, pay_date: date, dividend_amount: float):
    """
    Adds a new dividend record to the database.
    Dates should be Python date objects.
    Returns True on success, False on error.
    """
    _ensure_dividends_table() # Ensure table exists before trying to insert
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO dividends (symbol, ex_date, pay_date, dividend) VALUES (?, ?, ?, ?)",
            (symbol, ex_date.isoformat(), pay_date.isoformat(), dividend_amount)
        )
        conn.commit()
        print(f"Dividend added for {symbol}, ex-date {ex_date}, amount {dividend_amount}")
        return True
    except sqlite3.Error as e:
        print(f"Error adding dividend for {symbol}: {e}")
        return False
    finally:
        if conn:
            conn.close()

def get_dividends(filter_symbol: str = None):
    """
    Retrieves dividend records from the database, optionally filtered by symbol.
    Returns a Pandas DataFrame.
    """
    _ensure_dividends_table() # Ensure table exists before trying to query
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        if filter_symbol:
            query = "SELECT symbol, ex_date, pay_date, dividend FROM dividends WHERE symbol = ? ORDER BY ex_date DESC"
            df = pd.read_sql_query(query, conn, params=(filter_symbol,))
        else:
            query = "SELECT symbol, ex_date, pay_date, dividend FROM dividends ORDER BY ex_date DESC"
            df = pd.read_sql_query(query, conn)
        
        # Standardize column names for consistency with original and potential template usage
        if not df.empty:
            df.columns = ['Symbol', 'Ex-Date', 'Pay-Date', 'Dividend'] # Match desired output
            # Convert date strings from DB to datetime objects if needed, then to string for display
            df['Ex-Date'] = pd.to_datetime(df['Ex-Date']).dt.strftime('%Y-%m-%d')
            df['Pay-Date'] = pd.to_datetime(df['Pay-Date']).dt.strftime('%Y-%m-%d')
        
        print(f"Retrieved {len(df)} dividend records. Filter symbol: {filter_symbol}")
        return df
    except sqlite3.Error as e:
        print(f"Error retrieving dividends: {e}")
        return pd.DataFrame(columns=['Symbol', 'Ex-Date', 'Pay-Date', 'Dividend'])
    finally:
        if conn:
            conn.close()


def get_all_stock_symbols():
    """
    Fetches US and Korean stock symbols and returns a sorted list of unique symbols.
    """
    us_stocks = get_us_stock_list() # List of tuples (symbol, name)
    ks_stocks = get_korea_stock_list() # List of tuples (symbol, name)
    
    # Extract symbols from the tuples
    us_symbols = [s[0] for s in us_stocks if s and len(s) > 0]
    ks_symbols = [s[0] for s in ks_stocks if s and len(s) > 0]
    
    combined_symbols = list(set(us_symbols + ks_symbols)) # Use set to ensure uniqueness
    combined_symbols.sort() # Sort for consistent dropdown order
    
    print(f"Combined stock symbols: {len(combined_symbols)} unique symbols found.")
    return combined_symbols

if __name__ == '__main__':
    print("Testing dividend_utils.py...")
    # Test _ensure_dividends_table (implicitly tested by add/get)
    
    # Test get_all_stock_symbols
    all_symbols = get_all_stock_symbols()
    print(f"All stock symbols for dropdown: {all_symbols[:5]}... (Total: {len(all_symbols)})")

    # Test add_dividend
    if all_symbols:
        test_symbol = all_symbols[0] if all_symbols else "TESTSYM"
        print(f"\nTesting add_dividend with symbol: {test_symbol}")
        from datetime import date, timedelta
        today = date.today()
        ex_d = today - timedelta(days=30)
        pay_d = today - timedelta(days=15)
        add_dividend(test_symbol, ex_d, pay_d, 1.25)
        add_dividend(test_symbol, ex_d - timedelta(days=365), pay_d - timedelta(days=365), 1.20)

        # Test get_dividends (all)
        print("\nTesting get_dividends (all):")
        all_divs_df = get_dividends()
        print(all_divs_df.head())

        # Test get_dividends (filtered)
        print(f"\nTesting get_dividends (filtered by {test_symbol}):")
        filtered_divs_df = get_dividends(filter_symbol=test_symbol)
        print(filtered_divs_df)
    else:
        print("No symbols available to test add_dividend effectively.")

    # Example with a non-existent symbol for get_dividends
    print("\nTesting get_dividends (filtered by NONEXISTENT_SYMBOL):")
    non_existent_df = get_dividends(filter_symbol="NONEXISTENT_SYMBOL")
    print(f"DataFrame is empty: {non_existent_df.empty}")
    print(non_existent_df)
