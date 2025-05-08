# stock_db_manager.py
import sqlite3
from sqlite3 import Error
import pandas as pd
from datetime import datetime
import os

# Path to SQLite database file
DB_PATH = os.path.join(os.path.dirname(__file__), 'investdb.db')

class StockInvestmentDB:
    def __init__(self):
        """Initialize database connection parameters"""
        self.connection = None
    
    def connect(self):
        """Establish connection to the SQLite database"""
        try:
            self.connection = sqlite3.connect(DB_PATH)
            return True
        except Error as e:
            print(f"Error connecting to SQLite: {e}")
            return False
    
    def disconnect(self):
        """Close the database connection"""
        if self.connection:
            self.connection.close()
    
    def get_all_investments(self):
        """Retrieve all stock investments from the database"""
        if not self.connection:
            if not self.connect():
                return pd.DataFrame()
        
        try:
            query = "SELECT * FROM endless_stock_invest_vw"
            df = pd.read_sql(query, self.connection)
            return df
        except Error as e:
            print(f"Error retrieving investment data: {e}")
            return pd.DataFrame()
    
    def get_active_investments(self):
        """Retrieve only active investments (not sold)"""
        if not self.connection:
            if not self.connect():
                return pd.DataFrame()
        
        try:
            query = "SELECT * FROM endless_stock_invest_vw WHERE status = 'active' AND avail = 1"
            df = pd.read_sql(query, self.connection)
            return df
        except Error as e:
            print(f"Error retrieving active investment data: {e}")
            return pd.DataFrame()
    
    def get_sold_investments(self):
        """Retrieve only sold investments"""
        if not self.connection:
            if not self.connect():
                return pd.DataFrame()
        
        try:
            query = "SELECT * FROM endless_stock_invest_vw WHERE status = 'sold' AND avail = 1"
            df = pd.read_sql(query, self.connection)
            return df
        except Error as e:
            print(f"Error retrieving sold investment data: {e}")
            return pd.DataFrame()
    
    def get_investments_by_ticker(self, ticker):
        """Retrieve investments for a specific stock ticker"""
        if not self.connection:
            if not self.connect():
                return pd.DataFrame()
        
        try:
            query = "SELECT * FROM endless_stock_invest_vw WHERE stock_ticker = ? AND avail = 1"
            df = pd.read_sql(query, self.connection, params=(ticker,))
            return df
        except Error as e:
            print(f"Error retrieving investment data for {ticker}: {e}")
            return pd.DataFrame()
    
    def get_investment_summary(self):
        """Get summary of investments grouped by ticker"""
        if not self.connection:
            if not self.connect():
                return pd.DataFrame()
        
        try:
            query = """
            SELECT 
                stock_ticker, 
                stock_name,
                COUNT(*) as total_transactions,
                SUM(CASE WHEN status = 'active' THEN invest_amount ELSE 0 END) as active_investment,
                SUM(CASE WHEN status = 'sold' THEN sell_profit_amount ELSE 0 END) as realized_profit
            FROM 
                endless_stock_invest_vw
            WHERE 
                avail = 1
            GROUP BY 
                stock_ticker, stock_name
            """
            df = pd.read_sql(query, self.connection)
            return df
        except Error as e:
            print(f"Error retrieving investment summary: {e}")
            return pd.DataFrame()
