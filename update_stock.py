import streamlit as st
import pandas as pd
import sqlite3
import os
from sqlite3 import Error

# Path to SQLite database file
DB_PATH = os.path.join(os.path.dirname(__file__), 'investdb.db')

def add_dividend(symbol, ex_date, pay_date, dividend):
    db = sqlite3.connect(DB_PATH)
    cursor = db.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS dividends (
        symbol VARCHAR(10) NOT NULL,
        ex_date DATE NOT NULL,
        pay_date DATE,
        dividend FLOAT,
        PRIMARY KEY(symbol, ex_date)
    )
    """)
    db.commit()
    cursor.execute(
        "INSERT OR IGNORE INTO dividends (symbol, ex_date, pay_date, dividend) VALUES (?, ?, ?, ?)",
        (symbol, ex_date.isoformat(), pay_date.isoformat(), dividend)
    )
    db.commit()
    db.close()

def get_dividends():
    db = sqlite3.connect(DB_PATH)
    cursor = db.cursor()
    cursor.execute("SELECT symbol, ex_date, pay_date, dividend FROM dividends ORDER BY ex_date DESC")
    rows = cursor.fetchall()
    db.close()
    df = pd.DataFrame(rows, columns=['Symbol', 'Ex Date', 'Pay Date', 'Dividend'])
    return df

def update_stock_page():
    st.subheader("배당 정보")
    # Load KR and US stock lists for symbol selection
    db = sqlite3.connect(DB_PATH)
    cursor = db.cursor()
    cursor.execute("SELECT symbol, name FROM ks_stock_list")
    ks_list = cursor.fetchall()
    cursor.execute("SELECT symbol, name FROM us_stock_list")
    us_list = cursor.fetchall()
    cursor.close()
    db.close()
    options = [f"{s} - {n}" for s, n in ks_list] + [f"{s} - {n}" for s, n in us_list]
    symbol_option = st.selectbox("심볼 선택", options)
    symbol = symbol_option.split(" - ")[0]
    # Input form
    ex_date = st.date_input("배당락 일자")
    pay_date = st.date_input("배당지급 일자")
    dividend = st.number_input("배당금", min_value=0.0, format="%.2f")
    if st.button("추가", key="div_add"):
        if symbol and dividend >= 0:
            add_dividend(symbol, ex_date, pay_date, dividend)
            st.success(f"{symbol} 배당 정보가 저장되었습니다.")
            st.experimental_rerun()
        else:
            st.warning("심볼과 배당금을 입력하세요.")
    # Display existing records
    df = get_dividends()
    # Filter displayed dividends by selected symbol
    if symbol:
        df = df[df['Symbol'] == symbol]
    if not df.empty:
        st.dataframe(df, use_container_width=True)
    else:
        st.info("저장된 배당 정보가 없습니다.")
