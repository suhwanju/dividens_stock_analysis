import streamlit as st
#st.set_page_config(layout="wide")
from stock_visualization import stock_chart
from ks_stock_visualization import kor_stock_chart
from update_stock import update_stock_page

def main():
    # 사이드바 숨기기
    st.markdown("""
        <style>
        [data-testid="stSidebar"] {display: none;}
        </style>
        """, unsafe_allow_html=True)
    st.title("주식 정보 분석")
    tabs = ["미국 주식", "한국 주식", "배당 정보 입력"]
    selected_tab = st.tabs(tabs)
    
    with selected_tab[0]:  # 미국 주식 탭
        stock_chart()
    
    with selected_tab[1]:  # 한국 주식 탭
        kor_stock_chart()
    
    with selected_tab[2]:  # 배당 정보 탭
        update_stock_page()
    
if __name__ == "__main__":
    main()