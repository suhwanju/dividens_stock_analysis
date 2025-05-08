import streamlit as st
import plotly.graph_objects as go
#import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import requests
from datetime import datetime, timedelta
import time

st.set_page_config(
    page_title="실시간 환율 모니터링",
    page_icon="💱",
    layout="wide"
)

# 사이드바 설정
st.sidebar.header("설정")
refresh_interval = st.sidebar.slider("자동 새로고침 간격(초)", 30, 300, 60, 30)
display_period = st.sidebar.selectbox("표시 기간", 
                                      ["1일", "1주일", "1개월", "3개월"], 
                                      index=1)

# 통화 선택 (KRW 기준)
default_currencies = ["USD", "EUR", "JPY", "CNY", "GBP"]
all_currencies = ["USD", "EUR", "JPY", "CNY", "GBP", "AUD", "CAD", "CHF", "HKD", "SGD"]
selected_currencies = st.sidebar.multiselect(
    "표시할 통화 선택",
    all_currencies,
    default=default_currencies
)

# 최대 데이터 포인트 설정
if display_period == "1일":
    max_points = 24  # 시간별
elif display_period == "1주일":
    max_points = 7   # 일별
elif display_period == "1개월":
    max_points = 30  # 일별
else:  # 3개월
    max_points = 90  # 일별

# get_exchange_rates 함수 수정
@st.cache_data(ttl=60*5)  # 5분 캐시 유지
def get_exchange_rates(base_currency="KRW", currencies=None, max_points=7):
    """
    Frankfurter API와 ExchangeRate-API를 조합해 환율 데이터를 가져옵니다.
    API 키가 필요 없는 완전 무료 서비스입니다.
    """
    if currencies is None:
        currencies = ["USD", "EUR", "JPY", "CNY", "GBP"]
    
    results = {}
    
    try:
        # 현재 날짜 가져오기
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        # 시작 날짜 계산 (max_points일 전)
        start_date = (datetime.now() - timedelta(days=max_points)).strftime('%Y-%m-%d')
        
        # ExchangeRate-API로 JPY와 다른 통화들의 데이터 가져오기 (무료, 키 불필요)
        for curr in currencies:
            try:
                # ExchangeRate-API는 USD 기준이므로, USD/KRW와 USD/통화 환율을 가져와 계산
                api_url = f"https://open.er-api.com/v6/time-series/{start_date}/{end_date}?base=USD&symbols=KRW,{curr}"
                response = requests.get(api_url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'time_series' in data:
                        dates = []
                        rates = []
                        
                        for date, rate_data in data['time_series'].items():
                            if 'KRW' in rate_data['rates'] and curr in rate_data['rates']:
                                dates.append(date)
                                # 통화/KRW 환율 = USD/KRW ÷ USD/통화
                                if curr == "KRW":
                                    # KRW는 자기 자신이므로 항상 1
                                    rates.append(1.0)
                                else:
                                    usd_krw = rate_data['rates']['KRW']
                                    usd_curr = rate_data['rates'][curr]
                                    curr_krw = usd_krw / usd_curr if usd_curr != 0 else 0
                                    rates.append(curr_krw)
                        
                        if dates:
                            # 날짜순 정렬
                            date_rate_pairs = list(zip(dates, rates))
                            date_rate_pairs.sort()  # 날짜 기준으로 정렬
                            dates, rates = zip(*date_rate_pairs)
                            
                            results[curr] = {"dates": dates, "rates": rates}
            except Exception as e:
                st.warning(f"{curr} 통화 데이터를 가져오는데 실패했습니다: {str(e)}")
        
        # 가짜 데이터 생성 옵션 (API 호출 한도 초과 또는 서비스 중단 시 테스트용)
        if not results or st.checkbox("API 이슈 시 테스트 데이터 사용", value=False):
            if not results:
                st.warning("실제 데이터를 가져오지 못해 테스트 데이터를 사용합니다.")
            else:
                st.warning("테스트 데이터를 사용합니다.")
            
            # 테스트 데이터 생성
            fake_dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(max_points)]
            fake_dates.reverse()  # 날짜순 정렬
            
            base_rates = {
                "USD": 1300,
                "EUR": 1400,
                "JPY": 8.5,
                "CNY": 180,
                "GBP": 1650,
                "AUD": 850,
                "CAD": 950,
                "CHF": 1450,
                "HKD": 165,
                "SGD": 960
            }
            
            # 선택된 통화별 임의의 변동성 있는 환율 데이터 생성
            for curr in currencies:
                if curr in base_rates:
                    base_rate = base_rates[curr]
                    
                    # 약간의 무작위 변동 추가
                    import random
                    random.seed(42)  # 재현 가능성을 위한 시드 설정
                    
                    fake_rates = []
                    for i in range(len(fake_dates)):
                        # 최대 3% 변동
                        change = random.uniform(-0.03, 0.03)
                        rate = base_rate * (1 + change)
                        fake_rates.append(rate)
                    
                    results[curr] = {"dates": fake_dates, "rates": fake_rates}
            
    except Exception as e:
        st.error(f"환율 데이터 API 요청 중 오류: {str(e)}")
    
    return results


    
import yfinance as yf

def get_exchange_rates_from_yahoo(currencies=None, max_points=30):
   
    """
    Yahoo Finance API를 사용하여 환율 데이터를 가져옵니다.
    """
    all_data = pd.DataFrame()

    # Calculate dates if they're not already provided
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Adjust as needed

    # Prepare Yahoo Finance currency pairs
    if currencies is None:
        currencies = ["USD", "EUR", "JPY", "CNY", "GBP"]
    currency_pairs = [f"{curr}KRW=X" for curr in currencies]  # Format for Yahoo Finance

    for pair in currency_pairs:
        try:
            # Add retry mechanism with exponential backoff
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    ticker = yf.Ticker(pair)
                    # Fetch historical data from Yahoo Finance
                    data = ticker.history(start=start_date, end=end_date)
                    if not data.empty:
                        data = data[['Close']].rename(columns={'Close': pair})
                        
                        if all_data.empty:
                            all_data = data
                        else:
                            all_data = all_data.join(data)
                        break
                    else:
                        if attempt < max_retries - 1:
                            time.sleep(2 ** attempt)  # Exponential backoff
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    else:
                        print(f"Failed to fetch data for {pair}: {str(e)}")
        except Exception as e:
            print(f"Error processing {pair}: {str(e)}")
    
    # Convert DataFrame to dict with dates and rates for compatibility
    results = {}
    if not all_data.empty:
        for col in all_data.columns:
            series = all_data[col]
            # Extract currency code from column name e.g. 'USDKRW=X'
            if col.endswith("KRW=X"):
                currency = col.replace("KRW=X", "")
            else:
                currency = col
            dates = [d.strftime('%Y-%m-%d') for d in series.index]
            rates = series.tolist()
            results[currency] = {"dates": dates, "rates": rates}
    return results

def visualize_exchange_rates(exchange_data):
    if not exchange_data:
        st.warning("표시할 데이터가 없습니다.")
        return
    
    fig = go.Figure()
    
    # 데이터 추가
    for currency, data in exchange_data.items():
        fig.add_trace(go.Scatter(
            x=data["dates"],
            y=data["rates"],
            mode='lines+markers',
            name=f'{currency}/KRW',
            hovertemplate=f'%{{y:.2f}} KRW<extra>{currency}</extra>'
        ))
    
    # 전체 날짜 범위 찾기
    all_dates = []
    for data in exchange_data.values():
        if "dates" in data and data["dates"]:
            all_dates.extend(data["dates"])
    
    if all_dates:
        # 중복 제거 및 정렬
        all_dates = sorted(set(all_dates))
        
        # 날짜를 datetime 객체로 변환
        date_objects = [datetime.strptime(date, '%Y-%m-%d') for date in all_dates]
        
        # 시작일과 종료일
        start_date = min(date_objects)
        end_date = max(date_objects)
        
        # 매월 1일 찾기
        month_firsts = []
        current = datetime(start_date.year, start_date.month, 1)
        
        # 시작일이 1일이 아니면 다음 달 1일부터 시작
        if start_date.day > 1:
            if start_date.month == 12:
                current = datetime(start_date.year + 1, 1, 1)
            else:
                current = datetime(start_date.year, start_date.month + 1, 1)
        
        # 종료일까지의 모든 월의 1일 찾기
        while current <= end_date:
            month_firsts.append(current.strftime('%Y-%m-%d'))
            
            # 다음 달 1일로 이동
            if current.month == 12:
                current = datetime(current.year + 1, 1, 1)
            else:
                current = datetime(current.year, current.month + 1, 1)
        
        # 매월 1일에 붉은색 수직선 추가
        for date in month_firsts:
            fig.add_shape(
                type="line",
                x0=date,
                x1=date,
                y0=0,
                y1=1,
                yref="paper",  # y축 기준을 상대적으로 설정 (0~1)
                line=dict(
                    color="red",
                    width=1,
                    dash="dot",
                ),
                name=f"1일 ({date})"
            )
            
            # 날짜 라벨 추가 (선택적)
            month_name = datetime.strptime(date, '%Y-%m-%d').strftime('%b')  # 월 이름 약어
            fig.add_annotation(
                x=date,
                y=1.05,
                yref="paper",
                text=f"{month_name} 1일",
                showarrow=False,
                font=dict(
                    color="red",
                    size=10
                ),
                textangle=-45
            )
    
    fig.update_layout(
        title="KRW 대비 주요 통화 환율 추이",
        xaxis_title="날짜",
        yaxis_title="환율 (KRW)",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # X축에 회색 점선 그리드 추가
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        griddash='dot'
    )
    
    # Y축도 일관성을 위해 동일한 스타일 적용
    fig.update_yaxes(
        gridcolor='lightgray',
        griddash='dot'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def visualize_exchange_volatility(exchange_data):
    """환율의 일별 변동성을 그래프로 시각화합니다."""
    if not exchange_data:
        st.warning("표시할 데이터가 없습니다.")
        return
    
    # 변동성 데이터 준비
    volatility_data = {}
    
    for currency, data in exchange_data.items():
        if 'dates' in data and 'rates' in data and len(data['dates']) > 1:
            dates = data['dates']
            rates = data['rates']
            
            # 날짜별로 정렬
            date_rate_pairs = sorted(zip(dates, rates), key=lambda x: x[0])
            sorted_dates = [pair[0] for pair in date_rate_pairs]
            sorted_rates = [pair[1] for pair in date_rate_pairs]
            
            # 일별 변동률 계산 (퍼센트 단위)
            daily_changes = []
            daily_dates = []
            
            for i in range(1, len(sorted_dates)):
                prev_rate = sorted_rates[i-1]
                curr_rate = sorted_rates[i]
                
                if prev_rate > 0:  # 0으로 나누기 방지
                    daily_change = ((curr_rate - prev_rate) / prev_rate) * 100  # 퍼센트로 변환
                    daily_changes.append(daily_change)
                    daily_dates.append(sorted_dates[i])
            
            if daily_dates:
                volatility_data[currency] = {
                    "dates": daily_dates,
                    "changes": daily_changes
                }
    
    if not volatility_data:
        st.warning("변동성 계산에 필요한 충분한 데이터가 없습니다.")
        return
    
    # 변동성 그래프 그리기
    fig = go.Figure()
    
    for currency, data in volatility_data.items():
        fig.add_trace(go.Scatter(
    x=data["dates"],
    y=data["changes"],
    mode='lines+markers',
    name=f'{currency}/KRW',
    hovertemplate=f'변동률: %{{y:.2f}}%<extra>{currency}</extra>'
))
    
    # 전체 날짜 범위 찾기
    all_dates = []
    for data in volatility_data.values():
        all_dates.extend(data["dates"])
    
    if all_dates:
        # 중복 제거 및 정렬
        all_dates = sorted(set(all_dates))
        
        # 0% 기준선 추가
        fig.add_shape(
            type="line",
            x0=min(all_dates),
            x1=max(all_dates),
            y0=0,
            y1=0,
            line=dict(
                color="black",
                width=1,
                dash="dash",
            )
        )
    
    fig.update_layout(
        title="KRW 대비 주요 통화 일별 변동률 (%)",
        xaxis_title="날짜",
        yaxis_title="일별 변동률 (%)",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # X축에 회색 점선 그리드 추가
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        griddash='dot'
    )
    
    # Y축도 일관성을 위해 동일한 스타일 적용
    fig.update_yaxes(
        gridcolor='lightgray',
        griddash='dot'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def visualize_combined_exchange_data(exchange_data):
    """환율 추이와 일별 변동성을 함께 표시하는 통합 그래프"""
    if not exchange_data:
        st.warning("표시할 데이터가 없습니다.")
        return
    
    # 그래프를 2행 1열로 구성 (상단: 환율 추이, 하단: 변동성)
    fig = make_subplots(rows=2, cols=1, 
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        subplot_titles=("환율 추이", "일별 변동률 (%)"))
    
    # 통화별 색상 지정
    colors = {
        "USD": "#1f77b4", "EUR": "#ff7f0e", "JPY": "#2ca02c",
        "CNY": "#d62728", "GBP": "#9467bd", "AUD": "#8c564b",
        "CAD": "#e377c2", "CHF": "#7f7f7f", "HKD": "#bcbd22",
        "SGD": "#17becf"
    }
    
    for currency, data in exchange_data.items():
        if 'dates' in data and 'rates' in data and len(data['dates']) > 1:
            # 날짜별로 정렬
            date_rate_pairs = sorted(zip(data['dates'], data['rates']), key=lambda x: x[0])
            sorted_dates = [pair[0] for pair in date_rate_pairs]
            sorted_rates = [pair[1] for pair in date_rate_pairs]
            
            # 1. 환율 추이 그래프 (상단)
            fig.add_trace(
                go.Scatter(
                    x=sorted_dates,
                    y=sorted_rates,
                    mode='lines',
                    name=f'{currency}/KRW (추이)',
                    line=dict(color=colors.get(currency, "#000000")),
                    hovertemplate='%{y:.2f} KRW<extra></extra>'
                ),
                row=1, col=1
            )
            
            # 2. 변동성 계산 및 그래프 (하단)
            daily_changes = []
            daily_dates = []
            
            for i in range(1, len(sorted_dates)):
                prev_rate = sorted_rates[i-1]
                curr_rate = sorted_rates[i]
                
                if prev_rate > 0:  # 0으로 나누기 방지
                    daily_change = ((curr_rate - prev_rate) / prev_rate) * 100
                    daily_changes.append(daily_change)
                    daily_dates.append(sorted_dates[i])
            
            if daily_dates:
                fig.add_trace(
                    go.Scatter(
                        x=daily_dates,
                        y=daily_changes,
                        mode='lines',
                        name=f'{currency}/KRW (변동률)',
                        line=dict(color=colors.get(currency, "#000000"), dash='dot'),
                        hovertemplate='변동률: %{y:.2f}%<extra></extra>'
                    ),
                    row=2, col=1
                )
    
    # 0% 기준선 추가 (변동성 그래프)
    all_dates = []
    for data in exchange_data.values():
        if 'dates' in data:
            all_dates.extend(data['dates'])
    
    if all_dates:
        # 중복 제거 및 정렬
        all_dates = sorted(set(all_dates))
        
        # 0% 기준선 추가
        fig.add_shape(
            type="line",
            x0=min(all_dates),
            x1=max(all_dates),
            y0=0,
            y1=0,
            line=dict(
                color="black",
                width=1,
                dash="dash",
            ),
            row=2, col=1
        )
    
    # 그래프 스타일 설정
    fig.update_layout(
        height=700,  # 그래프 높이 조정
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # X축에 회색 점선 그리드 추가
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        griddash='dot'
    )
    
    # Y축도 일관성을 위해 동일한 스타일 적용
    fig.update_yaxes(
        gridcolor='lightgray',
        griddash='dot'
    )
    
    # 축 레이블 추가
    fig.update_yaxes(title_text="환율 (KRW)", row=1, col=1)
    fig.update_yaxes(title_text="변동률 (%)", row=2, col=1)
    fig.update_xaxes(title_text="날짜", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

# 실시간 환율 표시 함수
def show_current_rates(exchange_data):
    if not exchange_data:
        return
    
    cols = st.columns(len(exchange_data))
    
    for i, (currency, data) in enumerate(exchange_data.items()):
        if data["rates"]:
            current_rate = data["rates"][-1]
            prev_rate = data["rates"][-2] if len(data["rates"]) > 1 else current_rate
            delta = current_rate - prev_rate
            delta_percent = (delta / prev_rate) * 100 if prev_rate else 0
            
            cols[i].metric(
                f"{currency}/KRW",
                f"{current_rate:.2f}",
                f"{delta:.2f} ({delta_percent:.2f}%)"
            )

# 메인 애플리케이션 로직
def main():
    # 프로그레스 표시
    progress_placeholder = st.empty()
    with progress_placeholder.container():
        st.info("환율 데이터를 가져오는 중...")
    
    # 데이터 가져오기
    exchange_data = get_exchange_rates(currencies=selected_currencies)
    
    # 프로그레스 제거
    progress_placeholder.empty()
    
    # 현재 시간 표시
    now = datetime.now()
    st.caption(f"마지막 업데이트: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 현재 환율 표시
    show_current_rates(exchange_data)
    
    # 그래프 시각화
    visualize_exchange_rates(exchange_data)
    
    # 자동 새로고침
    if st.sidebar.button("새로고침"):
        st.experimental_rerun()
    
    # 설명 추가
    with st.expander("사용 방법 및 참고 사항"):
        st.markdown("""
        - **API 키 설정**: 실제 사용을 위해서는 Alpha Vantage에서 API 키를 발급받아 코드에 삽입해야 합니다.
        - **자동 새로고침**: 왼쪽 사이드바에서 설정한 간격으로 데이터가 자동으로 새로고침됩니다.
        - **통화 선택**: 원하는 통화를 선택하여 환율 변동을 확인할 수 있습니다.
        - **기간 선택**: 표시할 기간을 선택할 수 있습니다.
        """)

#if __name__ == "__main__":
#    main()
    
    # 자동 새로고침 설정
#    time.sleep(refresh_interval)
#    st.experimental_rerun()