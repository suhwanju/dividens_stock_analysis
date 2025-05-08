# 주식 정보 분석 툴

이 프로젝트는 Streamlit을 이용해 미국 및 한국 주식 시세와 배당 정보를 시각화하고 관리할 수 있는 웹 애플리케이션입니다.

## 주요 기능

- **미국 주식 시각화**: YFinance를 통해 데이터를 조회하고 캔들스틱 차트, 이동평균선(MA), 볼린저 밴드(Bollinger Bands), RSI, 스토캐스틱, 거래량 등을 표시합니다.
- **한국 주식 시각화**: Naver 금융 웹페이지에서 스크래핑하여 한국 주식 데이터를 조회하고 동일한 지표를 시각화합니다. 한글 종목명을 표시하여 사용성을 높였습니다.
- **배당 정보 관리**: 배당락(ex_date) 및 배당지급(pay_date) 일자를 포함한 배당 내역을 데이터베이스에 저장하고 조회할 수 있습니다.
- **투자 포트폴리오**: 사용자가 DB에 저장한 투자 내역을 표로 확인할 수 있으며, 수익률과 통화 기호(₩, $)를 함께 표시합니다.
- **포트폴리오 시나리오 분석**: 몬테카를로 시뮬레이션을 통해 다양한 시장 상황에서의 투자 성과를 예측하고 시각화합니다.
- **투자 위험 평가**: Value at Risk(VaR) 및 Conditional VaR(CVaR)를 계산하여 포트폴리오의 최대 예상 하락폭을 분석합니다.
- **시장 심리 지표**: CNN Fear & Greed Index를 시각화하여 현재 시장 심리와 추천 현금 배분 비율을 제시합니다.
- **Wide 레이아웃**: `main.py`에서 자동으로 와이드 모드로 설정하여 더 넓은 화면을 활용합니다.

---

## 요구사항

- Python 3.8 이상
- MySQL 서버
- Windows / macOS / Linux

### 라이브러리

`requirements.txt` 파일 참조:

```txt
streamlit==1.32.0
yfinance==0.2.36
plotly==5.19.0
mysql-connector-python==8.0.31
numpy==1.26.0
pandas==2.1.0
scipy==1.11.3
beautifulsoup4==4.12.2
requests==2.31.0
```

---

## 설치 방법

1. 저장소를 클론하거나 다운로드합니다.
   ```bash
   git clone <이곳에_저장소_URL>
   cd "d:/projects/Vibe Coding/US Stock"
   ```
2. 가상환경 생성 및 활성화
   ```bash
   python -m venv venv
   # Windows (PowerShell)
   .\venv\Scripts\Activate.ps1
   # macOS/Linux
   source venv/bin/activate
   ```
3. 의존성 설치
   ```bash
   pip install -r requirements.txt
   ```
4. MySQL에 `investdb` 데이터베이스 및 필요한 테이블이 자동으로 생성됩니다. 
   (DB 접속 정보는 `localhost`, 사용자 `root`, 비밀번호 `Shju6256#$`로 하드코딩되어 있습니다.)

---

## 실행 방법

```bash
streamlit run main.py
```

웹 브라우저가 열리면 다음 탭을 이용해 기능을 확인할 수 있습니다:

1. **미국 주식** 탭: 특정 티커(예: AAPL, TSLA 등)를 선택하고 각종 지표를 토글로 켜거나 끌 수 있습니다. Fear & Greed Index 게이지가 표시되어 시장 심리를 파악할 수 있습니다.
2. **한국 주식** 탭: 한글 종목명으로 주식을 선택하여 한국 주식 차트를 확인합니다.
3. **배당 정보** 탭: 종목 목록에서 심볼을 선택하고 배당락/지급 일자 및 금액을 입력하여 DB에 저장합니다. 아래 테이블에서 저장된 배당 내역을 확인할 수 있습니다.
4. **투자 포트폴리오** 탭: DB에 저장된 투자 내역을 표 형식으로 조회하며, 수익률과 통화 기호(₩, $)가 함께 표시됩니다.
5. **포트폴리오 시나리오** 탭: 투자 포트폴리오의 미래 가치를 시뮬레이션하고 위험 평가를 확인할 수 있습니다. 예측 기간, 시뮬레이션 횟수, 신뢰 구간을 사용자가 설정할 수 있습니다.

---

## 프로젝트 구조

```
US Stock/
├── main.py                # Streamlit 진입점 및 페이지 구성
├── stock_visualization.py # 미국 주식 관련 기능
├── ks_stock_visualization.py # 한국 주식 관련 기능
├── update_stock.py        # 배당 정보 등록/조회
├── portfolio.py           # 투자 포트폴리오 및 시나리오 분석 기능
├── portfolio_simulation.py # 포트폴리오 시뮬레이션 및 위험 평가 기능
├── fear_greed_index.py    # Fear & Greed Index 시각화 및 분석
├── requirements.txt       # 의존성
└── README.md              # 사용 설명서
```

---

## 주의 사항

- DB 접속 정보(호스트, 사용자, 비밀번호)는 코드에 하드코딩되어 있으므로 보안이 필요한 경우 환경 변수 또는 설정 파일로 분리하세요.
- Naver 스크래핑은 페이지 구조 변경 시 동작하지 않을 수 있습니다.

---

## 라이선스

MIT
