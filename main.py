# # main.py (오류 수정 완료)
# from fastapi import FastAPI, Request
# from fastapi.responses import HTMLResponse, JSONResponse
# from fastapi.templating import Jinja2Templates
# from fastapi.staticfiles import StaticFiles
# import yfinance as yf
# import pandas as pd
# import numpy as np
# from datetime import datetime
# import logging
# import traceback

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = FastAPI()
# templates = Jinja2Templates(directory=".")
# app.mount("/static", StaticFiles(directory="static"), name="static")

# @app.get("/", response_class=HTMLResponse)
# async def serve_dashboard(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

# def calculate_technical_indicators(df: pd.DataFrame):
#     if df.empty or len(df) < 2: return {}
#     prices = df['Close']
#     current_price = prices.iloc[-1]
#     indicators = {}
    
#     for period in [20, 60, 120]:
#         if len(prices) >= period:
#             ma = prices.rolling(window=period).mean().iloc[-1]
#             indicators[f"ma_{period}"] = round(ma, 2) if pd.notna(ma) else 0
#         else:
#             indicators[f"ma_{period}"] = 0
            
#     window_52w = min(len(prices), 252)
#     high_52w = prices.tail(window_52w).max()
#     low_52w = prices.tail(window_52w).min()
#     indicators["high_52w"] = round(high_52w, 2) if pd.notna(high_52w) else current_price
#     indicators["low_52w"] = round(low_52w, 2) if pd.notna(low_52w) else current_price
    
#     if len(prices) >= 15:
#         delta = prices.diff()
#         gain = delta.where(delta > 0, 0).rolling(window=14).mean()
#         loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
#         rs = gain / loss.replace(0, 1e-9)
#         rsi = 100 - (100 / (1 + rs.iloc[-1]))
#         indicators["rsi_14"] = round(rsi, 2) if pd.notna(rsi) else 50
#     else:
#         indicators["rsi_14"] = 50
        
#     return indicators

# def analyze_trend_and_signals(df: pd.DataFrame):
#     if df.empty or len(df) < 60:
#         return { "trend_analysis": "데이터 부족", "buy_signal_strength": 0, "recommended_buy_price": 0, "recommended_buy_price_explanation": "데이터 부족", "trend_direction": "중립", "momentum_score": 0, "price_position_52w": 0 }
    
#     # --- FIX: 이 부분을 추가하여 indicators 변수를 먼저 정의합니다. ---
#     indicators = calculate_technical_indicators(df)
#     # -----------------------------------------------------------

#     prices = df['Close']
#     current_price = prices.iloc[-1]
#     ma_5 = prices.rolling(window=5).mean().iloc[-1] if len(prices) >= 5 else current_price
#     ma_20 = prices.rolling(window=20).mean().iloc[-1] if len(prices) >= 20 else current_price
#     ma_60 = prices.rolling(window=60).mean().iloc[-1] if len(prices) >= 60 else current_price
    
#     trend_score = sum([1 for ma_short, ma_long in [(current_price, ma_5), (ma_5, ma_20), (ma_20, ma_60)] if pd.notna(ma_short) and pd.notna(ma_long) and ma_short > ma_long])
#     trend_direction = "상승" if trend_score >= 2 else "하락" if trend_score <= 1 else "중립"
    
#     momentum_5d = ((current_price / prices.iloc[-6]) - 1) * 100 if len(prices) > 5 else 0
#     momentum_20d = ((current_price / prices.iloc[-21]) - 1) * 100 if len(prices) > 20 else 0
#     momentum_score = (momentum_5d * 0.3 + momentum_20d * 0.7)
    
#     buy_signal_strength = 0
#     # --- FIX: 정의된 indicators 변수에서 rsi 값을 가져옵니다. ---
#     rsi = indicators.get("rsi_14", 50)
#     if rsi < 30: buy_signal_strength += 3
#     elif rsi < 40: buy_signal_strength += 2
#     elif rsi > 70: buy_signal_strength -= 2
    
#     if trend_direction == "상승": buy_signal_strength += 2
#     elif trend_direction == "하락": buy_signal_strength -= 1
    
#     # --- FIX: 정의된 indicators 변수에서 52주 고가/저가 값을 가져옵니다. ---
#     high_52w, low_52w = indicators.get("high_52w", current_price), indicators.get("low_52w", current_price)
#     price_position = ((current_price - low_52w) / (high_52w - low_52w) * 100) if high_52w > low_52w else 50
#     if price_position < 30: buy_signal_strength += 1
#     elif price_position > 80: buy_signal_strength -= 1
    
#     valid_supports = [s for s in [ma_20, ma_60] if pd.notna(s) and s < current_price and s > current_price * 0.9]
#     if valid_supports:
#         recommended_buy_price = max(valid_supports)
#         explanation = f"지지선(이평선)인 {round(recommended_buy_price, 2)}원 근처 매수 고려."
#     else:
#         recommended_buy_price = current_price * 0.98
#         explanation = "뚜렷한 지지선 부재. 현제가 대비 2% 하락 지점 고려."
        
#     return { "trend_analysis": f"{trend_direction} 추세 (점수: {trend_score}/3)", "buy_signal_strength": max(0, min(10, buy_signal_strength + 5)), "recommended_buy_price": round(recommended_buy_price, 2), "recommended_buy_price_explanation": explanation, "trend_direction": trend_direction, "momentum_score": round(momentum_score, 2), "price_position_52w": round(price_position, 1) if pd.notna(price_position) else 0 }

# def get_signal_explanation(strength):
#     if strength >= 8: return "강력 매수 고려 구간"
#     if strength >= 6: return "매수 고려 구간"
#     if strength >= 4: return "관망 또는 분할 매수"
#     return "매수 주의 구간"

# def generate_technical_review(data):
#     cp = data['current_price']
#     ma20 = data['ma_20']
#     bu = data['bollinger_upper']
#     bl = data['bollinger_lower']
    
#     review = {}
    
#     # 이동평균선 분석
#     if cp > ma20:
#         diff = (cp / ma20 - 1) * 100
#         review['ma_analysis'] = f"주가가 20일선보다 {diff:.1f}% 위에 있어 단기 상승 추세입니다. 20일선을 지지선으로 보고 대응할 수 있습니다."
#     else:
#         diff = (ma20 / cp - 1) * 100
#         review['ma_analysis'] = f"주가가 20일선보다 {diff:.1f}% 아래에 있어 단기 조정 국면입니다. 20일선 돌파 여부가 중요합니다."

#     # 볼린저밴드 분석
#     band_width = (bu - bl)
#     if band_width == 0:
#         review['bb_analysis'] = "변동성이 매우 낮아 분석이 어렵습니다."
#     else:
#         pos_in_band = (cp - bl) / band_width
#         if pos_in_band > 1.0:
#             review['bb_analysis'] = "밴드 상단을 강하게 돌파한 과매수 상태입니다. 추격 매수는 주의가 필요하며, 차익 실현을 고려할 수 있습니다."
#         elif pos_in_band > 0.8:
#             review['bb_analysis'] = "밴드 상단에 근접해 상승 에너지가 강하지만, 단기 과열 가능성이 있습니다. 일부 분할 매도를 고려할 수 있습니다."
#         elif pos_in_band < 0.0:
#             review['bb_analysis'] = "밴드 하단을 이탈한 과매도 상태입니다. 기술적 반등을 노린 분할 매수 전략이 유효할 수 있습니다."
#         elif pos_in_band < 0.2:
#             review['bb_analysis'] = "밴드 하단에 근접해 저가 매수세 유입을 기대할 수 있습니다. 반등 시 중단선(20일선)이 1차 저항이 됩니다."
#         else:
#             review['bb_analysis'] = "밴드 내에서 움직이고 있어 변동성이 안정적입니다. 중단선(20일선) 지지/저항 여부를 확인하는 것이 중요합니다."
            
#     # 종합 의견
#     change_text = f"{abs(data['price_change_percent']):.2f}% {'상승' if data['price_change_percent'] >= 0 else '하락'}했습니다."
#     if cp > ma20 and pos_in_band > 0.5:
#         review['summary'] = f"전일 대비 {change_text} 20일선 위에 안착하고 볼린저밴드 상단부에서 움직이는 긍정적 흐름입니다. 추세 추종 관점의 접근이 유효합니다."
#     elif cp < ma20 and pos_in_band < 0.5:
#         review['summary'] = f"전일 대비 {change_text} 20일선 아래에서 조정을 받고 있으며 볼린저밴드 하단부에 위치합니다. 과매도 구간 진입 시 기술적 반등을 노려볼 수 있습니다."
#     else:
#         review['summary'] = f"전일 대비 {change_text} 주가가 20일선 근처에서 횡보하며 방향성을 탐색하는 중립적인 구간입니다."
        
#     return review

# @app.get("/api/krx-data/{ticker}", response_class=JSONResponse)
# async def get_krx_data(ticker: str):
#     try:
#         logger.info(f"Yahoo Finance 데이터 요청: {ticker}")
#         etf = yf.Ticker(f"{ticker}.KS")
#         df = etf.history(period="3y", auto_adjust=True)
#         if df.empty: return JSONResponse(status_code=404, content={"message": f"데이터 없음: {ticker}"})
        
#         current_price = df['Close'].iloc[-1]
        
#         chart_df = df.tail(63).copy()
#         high_3m = chart_df['High'].max()
        
#         chart_df['ma_20'] = chart_df['Close'].rolling(window=20).mean()
#         chart_df['ma_60'] = chart_df['Close'].rolling(window=60).mean()
#         chart_df['std_20'] = chart_df['Close'].rolling(window=20).std()
#         chart_df['bollinger_upper'] = chart_df['ma_20'] + (chart_df['std_20'] * 2)
#         chart_df['bollinger_lower'] = chart_df['ma_20'] - (chart_df['std_20'] * 2)

#         def series_to_list(series):
#             return [round(v, 2) if pd.notna(v) else None for v in series]

#         chart_data = {
#             "dates": [d.strftime('%Y-%m-%d') for d in chart_df.index],
#             "prices": series_to_list(chart_df['Close']),
#             "ma_20": series_to_list(chart_df['ma_20']),
#             "ma_60": series_to_list(chart_df['ma_60']),
#             "bollinger_upper": series_to_list(chart_df['bollinger_upper']),
#             "bollinger_lower": series_to_list(chart_df['bollinger_lower'])
#         }
        
#         indicators = calculate_technical_indicators(df)
#         trend = analyze_trend_and_signals(df)
        
#         response_data = {
#             "ticker_info": {"name": etf.info.get('longName', ticker)},
#             "market_data": {
#                 "current_price": round(current_price, 2),
#                 "price_change_percent": round(((current_price / df['Close'].iloc[-2]) - 1) * 100, 2),
#                 "volume": int(df['Volume'].iloc[-1]),
#                 "data_source": "Yahoo Finance"
#             },
#             "basic_info": {
#                 "high_52w": indicators.get('high_52w', 0),
#                 "low_52w": indicators.get('low_52w', 0),
#                 "high_3m": round(high_3m, 2) if pd.notna(high_3m) else 0,
#                 "price_range_percent": round(((current_price - indicators.get('low_52w', 0)) / max(indicators.get('high_52w', 1) - indicators.get('low_52w', 0), 1)) * 100, 2)
#             },
#             "trading_signals": {
#                 "buy_signal_strength": trend["buy_signal_strength"],
#                 "recommended_buy_price": trend["recommended_buy_price"],
#                 "recommended_buy_price_explanation": trend["recommended_buy_price_explanation"],
#                 "buy_recommendation": "강력 매수" if trend["buy_signal_strength"] >= 8 else "매수" if trend["buy_signal_strength"] >= 6 else "관망" if trend["buy_signal_strength"] >= 4 else "주의",
#                 "signal_explanation": get_signal_explanation(trend["buy_signal_strength"])
#             },
#             "chart_data": chart_data
#         }
#         return JSONResponse(content=response_data)
        
#     except Exception as e:
#         logger.error(f"오류 ({ticker}): {e}\n{traceback.format_exc()}")
#         return JSONResponse(status_code=500, content={"message": f"서버 오류 발생: {e}"})

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

# main.py (최종 수정 버전)
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import traceback

# --- (기존 코드와 동일한 부분은 생략) ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
templates = Jinja2Templates(directory=".")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def calculate_technical_indicators(df: pd.DataFrame):
    # ... (이전과 동일)
    if df.empty or len(df) < 2: return {}
    prices = df['Close']
    indicators = {}
    for period in [20, 60, 120]:
        if len(prices) >= period:
            ma = prices.rolling(window=period).mean().iloc[-1]
            indicators[f"ma_{period}"] = round(ma, 2) if pd.notna(ma) else 0
        else:
            indicators[f"ma_{period}"] = 0
    window_52w = min(len(prices), 252)
    indicators["high_52w"] = round(prices.tail(window_52w).max(), 2)
    indicators["low_52w"] = round(prices.tail(window_52w).min(), 2)
    if len(prices) >= 15:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss.replace(0, 1e-9)
        rsi = 100 - (100 / (1 + rs.iloc[-1]))
        indicators["rsi_14"] = round(rsi, 2) if pd.notna(rsi) else 50
    else:
        indicators["rsi_14"] = 50
    return indicators

def analyze_trend_and_signals(df: pd.DataFrame, indicators):
    # ... (이전과 거의 동일, indicators를 인자로 받도록 수정)
    if df.empty or len(df) < 60: return { "trend_analysis": "데이터 부족", "buy_signal_strength": 0, "recommended_buy_price": 0, "recommended_buy_price_explanation": "데이터 부족", "trend_direction": "중립", "momentum_score": 0, "price_position_52w": 0 }
    prices = df['Close']
    current_price = prices.iloc[-1]
    ma_5, ma_20, ma_60 = prices.rolling(window=5).mean().iloc[-1], indicators.get("ma_20"), indicators.get("ma_60")
    trend_score = sum([1 for s, l in [(current_price, ma_5), (ma_5, ma_20), (ma_20, ma_60)] if pd.notna(s) and pd.notna(l) and s > l])
    trend_direction = "상승" if trend_score >= 2 else "하락" if trend_score <= 1 else "중립"
    momentum_score = (((current_price / prices.iloc[-6]) - 1) * 100 * 0.3 + ((current_price / prices.iloc[-21]) - 1) * 100 * 0.7) if len(prices) > 20 else 0
    buy_signal_strength = 0
    rsi = indicators.get("rsi_14", 50)
    if rsi < 30: buy_signal_strength += 3
    elif rsi < 40: buy_signal_strength += 2
    elif rsi > 70: buy_signal_strength -= 2
    if trend_direction == "상승": buy_signal_strength += 2
    elif trend_direction == "하락": buy_signal_strength -= 1
    high_52w, low_52w = indicators.get("high_52w"), indicators.get("low_52w")
    price_position = ((current_price - low_52w) / (high_52w - low_52w) * 100) if high_52w > low_52w else 50
    if price_position < 30: buy_signal_strength += 1
    elif price_position > 80: buy_signal_strength -= 1
    valid_supports = [s for s in [ma_20, ma_60] if pd.notna(s) and s < current_price and s > current_price * 0.9]
    if valid_supports:
        rec_price = max(valid_supports)
        explanation = f"지지선({round(rec_price,0)}원) 근처 매수 고려."
    else:
        rec_price = current_price * 0.98
        explanation = "지지선 부재. 현제가 대비 2% 하락 시 고려."
    return { "trend_analysis": f"{trend_direction} 추세 ({trend_score}/3)", "buy_signal_strength": max(0, min(10, buy_signal_strength + 5)), "recommended_buy_price": round(rec_price, 2), "recommended_buy_price_explanation": explanation, "trend_direction": trend_direction, "momentum_score": round(momentum_score, 2), "price_position_52w": round(price_position, 1) if pd.notna(price_position) else 0 }

def get_signal_explanation(strength):
    # ... (이전과 동일)
    if strength >= 8: return "강력 매수 고려 구간"
    if strength >= 6: return "매수 고려 구간"
    if strength >= 4: return "관망 또는 분할 매수"
    return "매수 주의 구간"

# ### START: 분석 텍스트 생성 함수 추가 ###
def generate_technical_review(data):
    cp = data['current_price']
    ma20 = data['ma_20']
    bu = data['bollinger_upper']
    bl = data['bollinger_lower']
    
    review = {}
    
    # 이동평균선 분석
    if cp > ma20:
        diff = (cp / ma20 - 1) * 100
        review['ma_analysis'] = f"주가가 20일선보다 {diff:.1f}% 위에 있어 단기 상승 추세입니다. 20일선을 지지선으로 보고 대응할 수 있습니다."
    else:
        diff = (ma20 / cp - 1) * 100
        review['ma_analysis'] = f"주가가 20일선보다 {diff:.1f}% 아래에 있어 단기 조정 국면입니다. 20일선 돌파 여부가 중요합니다."

    # 볼린저밴드 분석
    band_width = (bu - bl)
    if band_width == 0:
        review['bb_analysis'] = "변동성이 매우 낮아 분석이 어렵습니다."
    else:
        pos_in_band = (cp - bl) / band_width
        if pos_in_band > 1.0:
            review['bb_analysis'] = "밴드 상단을 강하게 돌파한 과매수 상태입니다. 추격 매수는 주의가 필요하며, 차익 실현을 고려할 수 있습니다."
        elif pos_in_band > 0.8:
            review['bb_analysis'] = "밴드 상단에 근접해 상승 에너지가 강하지만, 단기 과열 가능성이 있습니다. 일부 분할 매도를 고려할 수 있습니다."
        elif pos_in_band < 0.0:
            review['bb_analysis'] = "밴드 하단을 이탈한 과매도 상태입니다. 기술적 반등을 노린 분할 매수 전략이 유효할 수 있습니다."
        elif pos_in_band < 0.2:
            review['bb_analysis'] = "밴드 하단에 근접해 저가 매수세 유입을 기대할 수 있습니다. 반등 시 중단선(20일선)이 1차 저항이 됩니다."
        else:
            review['bb_analysis'] = "밴드 내에서 움직이고 있어 변동성이 안정적입니다. 중단선(20일선) 지지/저항 여부를 확인하는 것이 중요합니다."
            
    # 종합 의견
    change_text = f"{abs(data['price_change_percent']):.2f}% {'상승' if data['price_change_percent'] >= 0 else '하락'}했습니다."
    if cp > ma20 and pos_in_band > 0.5:
        review['summary'] = f"전일 대비 {change_text} 20일선 위에 안착하고 볼린저밴드 상단부에서 움직이는 긍정적 흐름입니다. 추세 추종 관점의 접근이 유효합니다."
    elif cp < ma20 and pos_in_band < 0.5:
        review['summary'] = f"전일 대비 {change_text} 20일선 아래에서 조정을 받고 있으며 볼린저밴드 하단부에 위치합니다. 과매도 구간 진입 시 기술적 반등을 노려볼 수 있습니다."
    else:
        review['summary'] = f"전일 대비 {change_text} 주가가 20일선 근처에서 횡보하며 방향성을 탐색하는 중립적인 구간입니다."
        
    return review
# ### END: 분석 텍스트 생성 함수 추가 ###

@app.get("/api/krx-data/{ticker}", response_class=JSONResponse)
async def get_krx_data(ticker: str):
    try:
        logger.info(f"Yahoo Finance 데이터 요청: {ticker}")
        etf = yf.Ticker(f"{ticker}.KS")
        df = etf.history(period="3y", auto_adjust=True)
        if df.empty: return JSONResponse(status_code=404, content={"message": f"데이터 없음: {ticker}"})
        
        current_price = df['Close'].iloc[-1]
        
        chart_df = df.tail(120).copy() # 리뷰 계산을 위해 데이터 기간 확장
        chart_df['ma_20'] = chart_df['Close'].rolling(window=20).mean()
        chart_df['ma_60'] = chart_df['Close'].rolling(window=60).mean()
        chart_df['std_20'] = chart_df['Close'].rolling(window=20).std()
        chart_df['bollinger_upper'] = chart_df['ma_20'] + (chart_df['std_20'] * 2)
        chart_df['bollinger_lower'] = chart_df['ma_20'] - (chart_df['std_20'] * 2)
        
        # 차트 시각화는 최근 63일 데이터만 사용
        view_chart_df = chart_df.tail(63)
        def series_to_list(series): return [round(v, 2) if pd.notna(v) else None for v in series]

        chart_data = {
            "dates": [d.strftime('%Y-%m-%d') for d in view_chart_df.index],
            "prices": series_to_list(view_chart_df['Close']), "ma_20": series_to_list(view_chart_df['ma_20']),
            "ma_60": series_to_list(view_chart_df['ma_60']), "bollinger_upper": series_to_list(view_chart_df['bollinger_upper']),
            "bollinger_lower": series_to_list(view_chart_df['bollinger_lower'])
        }
        
        indicators = calculate_technical_indicators(df)
        trend = analyze_trend_and_signals(df, indicators)
        price_change_percent = round(((current_price / df['Close'].iloc[-2]) - 1) * 100, 2)
        
        # ### START: 기술적 분석 리뷰 생성 ###
        review_data = {
            'current_price': current_price,
            'price_change_percent': price_change_percent,
            'ma_20': indicators.get('ma_20'),
            'bollinger_upper': chart_df['bollinger_upper'].iloc[-1],
            'bollinger_lower': chart_df['bollinger_lower'].iloc[-1]
        }
        technical_review = generate_technical_review(review_data)
        # ### END: 기술적 분석 리뷰 생성 ###
        
        response_data = {
            "ticker_info": {"name": etf.info.get('longName', ticker)},
            "market_data": { "current_price": round(current_price, 2), "price_change_percent": price_change_percent, "volume": int(df['Volume'].iloc[-1]), "data_source": "Yahoo Finance" },
            "basic_info": { "high_52w": indicators.get('high_52w', 0), "low_52w": indicators.get('low_52w', 0), "high_3m": round(view_chart_df['High'].max(), 2), "price_range_percent": round(((current_price - indicators.get('low_52w', 0)) / max(indicators.get('high_52w', 1) - indicators.get('low_52w', 0), 1)) * 100, 2) },
            "trading_signals": { **trend, "buy_recommendation": "강력 매수" if trend["buy_signal_strength"] >= 8 else "매수" if trend["buy_signal_strength"] >= 6 else "관망" if trend["buy_signal_strength"] >= 4 else "주의", "signal_explanation": get_signal_explanation(trend["buy_signal_strength"]) },
            "chart_data": chart_data,
            "technical_review": technical_review # 응답에 추가
        }
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"오류 ({ticker}): {e}\n{traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"message": f"서버 오류 발생: {e}"})
# ... (main 실행 부분 동일)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)