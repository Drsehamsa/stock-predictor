import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# محاولة استيراد scikit-learn
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import MinMaxScaler
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# إعدادات الصفحة
st.set_page_config(
    page_title="🚀 منصة التنبؤ بالأسهم",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===== CSS =====
st.markdown("""
<style>
.big-font { font-size: 40px !important; text-align: center;
    background: linear-gradient(90deg, #ff7e5f, #feb47b);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 20px; font-weight: bold; }
.metric-box { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white; padding: 20px; border-radius: 15px; text-align: center; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# ===== العنوان =====
st.markdown('<p class="big-font"> 🚀 منصة التنبؤ بالأسهم </p>', unsafe_allow_html=True)

# ===== إدخال المستخدم =====
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    symbol = st.text_input("أدخل رمز السهم:", value="AAPL").strip().upper()
with col2:
    period = st.selectbox("الفترة:", ["أسبوعي", "شهري"], index=0)
with col3:
    analyze = st.button("تحليل السهم")

# ===== دوال مساعدة =====
def safe_get(series, default=0.0):
    return float(series.iloc[-1]) if len(series) > 0 else default

def simple_prediction(data, days=7):
    prices = data['Close'].values
    if len(prices) == 0:
        return np.array([100]*days)
    last_price = prices[-1]
    return np.linspace(last_price, last_price*1.05, days)

def advanced_prediction(data, days=7):
    if not SKLEARN_OK: return simple_prediction(data, days)
    prices = data['Close'].values.reshape(-1,1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)
    X, y = [], []
    window = 10
    for i in range(window, len(scaled)):
        X.append(scaled[i-window:i,0]); y.append(scaled[i,0])
    if len(X) < 10: return simple_prediction(data, days)
    model = LinearRegression().fit(np.array(X), np.array(y))
    last = scaled[-window:,0]
    preds = []
    for _ in range(days):
        nxt = model.predict(last.reshape(1,-1))[0]
        preds.append(nxt)
        last = np.roll(last, -1); last[-1] = nxt
    return scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()

# ===== التحليل =====
if analyze:
    with st.spinner(f"جاري تحميل {symbol}..."):
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period="1y")
            if data.empty:
                st.error("❌ لا توجد بيانات للسهم")
            else:
                current = safe_get(data['Close'])
                days = 7 if period == "أسبوعي" else 30
                preds = advanced_prediction(data, days)
                future = preds[-1]

                st.success("✅ تم التحليل")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"<div class='metric-box'><h3>السعر الحالي</h3><h2>${current:.2f}</h2></div>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"<div class='metric-box'><h3>السعر المتوقع</h3><h2>${future:.2f}</h2></div>", unsafe_allow_html=True)

                # رسم بياني
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="السعر الفعلي"))
                future_dates = pd.date_range(start=data.index[-1]+timedelta(days=1), periods=days)
                fig.add_trace(go.Scatter(x=future_dates, y=preds, name="التنبؤ"))
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"خطأ: {e}")

st.warning("⚠️ هذا النظام للتجربة فقط، لا يعتبر نصيحة استثمارية.")
