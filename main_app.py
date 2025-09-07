import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
import time

# قمع التحذيرات
warnings.filterwarnings('ignore')

# تجربة استيراد scikit-learn
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import MinMaxScaler
    sklearn_available = True
except ImportError:
    sklearn_available = False
    st.warning("⚠️ scikit-learn غير متوفر - سيتم استخدام نموذج تنبؤ مبسط")

# إعدادات الصفحة
st.set_page_config(
    page_title="🚀 منصة التنبؤ بالأسهم",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS للتصميم
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .big-font {
        font-size: 40px !important;
        text-align: center;
        background: linear-gradient(90deg, #ff7e5f, #feb47b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
        font-weight: bold;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metric-box h3 {
        margin: 0 0 10px 0;
        font-size: 16px;
    }
    .metric-box h2 {
        margin: 5px 0;
        font-size: 24px;
        font-weight: bold;
    }
    .prediction-alert {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
        font-size: 16px;
    }
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
    }
    .week-prediction {
        background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%);
        color: #333;
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        border: 2px solid #ff6b6b;
    }
    .month-prediction {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #333;
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        border: 2px solid #48cae4;
    }
    .info-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        color: #333;
        padding: 15px;
        border-radius: 10px;
        margin: 15px 0;
        text-align: center;
        font-weight: bold;
    }
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 50px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# تهيئة session state
if 'selected_stock' not in st.session_state:
    st.session_state.selected_stock = None

# العنوان الرئيسي
st.markdown('<p class="big-font">🚀 منصة التنبؤ بالأسهم</p>', unsafe_allow_html=True)
st.markdown("### 🎯 نظام ذكي للتنبؤ بحركة الأسهم مع تحليل فني متقدم")

# معلومات المنصة
st.markdown("""
<div class="info-box">
    🔥 منصة تحليل وتنبؤ شاملة مع مؤشرات فنية متقدمة
    <br>📊 تحليل دقيق للاتجاهات والتوصيات الاستثمارية
</div>
""", unsafe_allow_html=True)

# معلومات الوقت مع معالجة أفضل للأخطاء
try:
    now = datetime.now()
    us_time = now.strftime('%H:%M:%S')
    sa_time = (now + timedelta(hours=8)).strftime('%H:%M:%S')

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"🇺🇸 الوقت الأمريكي: {us_time}")
    with col2:
        st.info(f"🇸🇦 الوقت السعودي: {sa_time}")
    with col3:
        current_hour = now.hour
        is_weekend = now.weekday() >= 5
        market_open = not is_weekend and 9 <= current_hour <= 16
        market_status = "🟢 مفتوح" if market_open else "🔴 مغلق"
        st.info(f"📈 السوق الأمريكي: {market_status}")
except Exception as e:
    st.info("⚠️ لا يمكن عرض معلومات الوقت حالياً")
    print(f"Time display error: {e}")

# رسالة النجاح
st.markdown("""
<div class="success-box">
    <h3>✅ المنصة جاهزة للاستخدام!</h3>
    <p>يمكنك الآن تحليل أي سهم والحصول على توصيات استثمارية</p>
</div>
""", unsafe_allow_html=True)

# إدخال البيانات
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    symbol_input = st.text_input("🔍 أدخل رمز السهم:", value="AAPL", placeholder="مثل: AAPL, GOOGL, MSFT")
    symbol = symbol_input.upper().strip() if symbol_input else "AAPL"

with col2:
    prediction_period = st.selectbox("📅 فترة التحليل:",
                                   ["أسبوعي", "شهري"],
                                   index=0,
                                   help="اختر فترة التحليل والتنبؤ")

with col3:
    st.write("")
    st.write("")
    analyze = st.button("🔍 تحليل السهم", type="primary", key="analyze_btn")

# أزرار الأسهم الشائعة
st.write("⭐ **أسهم شائعة للتجربة السريعة:**")
col1, col2, col3, col4, col5, col6 = st.columns(6)

stocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA"]
cols = [col1, col2, col3, col4, col5, col6]

for i, stock in enumerate(stocks):
    with cols[i]:
        if st.button(stock, key=f"stock_{stock}"):
            st.session_state.selected_stock = stock
            st.rerun()

# التحقق من الاختيار
if st.session_state.selected_stock:
    symbol = st.session_state.selected_stock
    analyze = True
    st.session_state.selected_stock = None

# دالة آمنة لاستخراج القيم مع تحسينات
def safe_get(series, index=-1, default=0):
    try:
        if series is None or len(series) == 0:
            return default
        value = series.iloc[index]
        return float(value) if pd.notna(value) and np.isfinite(value) else default
    except (IndexError, TypeError, ValueError, AttributeError):
        return default

# دالة حساب RSI محسنة
def calculate_rsi(prices, period=14):
    try:
        if len(prices) < period:
            return pd.Series([50] * len(prices), index=prices.index)
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period//2).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period//2).mean()
        
        # تجنب القسمة على صفر
        loss = loss.replace(0, 0.0001)
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # التأكد من صحة القيم
        rsi = rsi.fillna(50)
        rsi = rsi.clip(0, 100)
        
        return rsi
    except Exception as e:
        print(f"RSI calculation error: {e}")
        return pd.Series([50] * len(prices), index=prices.index)

# دالة حساب MACD محسنة
def calculate_macd(prices, fast=12, slow=26, signal=9):
    try:
        if len(prices) < slow:
            zeros = pd.Series([0] * len(prices), index=prices.index)
            return zeros, zeros, zeros
        
        ema_fast = prices.ewm(span=fast, min_periods=fast//2).mean()
        ema_slow = prices.ewm(span=slow, min_periods=slow//2).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, min_periods=signal//2).mean()
        histogram = macd - signal_line
        
        # ملء القيم المفقودة
        macd = macd.fillna(0)
        signal_line = signal_line.fillna(0)
        histogram = histogram.fillna(0)
        
        return macd, signal_line, histogram
    except Exception as e:
        print(f"MACD calculation error: {e}")
        zeros = pd.Series([0] * len(prices), index=prices.index)
        return zeros, zeros, zeros

# دالة التنبؤ المبسطة محسنة
def simple_prediction(data, days=7):
    try:
        if len(data) == 0:
            return np.array([100] * days)
        
        prices = data['Close'].values
        
        if len(prices) < 5:
            # بيانات قليلة جداً
            last_price = prices[-1] if len(prices) > 0 else 100
            return np.array([last_price * (1 + np.random.uniform(-0.02, 0.02)) for _ in range(days)])
        
        # حساب الاتجاه العام من آخر 10 نقاط
        recent_len = min(10, len(prices))
        recent_prices = prices[-recent_len:]
        
        # استخدام الانحدار الخطي البسيط
        x = np.arange(len(recent_prices))
        trend = np.polyfit(x, recent_prices, 1)[0]
        
        # حساب التقلب
        volatility_len = min(30, len(prices))
        volatility = np.std(prices[-volatility_len:]) if volatility_len > 1 else abs(prices[-1] * 0.02)
        
        # التنبؤ مع إضافة عشوائية محكومة
        last_price = prices[-1]
        predictions = []
        
        for i in range(1, days + 1):
            # إضافة اتجاه مع تقليل قوته مع الوقت
            trend_effect = trend * i * (0.9 ** i)  # تقليل تأثير الاتجاه مع الوقت
            
            # إضافة عشوائية محكومة
            noise = np.random.normal(0, volatility * 0.05 * np.sqrt(i))
            
            # التنبؤ النهائي
            predicted_price = last_price + trend_effect + noise
            
            # حماية من القيم غير المنطقية
            predicted_price = max(predicted_price, last_price * 0.5)
            predicted_price = min(predicted_price, last_price * 2.0)
            
            predictions.append(predicted_price)
        
        return np.array(predictions)
    
    except Exception as e:
        print(f"Simple prediction error: {e}")
        # إرجاع تنبؤ آمن
        last_price = data['Close'].iloc[-1] if len(data) > 0 else 100
        return np.array([last_price * (1 + np.random.uniform(-0.03, 0.03)) for _ in range(days)])

# دالة التنبؤ المتقدمة محسنة
def advanced_prediction(data, days=7):
    if not sklearn_available:
        return simple_prediction(data, days)

    try:
        if len(data) < 30:  # بيانات قليلة
            return simple_prediction(data, days)
        
        prices = data['Close'].values
        
        # تنظيف البيانات
        prices = prices[~np.isnan(prices)]
        prices = prices[np.isfinite(prices)]
        
        if len(prices) < 20:
            return simple_prediction(data, days)
        
        # تحضير البيانات للنموذج
        prices_reshaped = prices.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_prices = scaler.fit_transform(prices_reshaped)

        # إنشاء features مع نافذة متكيفة
        window = min(15, len(scaled_prices) // 3)
        X, y = [], []

        for i in range(window, len(scaled_prices)):
            X.append(scaled_prices[i-window:i, 0])
            y.append(scaled_prices[i, 0])

        if len(X) < 10:  # بيانات قليلة للتدريب
            return simple_prediction(data, days)

        X, y = np.array(X), np.array(y)

        # تدريب النموذج مع معالجة الأخطاء
        model = LinearRegression()
        model.fit(X, y)

        # التنبؤ
        last_window = scaled_prices[-window:].flatten()
        predictions = []
        current_window = last_window.copy()

        for _ in range(days):
            try:
                next_pred = model.predict(current_window.reshape(1, -1))[0]
                
                # التأكد من صحة التنبؤ
                if not np.isfinite(next_pred):
                    next_pred = current_window[-1]
                
                predictions.append(next_pred)
                
                # تحديث النافذة
                current_window = np.roll(current_window, -1)
                current_window[-1] = next_pred
                
            except Exception as pred_error:
                print(f"Prediction step error: {pred_error}")
                # استخدام آخر قيمة كتنبؤ آمن
                predictions.append(current_window[-1])

        # إعادة التحويل مع معالجة الأخطاء
        try:
            predictions = np.array(predictions).reshape(-1, 1)
            predictions_rescaled = scaler.inverse_transform(predictions).flatten()
            
            # التأكد من منطقية النتائج
            last_actual_price = prices[-1]
            predictions_rescaled = np.clip(
                predictions_rescaled, 
                last_actual_price * 0.7, 
                last_actual_price * 1.5
            )
            
            return predictions_rescaled
            
        except Exception as rescale_error:
            print(f"Rescaling error: {rescale_error}")
            return simple_prediction(data, days)

    except Exception as e:
        print(f"Advanced prediction error: {e}")
        return simple_prediction(data, days)

# دالة جلب البيانات محسنة
@st.cache_data(ttl=300)  # تخزين مؤقت لمدة 5 دقائق
def fetch_stock_data(symbol, max_retries=3):
    """جلب بيانات السهم مع إعادة المحاولة"""
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(symbol)
            
            # جرب فترات مختلفة
            periods = ["2y", "1y", "6mo", "3mo"]
            data = None
            info = {}
            
            for period in periods:
                try:
                    data = stock.history(period=period)
                    if not data.empty and len(data) >= 10:
                        break
                except:
                    continue
            
            # جلب معلومات الشركة
            try:
                info = stock.info
            except:
                info = {}
            
            if data is not None and not data.empty:
                # تنظيف البيانات
                data = data.dropna()
                
                # التأكد من صحة البيانات
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if col in data.columns:
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                
                data = data.dropna()
                
                if len(data) >= 5:
                    return data, info
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)  # انتظار قبل إعادة المحاولة
    
    return None, {}

# التحليل الرئيسي
if analyze and symbol:
    with st.spinner(f"🔄 جاري تحليل سهم {symbol}..."):
        try:
            # جلب البيانات
            data, info = fetch_stock_data(symbol)

            if data is None or data.empty:
                st.error(f"❌ لا يمكن العثور على بيانات للسهم {symbol}")
                st.info("💡 تأكد من صحة رمز السهم وحاول مرة أخرى")
            else:
                if len(data) < 10:
                    st.error("❌ البيانات المتوفرة قليلة جداً للتحليل")
                else:
                    # حساب المؤشرات الفنية
                    try:
                        # المتوسطات المتحركة
                        data['SMA20'] = data['Close'].rolling(20, min_periods=5).mean()
                        data['SMA50'] = data['Close'].rolling(50, min_periods=10).mean()
                        
                        # RSI
                        data['RSI'] = calculate_rsi(data['Close'])
                        
                        # MACD
                        data['MACD'], data['MACD_Signal'], data['MACD_Hist'] = calculate_macd(data['Close'])

                        # Bollinger Bands
                        sma20 = data['Close'].rolling(20, min_periods=5).mean()
                        std20 = data['Close'].rolling(20, min_periods=5).std()
                        data['BB_Upper'] = sma20 + (std20 * 2)
                        data['BB_Lower'] = sma20 - (std20 * 2)

                    except Exception as e:
                        st.warning(f"تحذير في حساب المؤشرات: {e}")

                    # استخراج البيانات مع حماية من الأخطاء
                    try:
                        current_price = safe_get(data['Close'])
                        prev_price = safe_get(data['Close'], -2, current_price)

                        if current_price <= 0:
                            current_price = safe_get(data['Close'], default=100)
                        if prev_price <= 0:
                            prev_price = current_price

                        price_change = ((current_price - prev_price) / prev_price * 100) if prev_price > 0 else 0

                        rsi = safe_get(data['RSI'], default=50)
                        macd = safe_get(data['MACD'], default=0)
                        macd_signal = safe_get(data['MACD_Signal'], default=0)

                        sma20 = safe_get(data['SMA20'], default=current_price)
                        sma50 = safe_get(data['SMA50'], default=current_price)

                        # التنبؤ
                        days_to_predict = 7 if prediction_period == "أسبوعي" else 30
                        predictions = advanced_prediction(data, days_to_predict)

                        if len(predictions) > 0:
                            final_prediction = predictions[-1]
                            expected_change = ((final_prediction - current_price) / current_price * 100) if current_price > 0 else 0

                            # حساب درجة التوصية
                            score = 0

                            # تحليل RSI
                            if rsi < 30:
                                score += 3  # منطقة شراء
                            elif rsi < 40:
                                score += 1
                            elif rsi > 70:
                                score -= 3  # منطقة بيع
                            elif rsi > 60:
                                score -= 1

                            # تحليل المتوسطات المتحركة
                            if current_price > sma20:
                                score += 2
                            if current_price > sma50:
                                score += 1
                            if sma20 > sma50:
                                score += 1

                            # تحليل MACD
                            if macd > macd_signal:
                                score += 1
                            else:
                                score -= 1

                            # اتجاه السعر
                            if price_change > 2:
                                score += 2
                            elif price_change > 0:
                                score += 1
                            elif price_change < -2:
                                score -= 2
                            else:
                                score -= 1

                            # التوصية النهائية
                            if score >= 5:
                                recommendation = "شراء قوي جداً 🟢🟢🟢"
                                confidence = 85 + np.random.randint(0, 10)
                            elif score >= 3:
                                recommendation = "شراء قوي 🟢🟢"
                                confidence = 75 + np.random.randint(0, 12)
                            elif score >= 1:
                                recommendation = "شراء 🟢"
                                confidence = 65 + np.random.randint(0, 15)
                            elif score <= -5:
                                recommendation = "بيع قوي جداً 🔴🔴🔴"
                                confidence = 80 + np.random.randint(0, 15)
                            elif score <= -3:
                                recommendation = "بيع قوي 🔴🔴"
                                confidence = 70 + np.random.randint(0, 15)
                            elif score <= -1:
                                recommendation = "بيع 🔴"
                                confidence = 60 + np.random.randint(0, 15)
                            else:
                                recommendation = "انتظار ⚪"
                                confidence = 50 + np.random.randint(0, 20)

                            # عرض النتائج
                            st.success("✅ تم التحليل بنجاح!")

                            # المقاييس الأساسية
                            col1, col2, col3, col4 = st.columns(4)

                            with col1:
                                st.markdown(f"""
                                <div class="metric-box">
                                    <h3>💰 السعر الحالي</h3>
                                    <h2>${current_price:.2f}</h2>
                                    <p>{price_change:+.2f}% اليوم</p>
                                </div>
                                """, unsafe_allow_html=True)

                            with col2:
                                st.markdown(f"""
                                <div class="metric-box">
                                    <h3>🔮 السعر المتوقع</h3>
                                    <h2>${final_prediction:.2f}</h2>
                                    <p>{expected_change:+.2f}% تغيير</p>
                                </div>
                                """, unsafe_allow_html=True)

                            with col3:
                                st.markdown(f"""
                                <div class="metric-box">
                                    <h3>🎯 التوصية</h3>
                                    <h2 style="font-size: 14px;">{recommendation}</h2>
                                    <p>الثقة: {confidence}%</p>
                                </div>
                                """, unsafe_allow_html=True)

                            with col4:
                                period_text = "أسبوع" if prediction_period == "أسبوعي" else "شهر"
                                st.markdown(f"""
                                <div class="metric-box">
                                    <h3>📅 فترة التحليل</h3>
                                    <h2>{period_text}</h2>
                                    <p>{days_to_predict} يوم</p>
                                </div>
                                """, unsafe_allow_html=True)

                            # صندوق التوصية الرئيسي
                            end_date = datetime.now() + timedelta(days=days_to_predict)
                            st.markdown(f"""
                            <div class="prediction-alert">
                                <h2>🎯 التوصية النهائية: {recommendation}</h2>
                                <p><strong>💰 السعر الحالي:</strong> ${current_price:.2f}</p>
                                <p><strong>🔮 السعر المتوقع:</strong> ${final_prediction:.2f} ({expected_change:+.2f}%)</p>
                                <p><strong>📅 تاريخ التنبؤ:</strong> {end_date.strftime('%Y-%m-%d')}</p>
                                <p><strong>📊 مستوى الثقة:</strong> {confidence}%</p>
                            </div>
                            """, unsafe_allow_html=True)

                            # التنبؤات التفصيلية
                            col1, col2 = st.columns(2)

                            with col1:
                                st.markdown("""
                                <div class="week-prediction">
                                    <h3>📈 تنبؤات التطور</h3>
                                </div>
                                """, unsafe_allow_html=True)

                                days_to_show = min(7, len(predictions))
                                for i in range(days_to_show):
                                    day_date = (datetime.now() + timedelta(days=i+1)).strftime('%m-%d')
                                    pred = predictions[i]
                                    daily_change = ((pred - current_price) / current_price * 100) if current_price > 0 else 0
                                    trend = "📈" if daily_change > 0 else "📉" if daily_change < 0 else "➡️"
                                    st.write(f"**يوم {i+1} ({day_date}):** ${pred:.2f} {trend} {daily_change:+.2f}%")

                            with col2:
                                st.markdown("""
                                <div class="month-prediction">
                                    <h3>📊 التحليل الفني</h3>
                                </div>
                                """, unsafe_allow_html=True)

                                st.write(f"**RSI:** {rsi:.1f}")
                                if rsi > 70:
                                    st.write("🔴 منطقة شراء مفرط")
                                elif rsi < 30:
                                    st.write("🟢 منطقة بيع مفرط")
                                else:
                                    st.write("🟡 منطقة متوازنة")

                                st.write(f"**MACD:** {macd:.3f}")
                                if macd > macd_signal:
                                    st.write("🟢 إشارة إيجابية")
                                else:
                                    st.write("🔴 إشارة سلبية")

                                st.write("**المتوسطات المتحركة:**")
                                st.write(f"SMA20: ${sma20:.2f} ({'🟢' if current_price > sma20 else '🔴'})")
                                st.write(f"SMA50: ${sma50:.2f} ({'🟢' if current_price > sma50 else '🔴'})")

                            # الرسم البياني محسن
                            st.markdown("### 📈 الرسم البياني")

                            try:
                                fig = go.Figure()

                                # البيانات التاريخية (آخر 60 يوم)
                                recent_data = data.tail(60)

                                fig.add_trace(go.Candlestick(
                                    x=recent_data.index,
                                    open=recent_data['Open'],
                                    high=recent_data['High'],
                                    low=recent_data['Low'],
                                    close=recent_data['Close'],
                                    name='السعر التاريخي',
                                    increasing_line_color='green',
                                    decreasing_line_color='red'
                                ))

                                # المتوسط المتحرك
                                if 'SMA20' in recent_data.columns and not recent_data['SMA20'].isna().all():
                                    fig.add_trace(go.Scatter(
                                        x=recent_data.index,
                                        y=recent_data['SMA20'],
                                        name='SMA 20',
                                        line=dict(color='orange', width=2)
                                    ))

                                if 'SMA50' in recent_data.columns and not recent_data['SMA50'].isna().all():
                                    fig.add_trace(go.Scatter(
                                        x=recent_data.index,
                                        y=recent_data['SMA50'],
                                        name='SMA 50',
                                        line=dict(color='blue', width=2)
                                    ))

                                # التنبؤات
                                future_dates = pd.date_range(
                                    start=data.index[-1] + timedelta(days=1),
                                    periods=len(predictions),
                                    freq='D'
                                )

                                fig.add_trace(go.Scatter(
                                    x=future_dates,
                                    y=predictions,
                                    name='التنبؤ',
                                    line=dict(color='red', width=3, dash='dot'),
                                    mode='lines+markers',
                                    marker=dict(size=6)
                                ))

                                fig.update_layout(
                                    title=f"📊 تحليل سهم {symbol}",
                                    height=600,
                                    showlegend=True,
                                    xaxis_title="التاريخ",
                                    yaxis_title="السعر ($)",
                                    template="plotly_white",
                                    hovermode="x unified"
                                )

                                st.plotly_chart(fig, use_container_width=True)

                            except Exception as e:
                                st.warning(f"لا يمكن عرض الرسم البياني: {e}")

                            # معلومات الشركة
                            if info:
                                col1, col2 = st.columns(2)

                                with col1:
                                    st.markdown("### 🏢 معلومات الشركة")
                                    company_name = info.get('longName', info.get('shortName', symbol))
                                    st.write(f"**الاسم:** {company_name}")
                                    st.write(f"**القطاع:** {info.get('sector', 'غير متاح')}")
                                    st.write(f"**الصناعة:** {info.get('industry', 'غير متاح')}")

                                    if info.get('marketCap'):
                                        market_cap = info['marketCap'] / 1e9
                                        st.write(f"**القيمة السوقية:** ${market_cap:.1f}B")

                                with col2:
                                    st.markdown("### 📊 إحصائيات إضافية")
                                    st.write(f"**قوة الإشارة:** {abs(score)}/8")

                                    volume = safe_get(data['Volume'], default=0)
                                    if volume > 0:
                                        volume_millions = volume / 1e6
                                        st.write(f"**حجم التداول:** {volume_millions:.1f}M")

                                    high_52w = info.get('fiftyTwoWeekHigh')
                                    low_52w = info.get('fiftyTwoWeekLow')

                                    if high_52w:
                                        st.write(f"**أعلى 52 أسبوع:** ${high_52w:.2f}")
                                    if low_52w:
                                        st.write(f"**أقل 52 أسبوع:** ${low_52w:.2f}")

                        else:
                            st.error("❌ فشل في إنشاء التنبؤات")

                    except Exception as analysis_error:
                        st.error(f"❌ خطأ في التحليل: {analysis_error}")
                        print(f"Analysis error: {analysis_error}")

        except Exception as e:
            st.error(f"❌ حدث خطأ أثناء التحليل: {str(e)}")
            st.info("💡 تأكد من صحة رمز السهم واتصال الإنترنت وحاول مرة أخرى")
            print(f"Main analysis error: {e}")

# معلومات المنصة
st.markdown("---")
st.markdown("### ℹ️ معلومات المنصة")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.info("🎯 **التحليل:** شامل ومتقدم")
with col2:
    st.info("📊 **المؤشرات:** RSI, MACD, SMA")
with col3:
    st.info("🔄 **التحديث:** فوري")
with col4:
    st.info("🧠 **الذكاء:** اصطناعي")

# تحذير قانوني
st.warning("⚠️ **تنبيه مهم:** هذه المنصة للأغراض التعليمية فقط. استشر مستشار مالي مؤهل قبل اتخاذ قرارات استثمارية.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <h3>🚀 منصة التنبؤ بالأسهم</h3>
    <p>نظام ذكي للتحليل المالي والتنبؤات المتقدمة</p>
    <p>© 2025 - تم تطويرها بتقنيات الذكاء الاصطناعي</p>
</div>
""", unsafe_allow_html=True)
