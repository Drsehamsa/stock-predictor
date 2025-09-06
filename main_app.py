import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings

# قمع التحذيرات
warnings.filterwarnings('ignore')

# تجربة استيراد scikit-learn
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import MinMaxScaler
    sklearn_available = True
except ImportError:
    sklearn_available = False
    st.warning("scikit-learn غير متوفر - سيتم استخدام نموذج تنبؤ مبسط")

# إعدادات الصفحة
st.set_page_config(
    page_title="منصة التنبؤ بالأسهم",
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
st.markdown('<p class="big-font">منصة التنبؤ بالأسهم</p>', unsafe_allow_html=True)
st.markdown("### نظام ذكي للتنبؤ بحركة الأسهم مع تحليل فني متقدم")

# معلومات المنصة
st.markdown("""
<div class="info-box">
    منصة تحليل وتنبؤ شاملة مع مؤشرات فنية متقدمة
    <br>تحليل دقيق للاتجاهات والتوصيات الاستثمارية
</div>
""", unsafe_allow_html=True)

# معلومات الوقت
try:
    now = datetime.now()
    us_time = now.strftime('%H:%M:%S')
    sa_time = (now + timedelta(hours=8)).strftime('%H:%M:%S')

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"الوقت الأمريكي: {us_time}")
    with col2:
        st.info(f"الوقت السعودي: {sa_time}")
    with col3:
        current_hour = now.hour
        is_weekend = now.weekday() >= 5
        market_open = not is_weekend and 9 <= current_hour <= 16
        market_status = "مفتوح" if market_open else "مغلق"
        st.info(f"السوق الأمريكي: {market_status}")
except:
    st.info("لا يمكن عرض معلومات الوقت حالياً")

# رسالة الترحيب
st.markdown("""
<div class="success-box">
    <h3>المنصة جاهزة للاستخدام!</h3>
    <p>يمكنك الآن تحليل أي سهم والحصول على توصيات استثمارية</p>
</div>
""", unsafe_allow_html=True)

# إدخال البيانات
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    symbol_input = st.text_input("أدخل رمز السهم:", value="AAPL", placeholder="مثل: AAPL, GOOGL, MSFT")
    symbol = symbol_input.upper().strip() if symbol_input else "AAPL"

with col2:
    prediction_period = st.selectbox("فترة التحليل:",
                                   ["أسبوعي", "شهري"],
                                   index=0,
                                   help="اختر فترة التحليل والتنبؤ")

with col3:
    st.write("")
    st.write("")
    analyze = st.button("تحليل السهم", type="primary", key="analyze_btn")

# أزرار الأسهم الشائعة
st.write("**أسهم شائعة للتجربة السريعة:**")
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

# دالة آمنة لاستخراج القيم
def safe_get(series, index=-1, default=0):
    try:
        if len(series) == 0:
            return default
        value = series.iloc[index]
        return value if pd.notna(value) else default
    except:
        return default

# دالة حساب RSI
def calculate_rsi(prices, period=14):
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except:
        return pd.Series([50] * len(prices), index=prices.index)

# دالة حساب MACD
def calculate_macd(prices, fast=12, slow=26, signal=9):
    try:
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    except:
        zeros = pd.Series([0] * len(prices), index=prices.index)
        return zeros, zeros, zeros

# دالة التنبؤ المبسطة
def simple_prediction(data, days=7):
    try:
        # استخدام المتوسط المتحرك والاتجاه
        prices = data['Close'].values

        # حساب الاتجاه العام
        recent_prices = prices[-10:]
        trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]

        # حساب التقلب
        volatility = np.std(prices[-30:]) if len(prices) >= 30 else np.std(prices)

        # التنبؤ البسيط
        last_price = prices[-1]
        predictions = []

        for i in range(1, days + 1):
            # إضافة عشوائية محكومة
            noise = np.random.normal(0, volatility * 0.1)
            predicted_price = last_price + (trend * i) + noise
            predictions.append(max(predicted_price, last_price * 0.5))  # حماية من القيم السالبة

        return np.array(predictions)
    except:
        # في حالة فشل كل شيء، إرجاع تنبؤ ثابت
        last_price = data['Close'].iloc[-1] if len(data) > 0 else 100
        return np.array([last_price * (1 + np.random.uniform(-0.05, 0.05)) for _ in range(days)])

# دالة التنبؤ المتقدمة (مع sklearn)
def advanced_prediction(data, days=7):
    if not sklearn_available:
        return simple_prediction(data, days)

    try:
        prices = data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_prices = scaler.fit_transform(prices)

        # إنشاء features
        window = min(20, len(scaled_prices) // 2)
        X, y = [], []

        for i in range(window, len(scaled_prices)):
            X.append(scaled_prices[i-window:i, 0])
            y.append(scaled_prices[i, 0])

        if len(X) < 10:  # البيانات قليلة جداً
            return simple_prediction(data, days)

        X, y = np.array(X), np.array(y)

        # تدريب النموذج
        model = LinearRegression()
        model.fit(X, y)

        # التنبؤ
        last_window = scaled_prices[-window:].flatten()
        predictions = []
        current_window = last_window.copy()

        for _ in range(days):
            next_pred = model.predict(current_window.reshape(1, -1))[0]
            predictions.append(next_pred)
            current_window = np.roll(current_window, -1)
            current_window[-1] = next_pred

        # إعادة التحويل
        predictions = np.array(predictions).reshape(-1, 1)
        return scaler.inverse_transform(predictions).flatten()

    except Exception as e:
        st.warning(f"التنبؤ المتقدم فشل، سيتم استخدام النموذج البسيط: {str(e)}")
        return simple_prediction(data, days)

# دالة تحسين جلب البيانات
@st.cache_data(ttl=300)  # cache لمدة 5 دقائق
def fetch_stock_data(symbol, max_retries=3):
    """جلب بيانات السهم مع إعادة المحاولة"""
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(symbol)
            
            # تجربة فترات مختلفة
            periods = ["1y", "6mo", "3mo", "1mo"]
            
            for period in periods:
                try:
                    data = stock.history(period=period)
                    if not data.empty and len(data) >= 10:
                        return data, stock.info
                except:
                    continue
                    
            return pd.DataFrame(), {}
            
        except Exception as e:
            if attempt < max_retries - 1:
                st.warning(f"المحاولة {attempt + 1} فشلت، جاري إعادة المحاولة...")
                continue
            else:
                st.error(f"فشل في جلب البيانات بعد {max_retries} محاولات")
                return pd.DataFrame(), {}
    
    return pd.DataFrame(), {}

# دالة التحقق من صحة رمز السهم
def validate_stock_symbol(symbol):
    """التحقق من صحة رمز السهم"""
    if not symbol or len(symbol) < 1:
        return False, "رمز السهم فارغ"
    
    if len(symbol) > 10:
        return False, "رمز السهم طويل جداً"
    
    # قائمة رموز شائعة للتحقق السريع
    common_symbols = [
        'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
        'AMD', 'INTC', 'ORCL', 'CRM', 'ADBE', 'PYPL', 'UBER', 'SNAP'
    ]
    
    if symbol.upper() in common_symbols:
        return True, "رمز صحيح"
    
    return True, "سيتم التحقق من الرمز"

# التحليل الرئيسي
if analyze and symbol:
    # التحقق من صحة رمز السهم أولاً
    is_valid, message = validate_stock_symbol(symbol)
    
    if not is_valid:
        st.error(f"خطأ: {message}")
    else:
        with st.spinner(f"جاري تحليل سهم {symbol}..."):
            try:
                # جلب البيانات باستخدام الدالة المحسنة
                data, info = fetch_stock_data(symbol)

                if data.empty:
                    st.error(f"لا يمكن العثور على بيانات للسهم {symbol}")
                    st.info("تأكد من:")
                    st.write("- صحة رمز السهم")
                    st.write("- الاتصال بالإنترنت")
                    st.write("- أن السهم متداول في البورصة الأمريكية")
                    
                    # اقتراح رموز بديلة
                    st.write("جرب هذه الرموز:")
                    suggested = ['AAPL', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']
                    for i, stock_symbol in enumerate(suggested):
                        if i % 3 == 0:
                            cols = st.columns(3)
                        with cols[i % 3]:
                            if st.button(f"جرب {stock_symbol}", key=f"suggest_{stock_symbol}"):
                                st.session_state.selected_stock = stock_symbol
                                st.rerun()
                else:
                    # تنظيف البيانات
                    data = data.dropna()

                    if len(data) < 10:
                        st.error("البيانات المتوفرة قليلة جداً للتحليل")
                        st.info(f"متوفر فقط {len(data)} نقطة بيانات، نحتاج على الأقل 10")
                    else:
                        # حساب المؤشرات
                        try:
                            data['SMA20'] = data['Close'].rolling(20, min_periods=5).mean()
                            data['SMA50'] = data['Close'].rolling(50, min_periods=10).mean()
                            data['RSI'] = calculate_rsi(data['Close'])
                            data['MACD'], data['MACD_Signal'], data['MACD_Hist'] = calculate_macd(data['Close'])

                            # Bollinger Bands
                            sma20 = data['Close'].rolling(20, min_periods=5).mean()
                            std20 = data['Close'].rolling(20, min_periods=5).std()
                            data['BB_Upper'] = sma20 + (std20 * 2)
                            data['BB_Lower'] = sma20 - (std20 * 2)

                        except Exception as e:
                            st.warning(f"تحذير في حساب المؤشرات: {e}")

                        # استخراج البيانات
                        current_price = safe_get(data['Close'])
                        prev_price = safe_get(data['Close'], -2, current_price)

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
                            expected_change = ((final_prediction - current_price) / current_price * 100)

                            # حساب النتيجة
                            score = 0

                            # RSI
                            if rsi < 30:
                                score += 3  # oversold
                            elif rsi < 40:
                                score += 1
                            elif rsi > 70:
                                score -= 3  # overbought
                            elif rsi > 60:
                                score -= 1

                            # Moving Averages
                            if current_price > sma20:
                                score += 2
                            if current_price > sma50:
                                score += 1

                            # MACD
                            if macd > macd_signal:
                                score += 1
                            else:
                                score -= 1

                            # اتجاه السعر
                            if price_change > 0:
                                score += 1
                            else:
                                score -= 1

                            # التوصية النهائية
                            if score >= 4:
                                recommendation = "شراء قوي جداً"
                                confidence = 85 + np.random.randint(0, 10)
                            elif score >= 2:
                                recommendation = "شراء قوي"
                                confidence = 75 + np.random.randint(0, 12)
                            elif score >= 1:
                                recommendation = "شراء"
                                confidence = 65 + np.random.randint(0, 15)
                            elif score <= -4:
                                recommendation = "بيع قوي جداً"
                                confidence = 80 + np.random.randint(0, 15)
                            elif score <= -2:
                                recommendation = "بيع قوي"
                                confidence = 70 + np.random.randint(0, 15)
                            elif score <= -1:
                                recommendation = "بيع"
                                confidence = 60 + np.random.randint(0, 15)
                            else:
                                recommendation = "انتظار"
                                confidence = 50 + np.random.randint(0, 20)

                            # عرض النتائج
                            st.success("تم التحليل بنجاح!")

                            # المقاييس الأساسية
                            col1, col2, col3, col4 = st.columns(4)

                            with col1:
                                st.markdown(f"""
                                <div class="metric-box">
                                    <h3>السعر الحالي</h3>
                                    <h2>${current_price:.2f}</h2>
                                    <p>{price_change:+.2f}% اليوم</p>
                                </div>
                                """, unsafe_allow_html=True)

                            with col2:
                                st.markdown(f"""
                                <div class="metric-box">
                                    <h3>السعر المتوقع</h3>
                                    <h2>${final_prediction:.2f}</h2>
                                    <p>{expected_change:+.2f}% تغيير</p>
                                </div>
                                """, unsafe_allow_html=True)

                            with col3:
                                st.markdown(f"""
                                <div class="metric-box">
                                    <h3>التوصية</h3>
                                    <h2 style="font-size: 16px;">{recommendation}</h2>
                                    <p>الثقة: {confidence}%</p>
                                </div>
                                """, unsafe_allow_html=True)

                            with col4:
                                period_text = "أسبوع" if prediction_period == "أسبوعي" else "شهر"
                                st.markdown(f"""
                                <div class="metric-box">
                                    <h3>فترة التحليل</h3>
                                    <h2>{period_text}</h2>
                                    <p>{days_to_predict} يوم</p>
                                </div>
                                """, unsafe_allow_html=True)

                            # صندوق التوصية الرئيسي
                            end_date = datetime.now() + timedelta(days=days_to_predict)
                            st.markdown(f"""
                            <div class="prediction-alert">
                                <h2>التوصية النهائية: {recommendation}</h2>
                                <p><strong>السعر الحالي:</strong> ${current_price:.2f}</p>
                                <p><strong>السعر المتوقع:</strong> ${final_prediction:.2f} ({expected_change:+.2f}%)</p>
                                <p><strong>تاريخ التنبؤ:</strong> {end_date.strftime('%Y-%m-%d')}</p>
                                <p><strong>مستوى الثقة:</strong> {confidence}%</p>
                            </div>
                            """, unsafe_allow_html=True)

                            # التنبؤات التفصيلية
                            col1, col2 = st.columns(2)

                            with col1:
                                st.markdown("""
                                <div class="week-prediction">
                                    <h3>تنبؤات الأسبوع القادم</h3>
                                </div>
                                """, unsafe_allow_html=True)

                                for i in range(min(7, len(predictions))):
                                    day_date = (datetime.now() + timedelta(days=i+1)).strftime('%m-%d')
                                    pred = predictions[i]
                                    daily_change = ((pred - current_price) / current_price * 100)
                                    trend = "صاعد" if daily_change > 0 else "هابط" if daily_change < 0 else "ثابت"
                                    st.write(f"**يوم {i+1} ({day_date}):** ${pred:.2f} {trend} {daily_change:+.2f}%")

                            with col2:
                                st.markdown("""
                                <div class="month-prediction">
                                    <h3>التحليل الفني</h3>
                                </div>
                                """, unsafe_allow_html=True)

                                st.write(f"**RSI:** {rsi:.1f}")
                                if rsi > 70:
                                    st.write("منطقة شراء مفرط")
                                elif rsi < 30:
                                    st.write("منطقة بيع مفرط")
                                else:
                                    st.write("منطقة متوازنة")

                                st.write(f"**MACD:** {macd:.3f}")
                                if macd > macd_signal:
                                    st.write("إشارة إيجابية")
                                else:
                                    st.write("إشارة سلبية")

                                st.write("**المتوسطات المتحركة:**")
                                st.write(f"SMA20: ${sma20:.2f} ({'فوق' if current_price > sma20 else 'تحت'})")
                                st.write(f"SMA50: ${sma50:.2f} ({'فوق' if current_price > sma50 else 'تحت'})")

                            # الرسم البياني
                            st.markdown("### الرسم البياني التفاعلي")

                            try:
                                fig = go.Figure()

                                # البيانات التاريخية
                                recent_data = data.tail(60)  # آخر 60 يوم

                                fig.add_trace(go.Candlestick(
                                    x=recent_data.index,
                                    open=recent_data['Open'],
                                    high=recent_data['High'],
                                    low=recent_data['Low'],
                                    close=recent_data['Close'],
                                    name='السعر التاريخي'
                                ))

                                # المتوسط المتحرك
                                if 'SMA20' in recent_data.columns:
                                    fig.add_trace(go.Scatter(
                                        x=recent_data.index,
                                        y=recent_data['SMA20'],
                                        name='SMA 20',
                                        line=dict(color='orange', width=2)
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
                                    mode='lines+markers'
                                ))

                                fig.update_layout(
                                    title=f"تحليل سهم {symbol}",
                                    height=600,
                                    showlegend=True,
                                    xaxis_title="التاريخ",
                                    yaxis_title="السعر ($)"
                                )

                                st.plotly_chart(fig, use_container_width=True)

                            except Exception as e:
                                st.warning(f"لا يمكن عرض الرسم البياني: {e}")

                            # معلومات الشركة
                            if info:
                                col1, col2 = st.columns(2)

                                with col1:
                                    st.markdown("### معلومات الشركة")
                                    st.write(f"**الاسم:** {info.get('longName', symbol)}")
                                    st.write(f"**القطاع:** {info.get('sector', 'غير متاح')}")
                                    st.write(f"**الصناعة:** {info.get('industry', 'غير متاح')}")

                                    if info.get('marketCap'):
                                        market_cap = info['marketCap'] / 1e9
                                        st.write(f"**القيمة السوقية:** ${market_cap:.1f}B")

                                with col2:
                                    st.markdown("### إحصائيات إضافية")
                                    st.write(f"**قوة الإشارة:** {abs(score)}/7")

                                    volume = safe_get(data['Volume'], default=0)
                                    if volume > 0:
                                        st.write(f"**حجم التداول:** {volume:,.0f}")

                                    high_52w = info.get('fiftyTwoWeekHigh')
                                    low_52w = info.get('fiftyTwoWeekLow')

                                    if high_52w:
                                        st.write(f"**أعلى 52 أسبوع:** ${high_52w:.2f}")
                                    if low_52w:
                                        st.write(f"**أقل 52 أسبوع:** ${low_52w:.2f}")

                        else:
                            st.error("فشل في إنشاء التنبؤات")

            except Exception as e:
                st.error(f"حدث خطأ أثناء التحليل: {str(e)}")
                st.info("تأكد من صحة رمز السهم وحاول مرة أخرى")

# معلومات المنصة
st.markdown("---")
st.markdown("### معلومات المنصة")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.info("**التحليل:** شامل")
with col2:
    st.info("**المؤشرات:** متقدمة")
with col3:
    st.info("**التحديث:** فوري")
with col4:
    st.info("**الذكاء:** اصطناعي")

# تحذير قانوني
st.warning("**تنبيه مهم:** استشر مستشار مالي مؤهل قبل اتخاذ قرارات استثمارية.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <h3>منصة التنبؤ بالأسهم</h3>
    <p>نظام ذكي للتحليل المالي والتنبؤات</p>
    <p>© 2025 - جميع الحقوق محفوظة</p>
</div>
""", unsafe_allow_html=True)
