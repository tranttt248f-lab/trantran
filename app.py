import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
import random
import os
from sklearn.preprocessing import StandardScaler

# Import TensorFlow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Import local modules
from vnstock import Vnstock

# -------------------------------------------------------------
# CẤU HÌNH TRANG & CSS CUSTOM
# -------------------------------------------------------------
st.set_page_config(
    page_title="Tối ưu hóa danh mục đầu tư bằng LSTM-GRU",
    layout="wide",
    page_icon="📈"
)

# Thử import module industry_tickers
try:
    from industry_tickers import INDUSTRY_TICKERS
except ModuleNotFoundError:
    try:
        from data.industry_tickers import INDUSTRY_TICKERS
    except ModuleNotFoundError:
        st.error("❌ Không tìm thấy file `industry_tickers.py` trong thư mục gốc hoặc thư mục `data/`. Vui lòng đảm bảo bạn đã đẩy (push) file `industry_tickers.py` lên kho lưu trữ GitHub của mình.")
        st.stop()

# Thêm CSS custom để giao diện chuyên nghiệp hơn
st.markdown("""
<style>
    .reportview-container {
        background: #0f1116;
    }
    .main-header {
        font-size: 38px;
        font-weight: 800;
        background: linear-gradient(135deg, #ff4b4b 0%, #8522f0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
        text-align: center;
    }
    .sub-header {
        font-size: 16px;
        color: #8a92a6;
        text-align: center;
        margin-bottom: 30px;
    }
    .stCard {
        background-color: #161922;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #282c37;
        margin-bottom: 15px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #ff4b4b;
    }
    .metric-label {
        font-size: 14px;
        color: #8a92a6;
    }
</style>
""", unsafe_style_html=True)

st.markdown('<div class="main-header">Ứng Dụng Tối Ưu Hóa Danh Mục Đầu Tư Chứng Khoán</div>', unsafe_style_html=True)
st.markdown('<div class="sub-header">Mô hình học sâu lai LSTM-GRU tối ưu trực tiếp Hệ số Sharpe động (Dữ liệu VNStock)</div>', unsafe_style_html=True)

# -------------------------------------------------------------
# THAM SỐ CẤU HÌNH TRÊN SIDEBAR
# -------------------------------------------------------------
st.sidebar.header("⚙️ Cấu hình Tham số")

# Chọn ngành và mã cổ phiếu
industry_keys = list(INDUSTRY_TICKERS.keys())
selected_industry = st.sidebar.selectbox("Chọn ngành đầu tư:", industry_keys, index=industry_keys.index("Thép") if "Thép" in industry_keys else 0)

default_tickers = INDUSTRY_TICKERS[selected_industry]
tickers_selection = st.sidebar.multiselect(
    "Mã cổ phiếu trong ngành:", 
    options=default_tickers,
    default=default_tickers,
    help="Bạn có thể bỏ bớt mã nếu muốn tăng tốc độ tải dữ liệu."
)

# Lãi suất phi rủi ro & Ngày giao dịch
rf_annual = st.sidebar.number_input("Lãi suất phi rủi ro năm (RF):", min_value=0.0, max_value=0.2, value=0.045, step=0.005, format="%.3f")
trading_days = st.sidebar.number_input("Số ngày giao dịch/năm:", min_value=100, max_value=365, value=252, step=1)

# Cấu hình thời gian
st.sidebar.subheader("📅 Khoảng thời gian")
train_start = st.sidebar.date_input("Bắt đầu Train:", value=pd.to_datetime("2015-01-01"))
train_end = st.sidebar.date_input("Kết thúc Train:", value=pd.to_datetime("2024-12-31"))
test_start = st.sidebar.date_input("Bắt đầu Test:", value=pd.to_datetime("2025-01-01"))
test_end = st.sidebar.date_input("Kết thúc Test:", value=pd.to_datetime("2025-12-31"))

# Thông số mô hình
with st.sidebar.expander("🤖 Cấu hình LSTM-GRU"):
    top_n_select = st.slider("Số lượng Top mã chọn lọc (N):", min_value=3, max_value=20, value=10)
    window_size = st.slider("Độ rộng cửa sổ (Window Size):", min_value=5, max_value=60, value=30)
    horizon = st.slider("Horizon dự báo (Ngày):", min_value=1, max_value=20, value=5)
    lstm_units = st.slider("LSTM Units:", min_value=16, max_value=256, value=96, step=16)
    gru_units = st.slider("GRU Units:", min_value=8, max_value=128, value=48, step=8)
    epochs = st.slider("Số lượng Epochs:", min_value=10, max_value=200, value=50, step=10)
    batch_size = st.selectbox("Batch Size:", options=[16, 32, 64, 128], index=1)
    
    seeds_input = st.text_input("Danh sách Seeds ngẫu nhiên:", value="7, 21, 42, 99, 123")
    try:
        seed_list = [int(s.strip()) for s in seeds_input.split(",") if s.strip().isdigit()]
    except Exception:
        seed_list = [7, 21, 42, 99, 123]

# -------------------------------------------------------------
# HÀM TẢI DỮ LIỆU ĐƯỢC CACHE CHO TỪNG MÃ (TỐI ƯU RATE LIMIT)
# -------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def download_single_ticker(ticker, start_date_str, end_date_str):
    try:
        # Sử dụng API vnstock mới
        stock = Vnstock().stock(symbol=ticker, source="KBS")
        df = stock.quote.history(
            start=start_date_str,
            end=end_date_str,
            interval="1D"
        )
        if df is None or df.empty:
            return None
        
        df = df.copy()
        df["ticker"] = ticker
        
        # Đồng nhất cột thời gian
        if "time" not in df.columns:
            if "date" in df.columns:
                df["time"] = df["date"]
            elif "datetime" in df.columns:
                df["time"] = df["datetime"]
            else:
                return None
        
        keep_cols = [c for c in ["time", "open", "high", "low", "close", "volume", "ticker"] if c in df.columns]
        return df[keep_cols]
    except Exception:
        return None

# Khởi tạo session state
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False

# Nút tải dữ liệu
col_btn1, col_btn2 = st.columns([1, 5])
with col_btn1:
    load_btn = st.button("📥 Tải & Tiền xử lý dữ liệu", use_container_width=True)

if load_btn:
    if len(tickers_selection) == 0:
        st.error("Vui lòng chọn ít nhất 1 mã cổ phiếu.")
    else:
        st.session_state.data_loaded = False
        st.session_state.model_trained = False
        
        # Chuyển đổi định dạng ngày thành string YYYY-MM-DD
        train_start_str = train_start.strftime("%Y-%m-%d")
        test_end_str = test_end.strftime("%Y-%m-%d")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_dfs = []
        failed_tickers = []
        
        for idx, t in enumerate(tickers_selection):
            status_text.text(f"Đang tải {t} ({idx + 1}/{len(tickers_selection)})...")
            
            # Đo thời gian để xem có phải cache hit không
            t0 = time.time()
            df = download_single_ticker(t, train_start_str, test_end_str)
            elapsed = time.time() - t0
            
            if df is not None:
                all_dfs.append(df)
            else:
                failed_tickers.append(t)
                
            progress_bar.progress((idx + 1) / len(tickers_selection))
            
            # Nếu chạy thực tế không qua cache (elapsed > 0.2s), nghỉ để tránh bị block IP
            if elapsed > 0.2:
                time.sleep(2.0)
        
        progress_bar.empty()
        status_text.empty()
        
        if len(all_dfs) == 0:
            st.error("Không tải được dữ liệu cho bất kỳ mã nào. Vui lòng kiểm tra lại kết nối mạng hoặc thử lại sau.")
        else:
            raw_data = pd.concat(all_dfs, ignore_index=True)
            st.session_state.raw_data = raw_data
            st.session_state.failed_tickers = failed_tickers
            
            # Pivot thành bảng giá đóng cửa
            pivot_df = raw_data.pivot_table(
                index="time",
                columns="ticker",
                values="close",
                aggfunc="last"
            ).sort_index()
            pivot_df.index = pd.to_datetime(pivot_df.index)
            st.session_state.pivot_df_clean = pivot_df.sort_index()
            
            # Tính Daily Returns an toàn
            price_filled = st.session_state.pivot_df_clean.ffill()
            returns_df = price_filled.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="any")
            st.session_state.returns_df = returns_df
            
            # Tính Sharpe từng mã để lọc
            rf_daily = rf_annual / trading_days
            mean_ret = returns_df.mean()
            std_ret = returns_df.std().replace(0, np.nan)
            sharpe_ratio = ((mean_ret - rf_daily) / std_ret).dropna().sort_values(ascending=False)
            st.session_state.sharpe_ratio = sharpe_ratio
            
            # Chọn Top N
            actual_n = min(top_n_select, len(sharpe_ratio))
            top_symbols = sharpe_ratio.head(actual_n).index.tolist()
            st.session_state.top_symbols = top_symbols
            st.session_state.actual_n = actual_n
            
            st.session_state.price_top10 = st.session_state.pivot_df_clean[top_symbols].copy()
            st.session_state.returns_top10 = returns_df[top_symbols].copy()
            
            st.session_state.data_loaded = True
            st.success(f"Tải thành công dữ liệu! Đã lọc Top {actual_n} cổ phiếu có hệ số Sharpe tốt nhất.")
            
            if failed_tickers:
                st.warning(f"Các mã tải lỗi: {', '.join(failed_tickers)}")

# -------------------------------------------------------------
# PHÂN CHIA GIAO DIỆN CHÍNH THÀNH CÁC TABS
# -------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dữ liệu & Sharpe Ratio", 
    "⚙️ Huấn luyện Mô hình", 
    "🍕 Phân bổ tỷ trọng (LSTM-GRU)", 
    "📈 So sánh hiệu quả"
])

# -------------------------------------------------------------
# TAB 1: DỮ LIỆU & SHARPE RATIO
# -------------------------------------------------------------
with tab1:
    if st.session_state.data_loaded:
        st.markdown("### 📊 Tổng quan dữ liệu & Tính toán Sharpe Ratio")
        
        # Layout cột hiển thị chỉ số cơ bản
        col1_1, col1_2, col1_3 = st.columns(3)
        with col1_1:
            st.markdown(f"""
            <div class="stCard">
                <div class="metric-label">Tổng số mã tải thành công</div>
                <div class="metric-value">{st.session_state.pivot_df_clean.shape[1]} mã</div>
            </div>
            """, unsafe_style_html=True)
        with col1_2:
            st.markdown(f"""
            <div class="stCard">
                <div class="metric-label">Kích thước bảng dữ liệu giá</div>
                <div class="metric-value">{st.session_state.pivot_df_clean.shape[0]} ngày</div>
            </div>
            """, unsafe_style_html=True)
        with col1_3:
            st.markdown(f"""
            <div class="stCard">
                <div class="metric-label">Ngành đang chọn phân tích</div>
                <div class="metric-value" style="color: #8522f0;">{selected_industry}</div>
            </div>
            """, unsafe_style_html=True)
            
        # Vẽ biểu đồ cột hệ số Sharpe
        st.markdown("#### Biểu đồ Sharpe Ratio lịch sử của các mã trong ngành")
        sharpe_data = st.session_state.sharpe_ratio.reset_index()
        sharpe_data.columns = ["Mã cổ phiếu", "Sharpe Ratio"]
        
        fig_sharpe = px.bar(
            sharpe_data, 
            x="Mã cổ phiếu", 
            y="Sharpe Ratio",
            title="Hệ số Sharpe của các cổ phiếu (sắp xếp giảm dần)",
            color="Sharpe Ratio",
            color_continuous_scale=px.colors.sequential.Sunsetdark
        )
        fig_sharpe.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig_sharpe, use_container_width=True)
        
        # Preview dữ liệu
        col_preview1, col_preview2 = st.columns(2)
        with col_preview1:
            st.markdown("#### Preview Giá Đóng Cửa (Close Prices)")
            st.dataframe(st.session_state.pivot_df_clean.head(), use_container_width=True)
        with col_preview2:
            st.markdown(f"#### Top {st.session_state.actual_n} cổ phiếu được chọn để Train Model")
            st.write(st.session_state.top_symbols)
            st.dataframe(st.session_state.sharpe_ratio.head(st.session_state.actual_n), use_container_width=True)
            
    else:
        st.info("💡 Vui lòng bấm nút **'Tải & Tiền xử lý dữ liệu'** ở phía trên để bắt đầu tải dữ liệu lịch sử.")

# -------------------------------------------------------------
# TAB 2: HUẤN LUYỆN MÔ HÌNH
# -------------------------------------------------------------
# Helper set seed
def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# Hàm build features
def compute_rsi(price_df, period=14):
    delta = price_df.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def build_features(price_df, return_df):
    common_idx = price_df.index.intersection(return_df.index)
    price_df = price_df.loc[common_idx].copy()
    return_df = return_df.loc[common_idx].copy()

    price_df = price_df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return_df = return_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    feat_list = []

    ret_1 = return_df.copy()
    ret_1.columns = [f"{c}_ret1" for c in ret_1.columns]
    feat_list.append(ret_1)

    ret_5 = price_df.pct_change(5)
    ret_5.columns = [f"{c}_ret5" for c in ret_5.columns]
    feat_list.append(ret_5)

    ret_10 = price_df.pct_change(10)
    ret_10.columns = [f"{c}_ret10" for c in ret_10.columns]
    feat_list.append(ret_10)

    ma5_ratio = price_df / (price_df.rolling(5, min_periods=5).mean() + 1e-9) - 1
    ma5_ratio.columns = [f"{c}_ma5_ratio" for c in ma5_ratio.columns]
    feat_list.append(ma5_ratio)

    ma10_ratio = price_df / (price_df.rolling(10, min_periods=10).mean() + 1e-9) - 1
    ma10_ratio.columns = [f"{c}_ma10_ratio" for c in ma10_ratio.columns]
    feat_list.append(ma10_ratio)

    vol5 = return_df.rolling(5, min_periods=5).std()
    vol5.columns = [f"{c}_vol5" for c in vol5.columns]
    feat_list.append(vol5)

    vol10 = return_df.rolling(10, min_periods=10).std()
    vol10.columns = [f"{c}_vol10" for c in vol10.columns]
    feat_list.append(vol10)

    mom5 = price_df.pct_change(5)
    mom5.columns = [f"{c}_mom5" for c in mom5.columns]
    feat_list.append(mom5)

    rsi14 = compute_rsi(price_df, period=14) / 100.0
    rsi14.columns = [f"{c}_rsi14" for c in rsi14.columns]
    feat_list.append(rsi14)

    features = pd.concat(feat_list, axis=1)
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.dropna(axis=0, how="any")

    return features

# Hàm tạo sequences
def create_sequences_and_targets(features_df, target_returns_df, w_size, h_size=5):
    X, y, dates = [], [], []
    feat_values = features_df.values.astype(np.float32)
    target_values = target_returns_df.values.astype(np.float32)
    idx = features_df.index

    for i in range(len(features_df) - w_size - h_size + 1):
        X.append(feat_values[i:i + w_size])
        y.append(target_values[i + w_size:i + w_size + h_size].mean(axis=0))
        dates.append(idx[i + w_size + h_size - 1])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), pd.Index(dates)

# Build custom loss function
def get_sharpe_loss(rf_ann, tr_days, lambda_entropy=0.01):
    rf_daily = rf_ann / tr_days
    def loss_fn(y_true, y_pred):
        portfolio_returns = K.sum(y_true * y_pred, axis=1)
        portfolio_returns = portfolio_returns - rf_daily

        mean_returns = K.mean(portfolio_returns)
        std_returns = K.std(portfolio_returns)

        sharpe = mean_returns / (std_returns + 1e-9)

        entropy = -K.sum(y_pred * K.log(y_pred + 1e-9), axis=1)
        entropy = K.mean(entropy)

        return -sharpe - lambda_entropy * entropy
    return loss_fn

# Model building
def build_lstm_gru_model(timesteps, n_features, n_assets, lstm_u, gru_u):
    model = Sequential([
        Input(shape=(timesteps, n_features)),
        LSTM(lstm_u, return_sequences=True, activation="tanh", recurrent_activation="sigmoid"),
        Dropout(0.2),
        GRU(gru_u, return_sequences=False, activation="tanh", recurrent_activation="sigmoid"),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dropout(0.1),
        Dense(n_assets, activation="softmax")
    ])
    return model

with tab2:
    if st.session_state.data_loaded:
        st.markdown("### 🤖 Tiền xử lý & Huấn luyện mô hình lai LSTM-GRU")
        
        # Hiển thị cấu hình hiện tại
        st.info(f"**Các tham số hiện tại:** Top={st.session_state.actual_n} mã, Window Size={window_size}, Epochs={epochs}, Batch Size={batch_size}, Seeds={seed_list}")
        
        # Nút bấm bắt đầu train
        train_btn = st.button("🚀 Bắt đầu huấn luyện mô hình", use_container_width=False)
        
        if train_btn:
            st.session_state.model_trained = False
            
            # Phân chia train/test dữ liệu giá
            train_start_ts = pd.to_datetime(train_start)
            train_end_ts = pd.to_datetime(train_end)
            test_start_ts = pd.to_datetime(test_start)
            test_end_ts = pd.to_datetime(test_end)
            
            # Chia dữ liệu theo mốc thời gian
            train_prices = st.session_state.price_top10.loc[train_start_ts:train_end_ts].copy()
            test_prices  = st.session_state.price_top10.loc[test_start_ts:test_end_ts].copy()
            
            if len(train_prices) < window_size + horizon + 5:
                st.error("Dữ liệu tập Train quá ngắn cho cấu hình Window Size và Horizon hiện tại.")
            elif len(test_prices) < window_size + horizon + 5:
                st.error("Dữ liệu tập Test quá ngắn. Vui lòng nới rộng khoảng thời gian Test.")
            else:
                with st.spinner("Đang xử lý đặc trưng (Feature Engineering)..."):
                    train_prices = train_prices.sort_index().ffill().bfill()
                    test_prices  = test_prices.sort_index().ffill().bfill()
                    
                    train_returns = train_prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="any")
                    test_returns  = test_prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="any")
                    
                    st.session_state.test_returns_eval = test_returns.copy()
                    st.session_state.train_returns_eval = train_returns.copy()
                    
                    train_features = build_features(train_prices, train_returns)
                    test_features = build_features(test_prices, test_returns)
                    
                    scaler = StandardScaler()
                    train_features_scaled = pd.DataFrame(
                        scaler.fit_transform(train_features),
                        index=train_features.index,
                        columns=train_features.columns
                    )
                    test_features_scaled = pd.DataFrame(
                        scaler.transform(test_features),
                        index=test_features.index,
                        columns=test_features.columns
                    )
                    
                    train_target_returns = train_returns.loc[train_features_scaled.index].copy()
                    test_target_returns = test_returns.loc[test_features_scaled.index].copy()
                    
                    X_train, y_train_target, train_seq_dates = create_sequences_and_targets(
                        train_features_scaled, train_target_returns, window_size, horizon=horizon
                    )
                    X_test, y_test_target, test_seq_dates = create_sequences_and_targets(
                        test_features_scaled, test_target_returns, window_size, horizon=horizon
                    )
                
                st.success("Tạo Sequence dữ liệu xong!")
                
                # Biến lưu trữ tiến trình train
                results_runs = []
                best_model = None
                best_sharpe = -1e9
                best_seed = None
                best_history_data = None
                best_portfolio_returns = None
                best_pred_weights_test = None
                
                # Tiến hành train qua các seed
                seed_progress = st.progress(0)
                status_train = st.empty()
                
                for s_idx, seed in enumerate(seed_list):
                    status_train.info(f"Đang huấn luyện Seed {seed} ({s_idx + 1}/{len(seed_list)})...")
                    set_seed(seed)
                    
                    model = build_lstm_gru_model(
                        timesteps=X_train.shape[1],
                        n_features=X_train.shape[2],
                        n_assets=y_train_target.shape[1],
                        lstm_u=lstm_units,
                        gru_u=gru_units
                    )
                    
                    # Compile model với Sharpe loss tùy chỉnh
                    model.compile(
                        optimizer=Adam(learning_rate=0.0005),
                        loss=get_sharpe_loss(rf_annual, trading_days)
                    )
                    
                    callbacks = [
                        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
                        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5)
                    ]
                    
                    # Fit model
                    history = model.fit(
                        X_train,
                        y_train_target,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=False,
                        verbose=0,
                        validation_split=0.2,
                        callbacks=callbacks
                    )
                    
                    # Đánh giá trên test set
                    pred_weights_test = model.predict(X_test, verbose=0)
                    weights_test_df = pd.DataFrame(
                        pred_weights_test,
                        index=test_seq_dates,
                        columns=st.session_state.top_symbols
                    )
                    y_test_df = pd.DataFrame(
                        y_test_target,
                        index=test_seq_dates,
                        columns=st.session_state.top_symbols
                    )
                    
                    portfolio_returns = (weights_test_df * y_test_df).sum(axis=1)
                    
                    run_er = portfolio_returns.mean() * trading_days
                    run_std = portfolio_returns.std() * np.sqrt(trading_days)
                    run_sharpe = (run_er - rf_annual) / (run_std + 1e-12)
                    
                    results_runs.append({
                        "seed": seed,
                        "Lợi nhuận TB năm": run_er,
                        "Độ lệch chuẩn năm": run_std,
                        "Sharpe": run_sharpe
                    })
                    
                    if run_sharpe > best_sharpe:
                        best_sharpe = run_sharpe
                        best_seed = seed
                        best_model = model
                        best_history_data = {
                            "loss": history.history["loss"],
                            "val_loss": history.history["val_loss"]
                        }
                        best_portfolio_returns = portfolio_returns.copy()
                        best_pred_weights_test = pred_weights_test.copy()
                        
                    seed_progress.progress((s_idx + 1) / len(seed_list))
                
                status_train.success("Huấn luyện tất cả các Seeds hoàn tất!")
                seed_progress.empty()
                
                # Lưu trữ kết quả vào session state
                st.session_state.results_runs_df = pd.DataFrame(results_runs).sort_values("Sharpe", ascending=False).reset_index(drop=True)
                st.session_state.best_seed = best_seed
                st.session_state.best_history = best_history_data
                st.session_state.portfolio_returns_lstm_dynamic = best_portfolio_returns
                
                # Trọng số trung bình
                weights_lstm_gru_avg = best_pred_weights_test.mean(axis=0)
                st.session_state.results_LSTM_GRU = pd.DataFrame({
                    "Asset": st.session_state.top_symbols,
                    "Weight": weights_lstm_gru_avg
                }).sort_values("Weight", ascending=False).reset_index(drop=True)
                
                st.session_state.model_trained = True
                
        # Hiển thị kết quả huấn luyện nếu đã có
        if st.session_state.model_trained:
            st.markdown("### 🏆 Kết quả Huấn Luyện Các Runs")
            
            # Cột hiển thị best run
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                st.write("**Bảng so sánh các Seeds:**")
                st.dataframe(st.session_state.results_runs_df.style.highlight_max(subset=["Sharpe"], color="#3c1e5a"), use_container_width=True)
                st.success(f"🌟 **Seed tốt nhất:** {st.session_state.best_seed} với hệ số Sharpe = {st.session_state.results_runs_df['Sharpe'].max():.4f}")
            
            with col_b2:
                # Vẽ biểu đồ Loss của best run
                loss_df = pd.DataFrame(st.session_state.best_history)
                loss_df["Epoch"] = range(1, len(loss_df) + 1)
                
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(x=loss_df["Epoch"], y=loss_df["loss"], name="Train Loss", line=dict(color="#ff4b4b", width=2)))
                fig_loss.add_trace(go.Scatter(x=loss_df["Epoch"], y=loss_df["val_loss"], name="Val Loss", line=dict(color="#00cc96", width=2)))
                fig_loss.update_layout(
                    title=f"Biểu đồ Loss của Best Seed ({st.session_state.best_seed})",
                    xaxis_title="Epoch",
                    yaxis_title="Loss (Negative Sharpe + Entropy)",
                    template="plotly_dark",
                    height=350
                )
                st.plotly_chart(fig_loss, use_container_width=True)
                
    else:
        st.info("💡 Vui lòng tải dữ liệu ở Tab 1 trước khi bắt đầu huấn luyện mô hình.")

# -------------------------------------------------------------
# TAB 3: PHÂN BỔ TỶ TRỌNG (LSTM-GRU)
# -------------------------------------------------------------
with tab3:
    if st.session_state.model_trained:
        st.markdown("### 🍕 Tỷ trọng phân bổ danh mục đầu tư tối ưu")
        
        col_w1, col_w2 = st.columns([3, 2])
        
        with col_w1:
            st.markdown("#### Biểu đồ Treemap tỷ trọng tối ưu của mô hình học sâu")
            # Vẽ Treemap đẹp mắt với Plotly
            treemap_df = st.session_state.results_LSTM_GRU.copy()
            treemap_df["Tỷ trọng (%)"] = (treemap_df["Weight"] * 100).round(2)
            treemap_df["Mô tả"] = treemap_df["Asset"] + ": " + treemap_df["Tỷ trọng (%)"].astype(str) + "%"
            
            fig_tree = px.treemap(
                treemap_df,
                path=["Asset"],
                values="Weight",
                color="Weight",
                color_continuous_scale=px.colors.sequential.Plotly3,
                hover_data=["Tỷ trọng (%)"],
                title="Sơ đồ Treemap phân phối vốn"
            )
            fig_tree.update_layout(template="plotly_dark", height=500)
            st.plotly_chart(fig_tree, use_container_width=True)
            
        with col_w2:
            st.markdown("#### Bảng tỷ trọng chi tiết")
            st.dataframe(
                st.session_state.results_LSTM_GRU.style.format({"Weight": "{:.2%}"}),
                use_container_width=True,
                height=300
            )
            
            # Biểu đồ cột ngang
            fig_bar_w = px.bar(
                st.session_state.results_LSTM_GRU,
                x="Weight",
                y="Asset",
                orientation="h",
                title="Tỷ trọng phân bổ chi tiết",
                color="Weight",
                color_continuous_scale=px.colors.sequential.Agsunset
            )
            fig_bar_w.update_layout(template="plotly_dark", height=250, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_bar_w, use_container_width=True)
            
    else:
        st.info("💡 Vui lòng hoàn tất huấn luyện mô hình ở Tab 2 để tạo danh mục phân bổ tối ưu.")

# -------------------------------------------------------------
# TAB 4: SO SÁNH & KẾT LUẬN
# -------------------------------------------------------------
# Hàm tính toán đặc trưng danh mục
def port_char(weights_df, returns_df, annualize=True, freq=252):
    er = returns_df.mean().reset_index()
    er.columns = ["Asset", "Er"]

    weights_merged = pd.merge(weights_df, er, on="Asset", how="left")
    weights_merged["Er"] = weights_merged["Er"].fillna(0.0)

    portfolio_er_daily = np.dot(weights_merged["Weight"], weights_merged["Er"])

    cov_matrix = returns_df.cov()
    asset_order = weights_merged["Asset"].tolist()
    cov_matrix = cov_matrix.loc[asset_order, asset_order]

    w = weights_merged["Weight"].values
    portfolio_std_daily = np.sqrt(np.dot(w, np.dot(cov_matrix, w)))

    if annualize:
        portfolio_er = portfolio_er_daily * freq
        portfolio_std_dev = portfolio_std_daily * np.sqrt(freq)
    else:
        portfolio_er = portfolio_er_daily
        portfolio_std_dev = portfolio_std_daily

    return portfolio_er, portfolio_std_dev

def port_char_from_series(portfolio_return_series, annualize=True, freq=252):
    portfolio_return_series = pd.Series(portfolio_return_series).dropna()
    er_daily = portfolio_return_series.mean()
    std_daily = portfolio_return_series.std()

    if annualize:
        er = er_daily * freq
        std = std_daily * np.sqrt(freq)
    else:
        er = er_daily
        std = std_daily

    return er, std

def sharpe_port(weights_df, returns_df, rf=0.045, freq=252):
    portfolio_er, portfolio_std_dev = port_char(weights_df, returns_df, annualize=True, freq=freq)
    sharpe_ratio = (portfolio_er - rf) / (portfolio_std_dev + 1e-12)
    return sharpe_ratio

def sharpe_from_series(portfolio_return_series, rf=0.045, freq=252):
    portfolio_er, portfolio_std_dev = port_char_from_series(portfolio_return_series, annualize=True, freq=freq)
    sharpe_ratio = (portfolio_er - rf) / (portfolio_std_dev + 1e-12)
    return sharpe_ratio

def build_allocation_80_20(train_returns, rf_ann, tr_days):
    rf_daily = rf_ann / tr_days
    mean_ret = train_returns.mean()
    std_ret = train_returns.std().replace(0, np.nan)
    sharpe_train = ((mean_ret - rf_daily) / std_ret).dropna().sort_values(ascending=False)

    ranked = sharpe_train.reset_index()
    ranked.columns = ["Asset", "Score"]

    n_assets = len(ranked)
    top_count = max(1, int(np.ceil(0.2 * n_assets)))
    bottom_count = n_assets - top_count

    top_weights = [0.8 / top_count] * top_count
    bottom_weights = [0.2 / bottom_count] * bottom_count if bottom_count > 0 else []

    ranked["Weight"] = top_weights + bottom_weights
    return ranked[["Asset", "Weight"]]

with tab4:
    if st.session_state.model_trained:
        st.markdown("### 📊 So sánh hiệu quả các chiến lược trên dữ liệu Test")
        
        # 1. Khởi tạo danh mục phân bổ đều (Allo_1)
        Allo_1 = pd.DataFrame({
            "Asset": st.session_state.top_symbols,
            "Weight": [1 / len(st.session_state.top_symbols)] * len(st.session_state.top_symbols)
        })
        
        # 2. Khởi tạo danh mục phân bổ 80-20 (Allo_2)
        Allo_2 = build_allocation_80_20(st.session_state.train_returns_eval, rf_annual, trading_days)
        
        # 3. Tính toán các metric
        test_returns_eval = st.session_state.test_returns_eval
        portfolio_returns_lstm_dynamic = st.session_state.portfolio_returns_lstm_dynamic
        
        Er_lstm, std_lstm = port_char_from_series(portfolio_returns_lstm_dynamic, annualize=True, freq=trading_days)
        Er_1, std_1 = port_char(Allo_1, test_returns_eval, annualize=True, freq=trading_days)
        Er_2, std_2 = port_char(Allo_2, test_returns_eval, annualize=True, freq=trading_days)
        
        sharpe_lstm = sharpe_from_series(portfolio_returns_lstm_dynamic, rf=rf_annual, freq=trading_days)
        sharpe_1 = sharpe_port(Allo_1, test_returns_eval, rf=rf_annual, freq=trading_days)
        sharpe_2 = sharpe_port(Allo_2, test_returns_eval, rf=rf_annual, freq=trading_days)
        
        # Tạo bảng so sánh
        comparison_table = pd.DataFrame({
            "Chiến lược đầu tư": ["LSTM-GRU (Dynamic)", "Phân bổ đều", "Phân bổ 80-20"],
            "Lợi nhuận trung bình (%)": [Er_lstm * 100, Er_1 * 100, Er_2 * 100],
            "Độ lệch chuẩn (%)": [std_lstm * 100, std_1 * 100, std_2 * 100],
            "Hệ số Sharpe": [sharpe_lstm, sharpe_1, sharpe_2]
        })
        
        # Hiển thị bảng số liệu
        st.markdown("#### Bảng chỉ số so sánh chi tiết")
        st.dataframe(
            comparison_table.style.format({
                "Lợi nhuận trung bình (%)": "{:.2f}%",
                "Độ lệch chuẩn (%)": "{:.2f}%",
                "Hệ số Sharpe": "{:.4f}"
            }).highlight_max(subset=["Hệ số Sharpe"], color="#3c1e5a"),
            use_container_width=True
        )
        
        # Biểu đồ so sánh Plotly đẹp mắt (Dual-axis)
        st.markdown("#### Biểu đồ so sánh trực quan (Dual Axis)")
        
        categories = comparison_table["Chiến lược đầu tư"].tolist()
        er_values = comparison_table["Lợi nhuận trung bình (%)"].tolist()
        std_values = comparison_table["Độ lệch chuẩn (%)"].tolist()
        sharpe_values = comparison_table["Hệ số Sharpe"].tolist()
        
        fig_comp = go.Figure()
        
        # Trục bên trái: Cột Lợi nhuận & Độ lệch chuẩn
        fig_comp.add_trace(go.Bar(
            name="Lợi nhuận (%)",
            x=categories,
            y=er_values,
            marker_color="#00cc96",
            yaxis="y1",
            text=[f"{v:.2f}%" for v in er_values],
            textposition="auto"
        ))
        
        fig_comp.add_trace(go.Bar(
            name="Độ lệch chuẩn (%)",
            x=categories,
            y=std_values,
            marker_color="#8522f0",
            yaxis="y1",
            text=[f"{v:.2f}%" for v in std_values],
            textposition="auto"
        ))
        
        # Trục bên phải: Line Sharpe Ratio
        fig_comp.add_trace(go.Scatter(
            name="Sharpe Ratio",
            x=categories,
            y=sharpe_values,
            yaxis="y2",
            mode="lines+markers+text",
            line=dict(color="#ff4b4b", width=3),
            marker=dict(size=10),
            text=[f"{v:.2f}" for v in sharpe_values],
            textposition="top center"
        ))
        
        # Thiết kế layout dual-axis
        fig_comp.update_layout(
            title="So sánh Hiệu quả của các Chiến lược đầu tư",
            xaxis=dict(title="Chiến lược đầu tư"),
            yaxis=dict(
                title="Giá trị (%)",
                titlefont=dict(color="#8522f0"),
                tickfont=dict(color="#8522f0")
            ),
            yaxis2=dict(
                title="Hệ số Sharpe",
                titlefont=dict(color="#ff4b4b"),
                tickfont=dict(color="#ff4b4b"),
                overlaying="y",
                side="right"
            ),
            template="plotly_dark",
            barmode="group",
            height=500,
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)")
        )
        
        st.plotly_chart(fig_comp, use_container_width=True)
        
        # Kết luận nhanh
        best_idx = comparison_table["Hệ số Sharpe"].idxmax()
        best_strategy = comparison_table.loc[best_idx, "Chiến lược đầu tư"]
        best_sharpe_val = comparison_table.loc[best_idx, "Hệ số Sharpe"]
        
        st.markdown(f"""
        <div class="stCard" style="border-left: 5px solid #ff4b4b;">
            <h4>📝 Kết Luận Nhanh</h4>
            <p>Dựa trên kết quả thực nghiệm trên tập Test, chiến lược có hiệu quả tốt nhất (Hệ số Sharpe cao nhất) là 
            <strong>{best_strategy}</strong> với hệ số Sharpe đạt <strong>{best_sharpe_val:.4f}</strong>.</p>
            <p>Mô hình LSTM-GRU sử dụng Sharpe Loss tối ưu trực tiếp hiệu số Sharpe danh mục và tự động điều chỉnh 
            tỷ trọng giúp cân bằng giữa lợi nhuận dự phóng và rủi ro tương quan giữa các tài sản.</p>
        </div>
        """, unsafe_style_html=True)
        
    else:
        st.info("💡 Vui lòng hoàn tất huấn luyện mô hình ở Tab 2 để tiến hành phân tích so sánh hiệu quả các chiến lược.")
