import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os

# 指定 Keras 後端
os.environ["KERAS_BACKEND"] = "tensorflow"
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

import shap
import matplotlib.pyplot as plt

# ==========================================
# 1. 數據特徵處理 (TXF 替代 VIX 邏輯)
# ==========================================
df = pd.read_csv("final_training_data.csv", index_col=0, parse_dates=True)

# [新加入] 波動率提取 (Volatility Extraction): 計算大盤的 20日歷史波動率
df['TSEC_Volatility'] = df['TSEC_Lag1_Ret'].rolling(window=20).std()

# 更新因子清單：將 VIX 替換為台指期相關動能 (這裡以 TSEC 代表大盤 Beta)
features = [
    'Delta_Lag1_Ret',       # 軌道 A: 實體供應鏈 (電)
    'GlobalWafers_Lag1_Ret',# 軌道 A: 實體供應鏈 (矽)
    'MegaUnion_Lag1_Ret',   # 軌道 A: 實體供應鏈 (水)
    'TSEC_Lag1_Ret',        # 軌道 B: 市場動能 (台指期/大盤)
    'TSEC_Volatility',      # 軌道 B: 模擬恐慌程度 (ATR/Std)
    'RSI_Lag1',             # 技術過濾
    'BBWidth_Lag1',         # 波動擠壓
    'BBPercent_Lag1'        # 位階判斷
]

target = 'TSMC_Today_Ret'
X = df[features].dropna()
y = df.loc[X.index, target]

# 切分訓練與驗證集 (為了執行 Early Stopping)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# ==========================================
# 3. XGBoost 參數優化與早停機制
# ==========================================
print("正在訓練優化版 XGBoost (新版參數設定)...")

# 新版 XGBoost 將早停參數放在初始化中
xgb_model = xgb.XGBRegressor(
    n_estimators=1000,         
    learning_rate=0.03,
    max_depth=6,
    objective='reg:squarederror',
    early_stopping_rounds=20,    # 直接在這裡設定早停輪數
    tree_method='hist'
)

# 執行訓練
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# ==========================================
# 2. LSTM 訓練策略 (增加穩定性)
# ==========================================
X_train_lstm = np.array(X_train).reshape((X_train.shape[0], 1, X_train.shape[1]))
X_val_lstm = np.array(X_val).reshape((X_val.shape[0], 1, X_val.shape[1]))

lstm_model = Sequential([
    LSTM(64, activation='relu', input_shape=(1, X_train.shape[1])),
    Dense(32, activation='relu'),
    Dropout(0.3), 
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')

# 確保 LSTM 也有早停，防止死背數據
lstm_callback = EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    restore_best_weights=True  # 訓練停止後自動恢復到最好的那一組權重
)

lstm_model.fit(
    X_train_lstm, y_train, 
    validation_data=(X_val_lstm, y_val), 
    epochs=100, 
    callbacks=[lstm_callback], 
    verbose=0
)
# ==========================================
# 4. SHAP 分解與重要性演變
# ==========================================
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X)

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X, show=False)
plt.title("XGBoost SHAP: Supply Chain Alpha vs Market Beta")
plt.show()
# 第三步：因子重要性隨時間演變圖 (演變分析)