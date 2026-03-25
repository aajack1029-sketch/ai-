import yfinance as yf
import pandas as pd
import datetime

# 1. 定義標的
tickers = {
    'TSMC': '2330.TW',        
    'Delta': '2308.TW',       # 電 (AI電源)
    'GlobalWafers': '6488.TWO', # 矽 (矽晶圓)
    'MegaUnion': '6944.TW',   # 水 (水處理工程)           
    'TSEC': '^TWII'           
}

print(f"[{datetime.datetime.now()}] 啟動第一步：供應鏈數據採集...")

# 2. 下載數據 (auto_adjust=True 確保拿到還原股價)
raw_df = yf.download(list(tickers.values()), start="2023-01-01", auto_adjust=True)
df_raw = raw_df['Close'].rename(columns={v: k for k, v in tickers.items()})

# 3. 清洗並存檔
df_clean = df_raw.ffill().dropna()
df_clean.to_csv("raw_stock_data.csv")

print(f"✅ 下載完成！樣本起點: {df_clean.index[0].date()}, 樣本數: {len(df_clean)}")