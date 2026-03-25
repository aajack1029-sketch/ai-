import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 1. 讀取數據
df = pd.read_csv("raw_stock_data.csv", index_col=0, parse_dates=True)

# 2. 計算技術指標 (原始價格計算)
ma20 = df['TSMC'].rolling(window=20).mean()
std20 = df['TSMC'].rolling(window=20).std()
df['BB_Width'] = (ma20 + (std20 * 2) - (ma20 - (std20 * 2))) / ma20
df['BB_Percent'] = (df['TSMC'] - (ma20 - std20 * 2)) / (std20 * 4 + 1e-9)

delta_p = df['TSMC'].diff()
gain = (delta_p.where(delta_p > 0, 0)).rolling(window=14).mean()
loss = (-delta_p.where(delta_p < 0, 0)).rolling(window=14).mean()
df['RSI'] = 100 - (100 / (1 + (gain/(loss + 1e-9))))

# 3. 核心轉換：計算收益率 (Percentage Change)
df_ret = df[['TSMC', 'Delta', 'GlobalWafers', 'MegaUnion', 'TSEC']].pct_change() * 100

# 4. 建立 AI 特徵矩陣
ml_final = pd.DataFrame(index=df.index)
ml_final['TSMC_Today_Ret'] = df_ret['TSMC']

factors = ['Delta', 'GlobalWafers', 'MegaUnion', 'TSEC']
for f in factors:
    ml_final[f'{f}_Lag1_Ret'] = df_ret[f].shift(1)

ml_final['BBWidth_Lag1'] = df['BB_Width'].shift(1)
ml_final['BBPercent_Lag1'] = df['BB_Percent'].shift(1)
ml_final['RSI_Lag1'] = df['RSI'].shift(1)

df_heatmap = ml_final.dropna()

# ==========================================
# 5. 繪製優化版熱圖 (解決文字卡到問題)
# ==========================================
# 增加畫布寬度 (figsize=(16, 10))
fig, ax = plt.subplots(figsize=(16, 10))

corr_matrix = df_heatmap.corr()

# 繪製熱圖
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", center=0, 
            linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8})

# --- 標記關鍵區域 ---
cols = corr_matrix.columns.tolist()
highlights = [('TSMC_Today_Ret', 'Delta_Lag1_Ret'), 
              ('TSMC_Today_Ret', 'GlobalWafers_Lag1_Ret'),
              ('TSMC_Today_Ret', 'MegaUnion_Lag1_Ret'),
              ('TSMC_Today_Ret', 'TSEC_Lag1_Ret')]

for target, factor in highlights:
    if target in cols and factor in cols:
        x_idx = cols.index(factor)
        y_idx = cols.index(target)
        rect = patches.Rectangle((x_idx, y_idx), 1, 1, fill=False, edgecolor='black', lw=4)
        ax.add_patch(rect)

# --- 佈局優化 (重點解決文字被卡到) ---
plt.xticks(rotation=45, ha='right', fontsize=10) # 旋轉 45 度並靠右對齊
plt.yticks(rotation=0, fontsize=10)
plt.title("TSMC Supply Chain & Bollinger Analysis (Return-Based)", fontsize=18, pad=20)

# 手動留白，確保標籤不會超出邊框
plt.subplots_adjust(bottom=0.2, left=0.2) 

plt.show()

# 6. 存檔
df_heatmap.to_csv("final_training_data.csv")
print("✅ 特徵工程完成，熱圖顯示已優化！")