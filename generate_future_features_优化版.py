import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

# 1. 读取历史数据
history_file = 'query-impala-1632023.csv'
df = pd.read_csv(history_file)
df['dt'] = pd.to_datetime(df['dt'])

# 1.1 补齐每个SKU的日期到2025-04-30
all_sku = df['sku_id'].unique()
full_dates = pd.date_range(df['dt'].min(), '2025-04-30')
full_idx = pd.MultiIndex.from_product([all_sku, full_dates], names=['sku_id', 'dt'])
df = df.set_index(['sku_id', 'dt']).reindex(full_idx).sort_index().reset_index()

# 1.2 前向填充（每个sku单独填充），最早也缺则后向填充
for col in df.columns:
    if col not in ['sku_id', 'dt']:
        df[col] = df.groupby('sku_id')[col].ffill().bfill()

# 2. 读取特征名
with open('feature_cols_v1.json', 'r') as f:
    feature_cols = json.load(f)

# 确保feature_cols包含sku_id和dt
if 'sku_id' not in feature_cols:
    feature_cols = ['sku_id'] + feature_cols
if 'dt' not in feature_cols:
    feature_cols = ['dt'] + feature_cols

# 3. 获取sku_id列表
sku_ids = df['sku_id'].unique()

# 4. 只生成T+1（2025-05-01）的特征
target_date = pd.Timestamp('2025-05-01')
future_dates = [target_date]

# 5. 生成未来采样表
future_rows = []
for sku in sku_ids:
    for date in future_dates:
        future_rows.append({'sku_id': sku, 'dt': date})
future_df = pd.DataFrame(future_rows)

# 6. merge Prophet静态特征
prophet_static = pd.read_csv('商品价格时间序列特征_v2.csv')
static_cols = ['sku_id', 'trend', 'yearly', 'weekly']
prophet_static = prophet_static[static_cols].drop_duplicates('sku_id')
future_df = pd.merge(future_df, prophet_static, on='sku_id', how='left')

# 7. 时间特征真实生成
future_df['year'] = future_df['dt'].dt.year
future_df['month'] = future_df['dt'].dt.month
future_df['day'] = future_df['dt'].dt.day
future_df['weekday'] = future_df['dt'].dt.weekday
future_df['quarter'] = future_df['dt'].dt.quarter
future_df['is_weekend'] = future_df['weekday'].isin([5, 6]).astype(int)
future_df['is_month_start'] = future_df['dt'].dt.is_month_start.astype(int)
future_df['is_month_end'] = future_df['dt'].dt.is_month_end.astype(int)
future_df['is_quarter_start'] = future_df['dt'].dt.is_quarter_start.astype(int)
future_df['is_quarter_end'] = future_df['dt'].dt.is_quarter_end.astype(int)

# 8. 促销/节假日特征按日历生成
future_df['is_promotion'] = 0
future_df['is_promotion_period'] = 0
# 劳动节 5月1-5日
future_df['is_labor_day'] = future_df['dt'].isin(pd.date_range('2025-05-01', '2025-05-05')).astype(int)
future_df['is_promotion_period'] = future_df['is_promotion_period'] | future_df['is_labor_day']
# 你可按需补充其他大促/节假日

# 9. 滞后特征递推（以discount_price为例）
last_day = pd.to_datetime('2025-04-30')
last_df = df[df['dt'] <= last_day].copy()

for sku in sku_ids:
    sku_hist = last_df[last_df['sku_id'] == sku].sort_values('dt')
    # 取最后N天历史
    price_hist = list(sku_hist['discount_price'].values[-7:]) if 'discount_price' in sku_hist else [np.nan]*7
    for i, date in enumerate(future_dates):
        idx = (future_df['sku_id'] == sku) & (future_df['dt'] == date)
        # discount_price_lag_1~7
        for lag in [1,3,7,14,30]:
            col = f'discount_price_lag_{lag}'
            if col in feature_cols:
                if lag <= len(price_hist):
                    future_df.loc[idx, col] = price_hist[-lag]
                else:
                    future_df.loc[idx, col] = price_hist[0] if price_hist else np.nan
        # 其他滞后特征、滑动均值等
        if 'discount_price' in feature_cols:
            future_df.loc[idx, 'discount_price'] = price_hist[-1] if price_hist else np.nan
        if 'discount_price_mean_7' in feature_cols:
            future_df.loc[idx, 'discount_price_mean_7'] = np.mean(price_hist) if price_hist else np.nan
        if 'discount_price_std_7' in feature_cols:
            future_df.loc[idx, 'discount_price_std_7'] = np.std(price_hist) if price_hist else np.nan
        if 'discount_price_min_7' in feature_cols:
            future_df.loc[idx, 'discount_price_min_7'] = np.min(price_hist) if price_hist else np.nan
        if 'discount_price_max_7' in feature_cols:
            future_df.loc[idx, 'discount_price_max_7'] = np.max(price_hist) if price_hist else np.nan
        # 递推历史窗口
        price_hist.append(price_hist[-1] if price_hist else np.nan)
        if len(price_hist) > 30:
            price_hist = price_hist[-30:]

# 10. 其余特征如无法递推则用T-1填充
t1_df = df[df['dt'] == last_day].copy()  # 只保留2025-04-30那天
t1_df = t1_df.drop_duplicates('sku_id', keep='last')
for col in feature_cols:
    if col in ['sku_id', 'dt', 'trend', 'yearly', 'weekly', 'year', 'month', 'day', 'weekday', 'quarter', 'is_weekend', 'is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end', 'is_promotion', 'is_promotion_period', 'is_labor_day', 'discount_price', 'discount_price_lag_1', 'discount_price_lag_3', 'discount_price_lag_7', 'discount_price_lag_14', 'discount_price_lag_30', 'discount_price_mean_7', 'discount_price_std_7', 'discount_price_min_7', 'discount_price_max_7']:
        continue
    if col not in future_df.columns:
        if col in t1_df.columns:
            t1_map = t1_df.set_index('sku_id')[col]
        else:
            t1_map = None
        if t1_map is not None:
            future_df[col] = future_df['sku_id'].map(t1_map)
        else:
            future_df[col] = 0

# 11. 保证所有特征齐全
for col in feature_cols:
    if col not in future_df.columns:
        future_df[col] = 0  # 缺失特征补0
# 12. 按特征顺序排列
future_df = future_df[[col for col in feature_cols if col in future_df.columns]]
# 13. 保存
future_df.to_csv('未来特征数据_20250501_优化版.csv', index=False, encoding='utf-8-sig')
print('未来特征数据已保存到 未来特征数据_20250501_优化版.csv') 