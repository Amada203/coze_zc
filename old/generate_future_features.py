import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

# 1. 读取历史数据
history_file = 'query-impala-1632023.csv'
df = pd.read_csv(history_file)
df['dt'] = pd.to_datetime(df['dt'])

# 2. 读取特征名
with open('feature_cols.json', 'r') as f:
    feature_cols = json.load(f)

# 确保feature_cols包含sku_id和dt
if 'sku_id' not in feature_cols:
    feature_cols = ['sku_id'] + feature_cols
if 'dt' not in feature_cols:
    feature_cols = ['dt'] + feature_cols

# 3. 获取sku_id列表
sku_ids = df['sku_id'].unique()

# 4. 生成未来日期
future_dates = pd.date_range('2025-05-01', '2025-05-07')

# 5. 生成未来采样表
future_rows = []
for sku in sku_ids:
    for date in future_dates:
        future_rows.append({'sku_id': sku, 'dt': date})
future_df = pd.DataFrame(future_rows)

# 6. 用T-1（2025-04-30）数据补全特征
last_day = pd.to_datetime('2025-04-30')
last_df = df[df['dt'] == last_day].set_index('sku_id')

# 7. 补全所有特征（除sku_id和dt外）
for col in feature_cols:
    if col in ['sku_id', 'dt']:
        continue
    if col in last_df.columns:
        future_df[col] = future_df['sku_id'].map(last_df[col])
    else:
        # 若历史数据无此特征，填充为0
        future_df[col] = 0

# 8. merge Prophet静态特征
prophet_static = pd.read_csv('商品价格时间序列特征_v2.csv')
static_cols = ['sku_id', 'trend', 'yearly', 'weekly']
prophet_static = prophet_static[static_cols].drop_duplicates('sku_id')
future_df = pd.merge(future_df, prophet_static, on='sku_id', how='left')

# 9. 时间特征重新计算
future_df['dt'] = pd.to_datetime(future_df['dt'])
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

# 10. 保证字段顺序与训练/测试集一致，并确保sku_id和dt在前两列
future_df = future_df[[col for col in feature_cols if col in future_df.columns]]

# 11. 保存
future_df.to_csv('未来特征数据_20250501_20250507.csv', index=False, encoding='utf-8-sig')
print('未来特征数据已保存到 未来特征数据_20250501_20250507.csv') 