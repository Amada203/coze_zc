import pandas as pd
from datetime import datetime, timedelta

# 读取预测结果
try:
    df = pd.read_csv('未来价格变动预测结果.csv')
except FileNotFoundError:
    df = pd.read_csv('discount_price变动预测结果_全特征版_lightgbm.csv')

# 字段名适配
if 'price_change_prob' in df.columns:
    df = df.rename(columns={'price_change_prob': 'prob'})
if 'dt' in df.columns:
    df = df.rename(columns={'dt': 'date'})

# 强制prob为float
if 'prob' in df.columns:
    df['prob'] = pd.to_numeric(df['prob'], errors='coerce')

# 只保留需要的字段
df = df[['sku_id', 'date', 'prob']]

# 只保留测试集最大日期之后的采样计划
future_start_date = '2025-04-30'
df = df[df['date'] > future_start_date].copy()

# 日期转为datetime
df['date'] = pd.to_datetime(df['date'])

# 全量采样成本
total_samples = len(df)

# 分层采样法
# 高概率（>0.5）全部采样
high_mask = df['prob'] > 0.5
# 中概率（0.2~0.5）隔天采样
mid_mask = (df['prob'] > 0.2) & (df['prob'] <= 0.5)
# 低概率（<=0.2）每周采样一次
low_mask = df['prob'] <= 0.2

# 采样标记初始化
df['is_sampled'] = False

# 高概率全部采样
df.loc[high_mask, 'is_sampled'] = True

# 中概率隔天采样
df_mid = df[mid_mask].copy()
if not df_mid.empty:
    # 按sku分组，隔天采样
    for sku, group in df_mid.groupby('sku_id'):
        group = group.sort_values('date')
        sampled_dates = group['date'].iloc[::2]  # 隔天采样
        df.loc[group.index[group['date'].isin(sampled_dates)], 'is_sampled'] = True

# 低概率每周采样一次
df_low = df[low_mask].copy()
if not df_low.empty:
    for sku, group in df_low.groupby('sku_id'):
        group = group.sort_values('date')
        # 每周采样一次（每7天的第一天）
        sampled_dates = group['date'].iloc[::7]
        df.loc[group.index[group['date'].isin(sampled_dates)], 'is_sampled'] = True

# 优化后采样成本
optimized_samples = df['is_sampled'].sum()

# 实际捕捉率
actual_capture_rate = df.loc[df['is_sampled'], 'prob'].sum() / df['prob'].sum() if df['prob'].sum() > 0 else 0

# 成本节省百分比
cost_saving = 1 - optimized_samples / total_samples if total_samples > 0 else 0

# 输出采样计划
df['是否采样'] = df['is_sampled'].map({True: '是', False: '否'})
df[['sku_id', 'date', 'prob', '是否采样']].to_csv('采样计划_分层采样法.csv', index=False, encoding='utf-8-sig')

# 打印优化评估
print(f'全量采样总次数: {total_samples}')
print(f'优化后采样次数: {optimized_samples}')
print(f'节省采样成本: {cost_saving:.2%}')
print(f'实际捕捉率: {actual_capture_rate:.2%}') 