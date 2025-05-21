import pandas as pd
from datetime import datetime

# 读取预测结果
# 自动适配字段名
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

# 2. 只保留需要的字段
df = df[['sku_id', 'date', 'prob']]

# 新增：只保留测试集最大日期之后的采样计划
future_start_date = '2025-04-30'
df = df[df['date'] > future_start_date].copy()

# 3. 全量采样成本
total_samples = len(df)

# 4. 按概率降序排序
df = df.sort_values(by='prob', ascending=False).reset_index(drop=True)

# 5. 计算总变动概率
total_prob = df['prob'].sum()

# 6. 累计采样，直到捕捉率达到目标
target_capture_rate = 0.95  # 你可以调整为0.98、0.97等
df['cum_prob'] = df['prob'].cumsum()
df['capture_rate'] = df['cum_prob'] / total_prob
df['is_sampled'] = df['capture_rate'] <= target_capture_rate

# 7. 优化后采样成本
optimized_samples = df['is_sampled'].sum() if len(df) > 0 else 0

# 8. 实际捕捉率
actual_capture_rate = df.loc[df['is_sampled'], 'prob'].sum() / total_prob if total_prob > 0 else 0

# 9. 成本节省百分比
cost_saving = 1 - optimized_samples / total_samples if total_samples > 0 else 0

# 10. 输出采样计划
df['是否采样'] = df['is_sampled'].map({True: '是', False: '否'}) if len(df) > 0 else []
df[['sku_id', 'date', 'prob', '是否采样']].to_csv('采样计划.csv', index=False, encoding='utf-8-sig')

# 11. 打印优化评估
print(f'全量采样总次数: {total_samples}')
print(f'优化后采样次数: {optimized_samples}')
print(f'节省采样成本: {cost_saving:.2%}')
print(f'实际捕捉率: {actual_capture_rate:.2%}') 