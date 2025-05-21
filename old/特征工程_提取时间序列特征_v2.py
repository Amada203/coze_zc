import pandas as pd
from prophet import Prophet
import numpy as np

# 读取训练集数据（假设训练集文件名为预处理后的商品数据_训练集.csv）
df = pd.read_csv('预处理后的商品数据_训练集.csv')
# 输出数据基本信息
print("数据基本信息:")
print(f"数据形状: {df.shape}")
print(f"SKU数量: {df['sku_id'].nunique()}")
print(f"日期范围: {df['dt'].min()} 至 {df['dt'].max()}")
print(f"价格范围: {df['page_price'].min():.2f} 至 {df['page_price'].max():.2f}")
print(f"促销商品记录数: {df[df['is_promotion']==1].shape[0]}")
print(f"非促销商品记录数: {df[df['is_promotion']==0].shape[0]}")

# 转换日期格式
df['dt'] = pd.to_datetime(df['dt'])

# 按sku_id分组处理
results = []
for sku_id, group in df.groupby('sku_id'):
    # 准备prophet所需格式
    prophet_df = group[['dt', 'discount_price']].rename(columns={'dt': 'ds', 'discount_price': 'y'})
    # 初始化并拟合prophet模型
    model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=True)
    model.fit(prophet_df)
    # 只用训练集历史数据做特征提取（不预测未来）
    forecast = model.predict(prophet_df[['ds']])
    # 取最后一天的分解特征
    last_row = forecast.iloc[-1]
    features = {
        'sku_id': sku_id,
        'trend': last_row['trend'],
        'yearly': last_row['yearly'],
        'weekly': last_row['weekly']
    }
    results.append(features)

# 转换为DataFrame
features_df = pd.DataFrame(results)

# 保存特征数据
features_df.to_csv('商品价格时间序列特征_v2.csv', index=False)
print("文件'商品价格时间序列特征_v2.csv'已成功保存。") 