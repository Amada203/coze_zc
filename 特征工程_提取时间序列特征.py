import pandas as pd
from prophet import Prophet
import numpy as np

# 读取数据
df = pd.read_csv('/Users/apple/Downloads/query-impala-1632023.csv')
# 输出数据基本信息
print("数据基本信息:")
print(f"数据形状: {df.shape}")
print(f"SKU数量: {df['sku_id'].nunique()}")
print(f"日期范围: {df['dt'].min()} 至 {df['dt'].max()}")
print(f"价格范围: {df['page_price'].min():.2f} 至 {df['page_price'].max():.2f}")
print(f"促销商品数量: {df[df['is_promotion']==1].shape[0]}")
print(f"非促销商品数量: {df[df['is_promotion']==0].shape[0]}")


# 转换日期格式
df['dt'] = pd.to_datetime(df['dt'])

# 按sku_id分组处理
results = []
for sku_id, group in df.groupby('sku_id'):
    # 准备prophet所需格式
    prophet_df = group[['dt', 'page_price']].rename(columns={'dt': 'ds', 'page_price': 'y'})
    
    # 初始化并拟合prophet模型
    model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=True)
    model.fit(prophet_df)
    
    # 预测未来1天
    future = model.make_future_dataframe(periods=1)
    forecast = model.predict(future)
    
    # 提取特征
    last_row = forecast.iloc[-1]
    features = {
        'sku_id': sku_id,
        'trend': last_row['trend'],
        'yearly': last_row['yearly'],
        'weekly': last_row['weekly'],
        'yhat': last_row['yhat'],
        'yhat_lower': last_row['yhat_lower'],
        'yhat_upper': last_row['yhat_upper']
    }
    results.append(features)

# 转换为DataFrame
features_df = pd.DataFrame(results)

# 保存特征数据
features_df.to_csv('商品价格时间序列特征.csv', index=False)
print("文件'商品价格时间序列特征.csv'已成功保存。")