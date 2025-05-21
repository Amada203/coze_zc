import pandas as pd
import joblib
import json

# 1. 加载未来特征数据
future_df = pd.read_csv('未来特征数据_20250501_优化版.csv')

# 2. 加载训练好的模型
model = joblib.load('lightgbm_price_change_model_v1.pkl')

# 3. 加载特征名
with open('feature_cols_v1.json', 'r') as f:
    feature_cols = json.load(f)

# 4. 特征对齐（自动过滤掉主键字段）
# 只保留模型需要的特征，且顺序一致
X_future = future_df[[col for col in feature_cols if col in future_df.columns]]

print('模型需要特征数:', len(feature_cols))
print('实际输入特征数:', X_future.shape[1])
print('输入特征名:', list(X_future.columns))

# 新增：检查输入特征分布和唯一值数
print('输入特征描述:')
print(X_future.describe())
print('每列唯一值个数:')
print(X_future.nunique())
print('模型特征名:')
print(model.booster_.feature_name())
print('输入特征名:')
print(list(X_future.columns))

if len(feature_cols) != X_future.shape[1]:
    raise ValueError(f"特征数量不一致！模型需要{len(feature_cols)}个特征，实际输入{X_future.shape[1]}个特征。请检查feature_cols_v1.json和未来特征数据_20250501_优化版.csv的列名是否完全一致，且无多余或缺失。")

# 新增：加载scaler并标准化特征
scaler = joblib.load('feature_scaler_v1.pkl')
X_future_scaled = scaler.transform(X_future)

# 5. 预测变动概率
future_df['price_change_prob'] = model.predict_proba(X_future_scaled)[:, 1]

# 6. 自动查找sku和dt列名
print('当前DataFrame列名:', list(future_df.columns))
sku_col = [col for col in future_df.columns if 'sku' in col][0]
dt_col = [col for col in future_df.columns if 'dt' in col][0]

# 7. 输出预测结果
future_df[[sku_col, dt_col, 'price_change_prob']].to_csv('未来价格变动预测结果_优化版.csv', index=False, encoding='utf-8-sig')
print("未来价格变动预测结果已保存到 未来价格变动预测结果_优化版.csv") 