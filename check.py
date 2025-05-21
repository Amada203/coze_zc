import pandas as pd
import json

# 读取未来特征数据
future_df = pd.read_csv('未来特征数据_20250501_优化版.csv')

# 读取特征名
with open('feature_cols_v1.json', 'r') as f:
    feature_cols = json.load(f)

print('feature_cols_v1.json字段数:', len(feature_cols))
print('未来特征数据字段数:', len(future_df.columns))
print('未来特征数据字段:', list(future_df.columns))
print('feature_cols:', feature_cols)

missing = [col for col in feature_cols if col not in future_df.columns]
extra = [col for col in future_df.columns if col not in feature_cols]

print('缺失字段:', missing)
print('多余字段:', extra)