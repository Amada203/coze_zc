import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 读取时间序列特征数据
with open('价格数据时间序列特征.json', 'r', encoding='utf-8') as f:
    features_data = json.load(f)

# 转换为DataFrame
df = pd.DataFrame(features_data)

# 数据预处理
# 1. 将日期转换为datetime类型
df['ds'] = pd.to_datetime(df['ds'])

# 2. 提取价格变动特征
# 假设我们使用trend作为目标变量，计算每日价格变动
df['price_change'] = df['trend'].diff().fillna(0)

# 3. 创建分类标签：1表示价格上涨，0表示价格不变或下跌
df['label'] = (df['price_change'] > 0).astype(int)

# 4. 准备特征和标签
features = ['trend', 'yearly', 'weekly', 'additive_terms', 'multiplicative_terms']
X = df[features]
y = df['label']

# 5. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. 创建LightGBM数据集
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# 设置LightGBM参数
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'verbose': 0
}

# 训练模型
num_round = 100
bst = lgb.train(params, train_data, num_round, valid_sets=[test_data])

# 预测
y_pred = bst.predict(X_test)
y_pred_binary = [1 if x > 0.5 else 0 for x in y_pred]  # 转换为二分类结果

# 评估模型
accuracy = accuracy_score(y_test, y_pred_binary)
report = classification_report(y_test, y_pred_binary)

# 保存模型
bst.save_model('lightgbm_price_prediction_model.txt')

# 保存预测结果
results = {
    'accuracy': accuracy,
    'classification_report': report,
    'feature_importance': bst.feature_importance().tolist(),
    'feature_names': features
}

with open('lightgbm价格预测结果.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

# 打印保存的文件名和成功消息
print("模型已保存到: lightgbm_price_prediction_model.txt")
print("预测结果已保存到: lightgbm价格预测结果.json")
print(f"模型准确率: {accuracy:.4f}")
print("分类报告:")
print(report)