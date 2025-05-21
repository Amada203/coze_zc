# 导入必要的库
import pandas as pd
import numpy as np
import lightgbm_test as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import json

# 读取时间序列特征数据
with open('价格数据时间序列特征.json', 'r', encoding='utf-8') as f:
    time_series_features = json.load(f)

# 读取原始价格数据
price_data = pd.read_csv('价格数据缺失值填充处理结果.csv')

# 合并特征数据
merged_data = pd.merge(price_data, pd.DataFrame(time_series_features), on=['sku_id', 'date'])

# 准备特征和目标变量
X = merged_data.drop(['price_change'], axis=1)
y = merged_data['price_change']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练LightGBM模型
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

model = lgb.LGBMClassifier(**params)
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# 评估模型性能
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# 保存模型
model.save_model('lightgbm_price_prediction_model.txt')

# 保存预测结果
predictions = pd.DataFrame({
    'sku_id': X_test['sku_id'],
    'date': X_test['date'],
    'y_true': y_test,
    'y_pred': y_pred,
    'y_pred_proba': y_pred_proba
})
predictions.to_json('lightgbm价格预测结果.json', orient='records', force_ascii=False)

# 生成结果报告
result_content = f"""模型已保存到: lightgbm_price_prediction_model.txt
预测结果已保存到: lightgbm价格预测结果.json
模型准确率: {accuracy:.4f}
分类报告:
{report}
"""

with open('商品价格变动概率预测结果.txt', 'w', encoding='utf-8') as f:
    f.write(result_content)

print("模型训练和预测已完成，结果已保存。")