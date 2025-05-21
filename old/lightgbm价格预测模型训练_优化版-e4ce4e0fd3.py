import json
import lightgbm_test as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

# 1. 加载时间序列特征数据
try:
    with open('价格数据时间序列特征.json', 'r', encoding='utf-8') as f:
        time_series_features = json.load(f)
    print("成功加载时间序列特征数据")
except FileNotFoundError:
    print("错误：未找到文件 '价格数据时间序列特征.json'")
    exit()

# 2. 加载价格数据缺失值填充处理结果
try:
    price_data = pd.read_csv('价格数据缺失值填充处理结果.csv')
    print("成功加载价格数据")
except FileNotFoundError:
    print("错误：未找到文件 '价格数据缺失值填充处理结果.csv'")
    exit()

# 3. 数据预处理
# 合并特征数据
features_df = pd.DataFrame(time_series_features)
data = pd.merge(price_data, features_df, on=['sku_id', 'date'])

# 划分训练集和测试集
X = data.drop(['price_change'], axis=1)
y = data['price_change']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 训练LightGBM模型
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

model = lgb.LGBMClassifier(**params)
model.fit(X_train, y_train)

# 5. 模型预测
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 6. 模型评估
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# 7. 保存模型和结果
# 保存模型
model.save_model('优化后的lightgbm价格预测模型.txt')

# 保存预测结果
predictions = {
    'y_test': y_test.tolist(),
    'y_pred': y_pred.tolist(),
    'y_prob': y_prob.tolist()
}
with open('优化后的lightgbm价格预测结果.json', 'w', encoding='utf-8') as f:
    json.dump(predictions, f, ensure_ascii=False, indent=4)

# 保存评估结果
with open('优化后的lightgbm模型评估报告.txt', 'w', encoding='utf-8') as f:
    f.write(f"模型准确率: {accuracy:.4f}\n\n")
    f.write("分类报告:\n")
    f.write(report)
    f.write("\n\n混淆矩阵:\n")
    f.write(str(conf_matrix))

print("模型训练和评估完成，结果已保存")
print(f"- 优化后的lightgbm价格预测模型.txt")
print(f"- 优化后的lightgbm价格预测结果.json")
print(f"- 优化后的lightgbm模型评估报告.txt")