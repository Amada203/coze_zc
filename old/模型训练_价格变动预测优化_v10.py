import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

# 读取训练集和测试集数据
print("正在加载数据...")
train_df = pd.read_csv('预处理后的商品数据_训练集.csv')
test_df = pd.read_csv('预处理后的商品数据_测试集.csv')

# 转换日期列
train_df['dt'] = pd.to_datetime(train_df['dt'])
test_df['dt'] = pd.to_datetime(test_df['dt'])

# 只保留discount_price及其历史衍生特征
# 以sku_id为分组，生成滞后、均值、方差、趋势等特征
for df in [train_df, test_df]:
    df['discount_price_lag_1'] = df.groupby('sku_id')['discount_price'].shift(1)
    df['discount_price_lag_3'] = df.groupby('sku_id')['discount_price'].shift(3)
    df['discount_price_lag_7'] = df.groupby('sku_id')['discount_price'].shift(7)
    df['discount_price_mean_7'] = df.groupby('sku_id')['discount_price'].shift(1).rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
    df['discount_price_std_7'] = df.groupby('sku_id')['discount_price'].shift(1).rolling(7, min_periods=1).std().reset_index(level=0, drop=True)
    df['discount_price_min_7'] = df.groupby('sku_id')['discount_price'].shift(1).rolling(7, min_periods=1).min().reset_index(level=0, drop=True)
    df['discount_price_max_7'] = df.groupby('sku_id')['discount_price'].shift(1).rolling(7, min_periods=1).max().reset_index(level=0, drop=True)
    df['discount_price_trend_7'] = df.groupby('sku_id')['discount_price'].shift(1).rolling(7, min_periods=2).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=False).reset_index(level=0, drop=True)

# 标签：T日discount_price是否变动（与T-1日不同即为变动）
train_df['label'] = (train_df['discount_price'] != train_df['discount_price_lag_1']).astype(int)
test_df['label'] = (test_df['discount_price'] != test_df['discount_price_lag_1']).astype(int)

# 特征列
feature_cols = [
    'discount_price_lag_1', 'discount_price_lag_3', 'discount_price_lag_7',
    'discount_price_mean_7', 'discount_price_std_7', 'discount_price_min_7',
    'discount_price_max_7', 'discount_price_trend_7'
]

# 去除首行等因滞后导致的NaN
train_df = train_df.dropna(subset=feature_cols + ['label'])
test_df = test_df.dropna(subset=feature_cols + ['label'])

X_train = train_df[feature_cols]
y_train = train_df['label']
X_test = test_df[feature_cols]
y_test = test_df['label']

# 标签分布分析
print("\n【标签分布分析】")
print("训练集变动/不变动比例：")
print(y_train.value_counts(normalize=True))
print("测试集变动/不变动比例：")
print(y_test.value_counts(normalize=True))

# 特征与标签相关性分析
print("\n【特征与标签相关性分析】")
correlations = train_df[feature_cols + ['label']].corr()['label'].sort_values(ascending=False)
print(correlations)
try:
    correlations.drop('label').abs().sort_values(ascending=False).head(8).plot(kind='bar')
    plt.title('Top 特征与标签的相关性')
    plt.show()
except Exception as e:
    print("相关性可视化失败：", e)

# 特征归一化
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# 样本均衡参数
pos = np.sum(y_train == 1)
neg = np.sum(y_train == 0)
scale_pos_weight = neg / pos if pos > 0 else 1

# XGBoost模型参数
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 0.01,
    'max_depth': 8,
    'min_child_weight': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': scale_pos_weight,
    'verbosity': 0,
    'nthread': 4,
    'seed': 42
}

print("\n初步训练XGBoost以筛选特征...")
init_model = xgb.XGBClassifier(**params, n_estimators=100)
init_model.fit(X_train, y_train)
importances = init_model.feature_importances_
important_features = [f for f, imp in zip(X_train.columns, importances) if imp > 0]
print(f"被保留的重要特征: {important_features}")

# 只保留重要特征
X_train = X_train[important_features]
X_test = X_test[important_features]

print("\n最终使用的特征：")
print(X_train.columns.tolist())
print(f"\n最终特征数量：{len(X_train.columns)}")

# 交叉验证
n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
fold_metrics = []
best_model = None
best_auc = 0

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train), 1):
    print(f"\n训练第 {fold} 折...")
    X_train_fold, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold = y_train.iloc[train_idx]
    y_val = y_train.iloc[val_idx]
    
    dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
    dval = xgb.DMatrix(X_val, label=y_val)
    evals = [(dtrain, 'train'), (dval, 'eval')]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=False
    )
    y_pred = model.predict(dval)
    y_pred_binary = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_val, y_pred_binary)
    auc = roc_auc_score(y_val, y_pred)
    fold_metrics.append({
        'fold': fold,
        'accuracy': accuracy,
        'auc': auc
    })
    if auc > best_auc:
        best_auc = auc
        best_model = model
    print(f"第 {fold} 折评估结果:")
    print(f"准确率: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")

mean_accuracy = np.mean([m['accuracy'] for m in fold_metrics])
mean_auc = np.mean([m['auc'] for m in fold_metrics])
std_accuracy = np.std([m['accuracy'] for m in fold_metrics])
std_auc = np.std([m['auc'] for m in fold_metrics])

dtest = xgb.DMatrix(X_test)
test_pred = best_model.predict(dtest)
test_pred_binary = (test_pred > 0.5).astype(int)
test_accuracy = accuracy_score(y_test, test_pred_binary)
test_auc = roc_auc_score(y_test, test_pred)

# 保存评估结果
with open('模型评估报告_v10.txt', 'w', encoding='utf-8') as f:
    f.write("K折交叉验证评估结果:\n")
    f.write(f"平均准确率: {mean_accuracy:.4f} (±{std_accuracy:.4f})\n")
    f.write(f"平均AUC: {mean_auc:.4f} (±{std_auc:.4f})\n\n")
    f.write("各折详细结果:\n")
    for metric in fold_metrics:
        f.write(f"第 {metric['fold']} 折:\n")
        f.write(f"  准确率: {metric['accuracy']:.4f}\n")
        f.write(f"  AUC: {metric['auc']:.4f}\n")
    f.write("\n独立测试集评估结果:\n")
    f.write(f"XGBoost模型准确率: {test_accuracy:.4f}\n")
    f.write(f"XGBoost模型AUC: {test_auc:.4f}\n")
    f.write("\n特征重要性:\n")
    feature_map = {f"f{i}": col for i, col in enumerate(X_train.columns)}
    score_dict = best_model.get_score(importance_type='weight')
    feature_importance = pd.DataFrame([
        {'feature': feature_map.get(f, f), 'importance': score}
        for f, score in score_dict.items()
    ])
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    for _, row in feature_importance.iterrows():
        f.write(f"{row['feature']}: {row['importance']}\n")

# 保存预测结果
results_df = pd.DataFrame({
    'sku_id': test_df['sku_id'],
    'dt': test_df['dt'],
    'discount_price': test_df['discount_price'],
    'discount_price_lag_1': test_df['discount_price_lag_1'],
    'label_true': y_test,
    'xgboost_pred': test_pred
})
results_df.to_csv('discount_price变动预测结果_v10.csv', index=False)

print("\nK折交叉验证完成！", flush=True)
print(f"交叉验证平均准确率: {mean_accuracy:.4f} (±{std_accuracy:.4f})", flush=True)
print(f"交叉验证平均AUC: {mean_auc:.4f} (±{std_auc:.4f})", flush=True)
print(f"\n独立测试集评估结果:", flush=True)
print(f"XGBoost模型准确率: {test_accuracy:.4f}", flush=True)
print(f"XGBoost模型AUC: {test_auc:.4f}", flush=True)
print("\n详细评估结果已保存到'模型评估报告_v10.txt'", flush=True)
print("预测结果已保存到'discount_price变动预测结果_v10.csv'", flush=True) 