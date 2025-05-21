import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, average_precision_score, log_loss
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")
import joblib
import json

# 读取全特征训练集和测试集
print("正在加载数据...")
train_df = pd.read_csv('预处理后的商品数据_训练集_全特征_v1.csv')
test_df = pd.read_csv('预处理后的商品数据_测试集_全特征_v1.csv')

# 自动筛选特征列（去除ID、时间、标签等无关列）
exclude_cols = ['sku_id', 'dt', 'label', 'discount_price', 'page_price','is_promotion', 'yhat_residual']
feature_cols = [col for col in train_df.columns if col not in exclude_cols]
yhat_residual_cols = [col for col in feature_cols if 'yhat_residual' in col]
feature_cols = [col for col in feature_cols if col not in yhat_residual_cols]
feature_cols = [col for col in feature_cols if col in test_df.columns]
X_train = train_df[feature_cols]
y_train = train_df['label']
X_test = test_df[feature_cols]
y_test = test_df['label']

print("\n最终使用的特征：")
print(feature_cols)
print(f"\n最终特征数量：{len(feature_cols)}")

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

n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)
fold_metrics = []
best_model = None
best_auc = 0
important_features = None

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
    print(f"\n训练第 {fold} 折...")
    X_train_fold, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold = y_train.iloc[train_idx]
    y_val = y_train.iloc[val_idx]
    scaler = StandardScaler()
    X_train_fold_scaled = pd.DataFrame(scaler.fit_transform(X_train_fold), columns=X_train.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_train.columns)
    if fold == 1:
        init_model = xgb.XGBClassifier(**params, n_estimators=100, use_label_encoder=False)
        init_model.fit(X_train_fold_scaled, y_train_fold)
        importances = init_model.feature_importances_
        important_features = [f for f, imp in zip(X_train.columns, importances) if imp > 0]
        print(f"被保留的重要特征: {important_features}")
    X_train_fold_scaled = X_train_fold_scaled[important_features]
    X_val_scaled = X_val_scaled[[f for f in important_features if f in X_val_scaled.columns]]
    dtrain = xgb.DMatrix(X_train_fold_scaled, label=y_train_fold)
    dval = xgb.DMatrix(X_val_scaled, label=y_val)
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
    precision = precision_score(y_val, y_pred_binary)
    recall = recall_score(y_val, y_pred_binary)
    f1 = f1_score(y_val, y_pred_binary)
    pr_auc = average_precision_score(y_val, y_pred)
    logloss = log_loss(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred_binary)
    fold_metrics.append({
        'fold': fold,
        'accuracy': accuracy,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'pr_auc': pr_auc,
        'logloss': logloss,
        'cm': cm.tolist()
    })
    if auc > best_auc:
        best_auc = auc
        best_model = model
    print(f"第 {fold} 折评估结果:")
    print(f"准确率: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率(重点): {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Logloss: {logloss:.4f}")
    print(f"混淆矩阵: {cm.tolist()}")

mean_accuracy = np.mean([m['accuracy'] for m in fold_metrics])
mean_auc = np.mean([m['auc'] for m in fold_metrics])
std_accuracy = np.std([m['accuracy'] for m in fold_metrics])
std_auc = np.std([m['auc'] for m in fold_metrics])

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train[important_features]), columns=important_features)
X_test_scaled = pd.DataFrame(scaler.transform(X_test[[f for f in important_features if f in X_test.columns]]), columns=[f for f in important_features if f in X_test.columns])
dtrain_full = xgb.DMatrix(X_train_scaled, label=y_train)
dtest = xgb.DMatrix(X_test_scaled, label=y_test)
model_full = xgb.train(
    params,
    dtrain_full,
    num_boost_round=1000,
    evals=[(dtrain_full, 'train'), (dtest, 'eval')],
    early_stopping_rounds=50,
    verbose_eval=False
)
y_pred_test = model_full.predict(dtest)
y_pred_test_binary = (y_pred_test > 0.5).astype(int)
test_accuracy = accuracy_score(y_test, y_pred_test_binary)
test_auc = roc_auc_score(y_test, y_pred_test)
precision_test = precision_score(y_test, y_pred_test_binary)
recall_test = recall_score(y_test, y_pred_test_binary)
f1_test = f1_score(y_test, y_pred_test_binary)
pr_auc_test = average_precision_score(y_test, y_pred_test)
logloss_test = log_loss(y_test, y_pred_test)
cm_test = confusion_matrix(y_test, y_pred_test_binary)

with open('模型评估报告_全特征版_xgboost_v1.txt', 'w', encoding='utf-8') as f:
    f.write("K折交叉验证评估结果:\n")
    f.write(f"平均准确率: {mean_accuracy:.4f} (±{std_accuracy:.4f})\n")
    f.write(f"平均AUC: {mean_auc:.4f} (±{std_auc:.4f})\n\n")
    f.write("各折详细结果:\n")
    for metric in fold_metrics:
        f.write(f"第 {metric['fold']} 折:\n")
        f.write(f"  准确率: {metric['accuracy']:.4f}\n")
        f.write(f"  AUC: {metric['auc']:.4f}\n")
        f.write(f"  精确率: {metric['precision']:.4f}\n")
        f.write(f"  召回率(重点): {metric['recall']:.4f}\n")
        f.write(f"  F1分数: {metric['f1']:.4f}\n")
        f.write(f"  PR-AUC: {metric['pr_auc']:.4f}\n")
        f.write(f"  Logloss: {metric['logloss']:.4f}\n")
        f.write(f"  混淆矩阵: {metric['cm']}\n")
    f.write("\n独立测试集评估结果:\n")
    f.write(f"XGBoost模型准确率: {test_accuracy:.4f}\n")
    f.write(f"XGBoost模型AUC: {test_auc:.4f}\n")
    f.write(f"XGBoost模型精确率: {precision_test:.4f}\n")
    f.write(f"XGBoost模型召回率(重点): {recall_test:.4f}\n")
    f.write(f"XGBoost模型F1分数: {f1_test:.4f}\n")
    f.write(f"XGBoost模型PR-AUC: {pr_auc_test:.4f}\n")
    f.write(f"XGBoost模型Logloss: {logloss_test:.4f}\n")
    f.write(f"XGBoost模型混淆矩阵: {cm_test.tolist()}\n")
    f.write("\n特征重要性:\n")
    feature_map = {f"f{i}": col for i, col in enumerate(X_train_scaled.columns)}
    score_dict = model_full.get_score(importance_type='weight')
    feature_importance = pd.DataFrame([
        {'feature': feature_map.get(f, f), 'importance': score}
        for f, score in score_dict.items()
    ])
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    for _, row in feature_importance.iterrows():
        f.write(f"{row['feature']}: {row['importance']}\n")

results_df = pd.DataFrame({
    'sku_id': test_df['sku_id'],
    'dt': test_df['dt'],
    'discount_price': test_df['discount_price'],
    'discount_price_lag_1': test_df['discount_price_lag_1'] if 'discount_price_lag_1' in test_df.columns else np.nan,
    'label_true': y_test,
    'xgboost_pred': y_pred_test
})
results_df.to_csv('discount_price变动预测结果_全特征版_xgboost_v1.csv', index=False)

print("\nK折交叉验证完成！", flush=True)
print(f"交叉验证平均准确率: {mean_accuracy:.4f} (±{std_accuracy:.4f})", flush=True)
print(f"交叉验证平均AUC: {mean_auc:.4f} (±{std_auc:.4f})", flush=True)
print(f"\n独立测试集评估结果:", flush=True)
print(f"XGBoost模型准确率: {test_accuracy:.4f}", flush=True)
print(f"XGBoost模型AUC: {test_auc:.4f}", flush=True)
print(f"XGBoost模型精确率: {precision_test:.4f}", flush=True)
print(f"XGBoost模型召回率(重点): {recall_test:.4f}", flush=True)
print(f"XGBoost模型F1分数: {f1_test:.4f}", flush=True)
print(f"XGBoost模型PR-AUC: {pr_auc_test:.4f}", flush=True)
print(f"XGBoost模型Logloss: {logloss_test:.4f}", flush=True)
print(f"XGBoost模型混淆矩阵: {cm_test.tolist()}", flush=True)
print("\n详细评估结果已保存到'模型评估报告_全特征版_xgboost_v1.txt'", flush=True)
print("预测结果已保存到'discount_price变动预测结果_全特征版_xgboost_v1.csv'", flush=True)

# 保存最佳模型
joblib.dump(model_full, 'xgboost_price_change_model_v1.pkl')
print("模型已保存为 xgboost_price_change_model_v1.pkl")

# 保存特征名
with open('feature_cols_v1.json', 'w') as f:
    json.dump(important_features, f)
print("特征名已保存为 feature_cols_v1.json") 