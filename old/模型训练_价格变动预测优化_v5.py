import pandas as pd
import numpy as np
import lightgbm_test as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from prophet import Prophet
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# 读取训练集和测试集数据
print("正在加载数据...")
train_df = pd.read_csv('预处理后的商品数据_训练集.csv')
test_df = pd.read_csv('预处理后的商品数据_测试集.csv')

# 转换日期列
train_df['dt'] = pd.to_datetime(train_df['dt'])
test_df['dt'] = pd.to_datetime(test_df['dt'])

# 1. 合并Prophet分解静态特征
prophet_decompose = pd.read_csv('商品价格时间序列特征.csv')
static_cols = ['sku_id', 'trend', 'yearly', 'weekly']
prophet_decompose = prophet_decompose[static_cols].drop_duplicates('sku_id')
train_df = pd.merge(train_df, prophet_decompose, on='sku_id', how='left')
test_df = pd.merge(test_df, prophet_decompose, on='sku_id', how='left')

# 2. 生成Prophet预测结果（动态特征）
print("\n正在生成Prophet预测结果...")
prophet_predictions = []
all_df = pd.concat([train_df, test_df], ignore_index=True)
all_df = all_df.sort_values(['sku_id', 'dt'])
for sku_id, group in tqdm(all_df.groupby('sku_id'), total=all_df['sku_id'].nunique(), desc='Prophet建模'):
    prophet_df = group[['dt', 'discount_price']].rename(columns={'dt': 'ds', 'discount_price': 'y'})
    model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=True)
    model.fit(prophet_df)
    future = prophet_df[['ds']]
    forecast = model.predict(future)
    forecast['sku_id'] = sku_id
    forecast['dt'] = forecast['ds']
    forecast = forecast[['sku_id', 'dt', 'yhat', 'yhat_lower', 'yhat_upper']]
    prophet_predictions.append(forecast)
prophet_pred = pd.concat(prophet_predictions, ignore_index=True)
prophet_pred.to_csv('prophet预测结果_v4.csv', index=False)  # 更新文件名
print("Prophet预测结果已保存到'prophet预测结果_v4.csv'")

# 3. 合并Prophet预测结果（动态特征）
train_df = pd.merge(train_df, prophet_pred, on=['sku_id', 'dt'], how='left')
test_df = pd.merge(test_df, prophet_pred, on=['sku_id', 'dt'], how='left')

# 4. 生成衍生特征（只保留yhat_diff和yhat_residual）
# 【可疑点1】yhat_residual = discount_price - yhat，其中yhat是用全量数据Prophet预测的结果，可能包含未来信息，导致数据泄漏。
# 建议：Prophet建模和预测应仅用训练集历史数据，不能用未来（测试集）数据。
for df in [train_df, test_df]:
    df['yhat_diff'] = df.groupby('sku_id')['yhat'].diff()
    df['yhat_residual'] = df['discount_price'] - df['yhat']

# 5. 缺失值处理
train_df = train_df.fillna(0)
test_df = test_df.fillna(0)

# 6. 特征选择（去除无关列，仅保留优化后的特征）
drop_cols = ['sku_id', 'dt', 'price_change', 'yhat', 'yhat_upper', 'yhat_lower', 'prophet_price_change', 'yhat_pct_change', 'yhat_ci_width']
# 【可疑点2】虽然drop_cols中去除了price_change，但后续标签和结果中依然用到了price_change，需警惕其衍生特征是否泄漏。
X_train = train_df.drop([col for col in drop_cols if col in train_df.columns], axis=1)
y_train = (train_df['price_change'] != 0).astype(int)  # 修改标签定义
X_test = test_df.drop([col for col in drop_cols if col in test_df.columns], axis=1)
y_test = (test_df['price_change'] != 0).astype(int)  # 修改标签定义

# 7. 特征归一化
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# 8. 样本均衡参数
pos = np.sum(y_train == 1)
neg = np.sum(y_train == 0)
scale_pos_weight = neg / pos if pos > 0 else 1

# 9. 优化后的模型参数
params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.01,  # 降低学习率，使模型更稳定
    'num_leaves': 63,  # 增加叶子节点数，提高模型复杂度
    'max_depth': 8,  # 控制树的深度，防止过拟合
    'min_data_in_leaf': 20,  # 每个叶子节点最少样本数
    'feature_fraction': 0.8,  # 每次迭代随机选择80%的特征
    'bagging_fraction': 0.8,  # 每次迭代随机选择80%的数据
    'bagging_freq': 5,  # 每5次迭代执行一次bagging
    'verbose': -1,
    'scale_pos_weight': scale_pos_weight
}

print("\n初步训练LightGBM以筛选特征...")
init_model = lgb.LGBMClassifier(**params)
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

# 10. 使用分层K折交叉验证
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
    
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # 增加早停轮数和最大迭代次数
    callbacks = [lgb.early_stopping(stopping_rounds=50)]
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,  # 增加最大迭代次数
        valid_sets=[train_data, val_data],
        callbacks=callbacks
    )
    
    y_pred = model.predict(X_val)
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

test_pred = best_model.predict(X_test)
test_pred_binary = (test_pred > 0.5).astype(int)
test_accuracy = accuracy_score(y_test, test_pred_binary)
test_auc = roc_auc_score(y_test, test_pred)

# 保存评估结果
with open('模型评估报告_v4.txt', 'w', encoding='utf-8') as f:  # 更新文件名
    f.write("K折交叉验证评估结果:\n")
    f.write(f"平均准确率: {mean_accuracy:.4f} (±{std_accuracy:.4f})\n")
    f.write(f"平均AUC: {mean_auc:.4f} (±{std_auc:.4f})\n\n")
    f.write("各折详细结果:\n")
    for metric in fold_metrics:
        f.write(f"第 {metric['fold']} 折:\n")
        f.write(f"  准确率: {metric['accuracy']:.4f}\n")
        f.write(f"  AUC: {metric['auc']:.4f}\n")
    f.write("\n独立测试集评估结果:\n")
    f.write(f"LightGBM模型准确率: {test_accuracy:.4f}\n")
    f.write(f"LightGBM模型AUC: {test_auc:.4f}\n")
    f.write("\n特征重要性:\n")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importance()
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    for _, row in feature_importance.iterrows():
        f.write(f"{row['feature']}: {row['importance']}\n")

# 保存预测结果
results_df = pd.DataFrame({
    'sku_id': test_df['sku_id'],
    'dt': test_df['dt'],
    'price_change': test_df['price_change'],  # 原始价格差值
    'price_change_label': (test_df['price_change'] != 0).astype(int),  # 0/1标签
    'lightgbm_pred': test_pred
})
# 【可疑点3】预测结果中直接输出了price_change和price_change_label，若下游用此文件做进一步建模，需警惕标签泄漏。
results_df.to_csv('商品价格变动预测结果_v4.csv', index=False)  # 更新文件名

print("\nK折交叉验证完成！")
print(f"交叉验证平均准确率: {mean_accuracy:.4f} (±{std_accuracy:.4f})")
print(f"交叉验证平均AUC: {mean_auc:.4f} (±{std_auc:.4f})")
print(f"\n独立测试集评估结果:")
print(f"LightGBM模型准确率: {test_accuracy:.4f}")
print(f"LightGBM模型AUC: {test_auc:.4f}")
print("\n详细评估结果已保存到'模型评估报告_v4.txt'")
print("预测结果已保存到'商品价格变动预测结果_v4.csv'") 