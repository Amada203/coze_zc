import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score
from prophet import Prophet
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

# 读取训练集和测试集数据
print("正在加载数据...")
train_df = pd.read_csv('预处理后的商品数据_训练集.csv')
test_df = pd.read_csv('预处理后的商品数据_测试集.csv')

# 转换日期列
train_df['dt'] = pd.to_datetime(train_df['dt'])
test_df['dt'] = pd.to_datetime(test_df['dt'])

# 1. 合并Prophet分解静态特征
prophet_decompose = pd.read_csv('商品价格时间序列特征.csv')
# 只保留静态特征列（不含yhat等预测结果）
static_cols = ['sku_id', 'trend', 'yearly', 'weekly']
prophet_decompose = prophet_decompose[static_cols].drop_duplicates('sku_id')
train_df = pd.merge(train_df, prophet_decompose, on='sku_id', how='left')
test_df = pd.merge(test_df, prophet_decompose, on='sku_id', how='left')

# 2. 生成Prophet预测结果（动态特征）
print("\n正在生成Prophet预测结果...")
prophet_predictions = []
all_df = pd.concat([train_df, test_df], ignore_index=True)
all_df = all_df.sort_values(['sku_id', 'dt'])
# 用tqdm包装groupby循环，显示进度条
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
prophet_pred.to_csv('prophet预测结果.csv', index=False)
print("Prophet预测结果已保存到'prophet预测结果.csv'")

# 3. 合并Prophet预测结果（动态特征）
train_df = pd.merge(train_df, prophet_pred, on=['sku_id', 'dt'], how='left')
test_df = pd.merge(test_df, prophet_pred, on=['sku_id', 'dt'], how='left')
print("\n训练集前两行：")
print(train_df.head(2))
print("\n测试集前两行：")
print(test_df.head(2))


# 4. 生成衍生特征
for df in [train_df, test_df]:
    # 预测价格涨跌幅
    df['yhat_diff'] = df.groupby('sku_id')['yhat'].diff()
    df['yhat_pct_change'] = df.groupby('sku_id')['yhat'].pct_change()
    # 置信区间宽度
    df['yhat_ci_width'] = df['yhat_upper'] - df['yhat_lower']
    # 预测残差
    df['yhat_residual'] = df['discount_price'] - df['yhat']
    # Prophet预测的价格变动（0/1）
    df['prophet_price_change'] = (df.groupby('sku_id')['yhat'].diff() != 0).astype(int)

# 5. 缺失值处理
train_df = train_df.fillna(0)
test_df = test_df.fillna(0)

# 6. 特征选择（去除无关列）
drop_cols = ['sku_id', 'dt', 'price_change']
X_train = train_df.drop(drop_cols, axis=1)
y_train = (train_df['price_change'] > 0).astype(int)  # 强制二值化
drop_cols = ['sku_id', 'dt', 'price_change']
X_test = test_df.drop(drop_cols, axis=1)
y_test = (test_df['price_change'] > 0).astype(int)    # 强制二值化

print("\n最终使用的特征：")
print(X_train.columns.tolist())
print(f"\n最终特征数量：{len(X_train.columns)}")

# 设置K折交叉验证
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
fold_metrics = []
best_model = None
best_auc = 0
params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbose': -1
}
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    print(f"\n训练第 {fold} 折...")
    X_train_fold, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold = (y_train.iloc[train_idx] > 0).astype(int)
    y_val = (y_train.iloc[val_idx] > 0).astype(int)
    print('y_val unique:', np.unique(y_val))  # 检查唯一值
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    callbacks = [lgb.early_stopping(stopping_rounds=10)]
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
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
final_pred = (test_pred + test_df['prophet_price_change']) / 2
final_pred_binary = (final_pred > 0.5).astype(int)
final_accuracy = accuracy_score(y_test, final_pred_binary)
final_auc = roc_auc_score(y_test, final_pred)
with open('模型评估报告.txt', 'w', encoding='utf-8') as f:
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
    f.write(f"融合模型准确率: {final_accuracy:.4f}\n")
    f.write(f"融合模型AUC: {final_auc:.4f}\n")
    f.write("\n特征重要性:\n")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importance()
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    for _, row in feature_importance.iterrows():
        f.write(f"{row['feature']}: {row['importance']}\n")
results_df = pd.DataFrame({
    'sku_id': test_df['sku_id'],
    'dt': test_df['dt'],
    'price_change': test_df['price_change'],
    'lightgbm_pred': test_pred,
    'prophet_pred': test_df['prophet_price_change'],
    'final_pred': final_pred
})
results_df.to_csv('商品价格变动预测结果.csv', index=False)
print("\nK折交叉验证完成！")
print(f"交叉验证平均准确率: {mean_accuracy:.4f} (±{std_accuracy:.4f})")
print(f"交叉验证平均AUC: {mean_auc:.4f} (±{std_auc:.4f})")
print(f"\n独立测试集评估结果:")
print(f"LightGBM模型准确率: {test_accuracy:.4f}")
print(f"LightGBM模型AUC: {test_auc:.4f}")
print(f"融合模型准确率: {final_accuracy:.4f}")
print(f"融合模型AUC: {final_auc:.4f}")
print("\n详细评估结果已保存到'模型评估报告.txt'")
print("预测结果已保存到'商品价格变动预测结果.csv'") 