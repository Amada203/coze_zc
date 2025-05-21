import pandas as pd
import numpy as np
import lightgbm_test as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# 读取训练集和测试集数据
print("正在加载数据...")
train_df = pd.read_csv('预处理后的商品数据_训练集.csv')
test_df = pd.read_csv('预处理后的商品数据_测试集.csv')

# 转换日期列
train_df['dt'] = pd.to_datetime(train_df['dt'])
test_df['dt'] = pd.to_datetime(test_df['dt'])

# 准备训练数据
X_train = train_df.drop(['sku_id', 'dt', 'price_change'], axis=1)
y_train = train_df['price_change']
X_test = test_df.drop(['sku_id', 'dt', 'price_change'], axis=1)
y_test = test_df['price_change']

print("\n最终使用的特征：")
print(X_train.columns.tolist())
print(f"\n最终特征数量：{len(X_train.columns)}")

# 设置K折交叉验证
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# 存储每折的评估结果
fold_metrics = []
best_model = None
best_auc = 0

# LightGBM参数
params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbose': -1
}

# 进行K折交叉验证
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    print(f"\n训练第 {fold} 折...")
    
    # 划分训练集和验证集
    X_train_fold, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # 训练模型
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # 设置早停回调
    callbacks = [lgb.early_stopping(stopping_rounds=10)]
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[train_data, val_data],
        callbacks=callbacks
    )
    
    # 预测和评估
    y_pred = model.predict(X_val)
    y_pred_binary = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_val, y_pred_binary)
    auc = roc_auc_score(y_val, y_pred)
    
    fold_metrics.append({
        'fold': fold,
        'accuracy': accuracy,
        'auc': auc
    })
    
    # 保存最佳模型
    if auc > best_auc:
        best_auc = auc
        best_model = model
    
    print(f"第 {fold} 折评估结果:")
    print(f"准确率: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")

# 计算交叉验证的平均评估指标
mean_accuracy = np.mean([m['accuracy'] for m in fold_metrics])
mean_auc = np.mean([m['auc'] for m in fold_metrics])
std_accuracy = np.std([m['accuracy'] for m in fold_metrics])
std_auc = np.std([m['auc'] for m in fold_metrics])

# 在测试集上评估最佳模型
test_pred = best_model.predict(X_test)
test_pred_binary = (test_pred > 0.5).astype(int)
test_accuracy = accuracy_score(y_test, test_pred_binary)
test_auc = roc_auc_score(y_test, test_pred)

# 保存模型评估报告
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
    f.write(f"准确率: {test_accuracy:.4f}\n")
    f.write(f"AUC: {test_auc:.4f}\n")
    
    # 添加特征重要性信息
    f.write("\n特征重要性:\n")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importance()
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    for _, row in feature_importance.iterrows():
        f.write(f"{row['feature']}: {row['importance']}\n")

# 保存最佳模型的预测结果
results_df = pd.DataFrame({
    'sku_id': test_df['sku_id'],
    'dt': test_df['dt'],
    'price_change': test_df['price_change'],
    'predicted_prob': test_pred
})
results_df.to_csv('商品价格变动预测结果.csv', index=False)

print("\nK折交叉验证完成！")
print(f"交叉验证平均准确率: {mean_accuracy:.4f} (±{std_accuracy:.4f})")
print(f"交叉验证平均AUC: {mean_auc:.4f} (±{std_auc:.4f})")
print(f"\n独立测试集评估结果:")
print(f"准确率: {test_accuracy:.4f}")
print(f"AUC: {test_auc:.4f}")
print("\n详细评估结果已保存到'模型评估报告.txt'")
print("预测结果已保存到'商品价格变动预测结果.csv'")