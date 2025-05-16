import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# 读取原始价格数据和特征数据
price_df = pd.read_csv('/Users/ruixue.li/lrx/coze_zc/query-impala-1632023.csv')
features_df = pd.read_csv('/Users/ruixue.li/lrx/coze_zc/商品价格时间序列特征.csv')

# 打印特征信息
print("\n原始特征数据列名：")
print(features_df.columns.tolist())
print(f"\n特征数量：{len(features_df.columns)}")

# 数据预处理
price_df['dt'] = pd.to_datetime(price_df['dt'])
price_df = price_df.sort_values(['sku_id', 'dt'])

# 添加价格相关特征
def add_price_features(df):
    # 按sku_id分组计算特征
    grouped = df.groupby('sku_id')
    
    # 价格变动频率（7天窗口）
    df['price_change_freq_7d'] = grouped['discount_price'].transform(
        lambda x: x.diff().abs().rolling(window=7, min_periods=1).mean()
    )
    
    # 价格变动幅度
    df['price_change_magnitude'] = grouped['discount_price'].transform(
        lambda x: x.diff().abs()
    )
    
    # 价格变动方向（1:上涨, -1:下跌, 0:不变）
    df['price_change_direction'] = grouped['discount_price'].transform(
        lambda x: np.sign(x.diff())
    )
    
    # 价格波动率（7天标准差）
    df['price_volatility_7d'] = grouped['discount_price'].transform(
        lambda x: x.rolling(window=7, min_periods=1).std()
    )
    
    # 移动平均价格（7天）
    df['price_ma7'] = grouped['discount_price'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    
    # 价格分位数（7天窗口）
    df['price_quantile_7d'] = grouped['discount_price'].transform(
        lambda x: x.rolling(window=7, min_periods=1).apply(
            lambda y: pd.Series(y).rank(pct=True).iloc[-1]
        )
    )
    
    return df

# 添加时间特征
def add_time_features(df):
    # 星期几（0-6，0是星期一）
    df['weekday'] = df['dt'].dt.weekday
    
    # 是否周末
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
    
    # 月份
    df['month'] = df['dt'].dt.month
    
    # 季度
    df['quarter'] = df['dt'].dt.quarter
    
    # 是否月初（1-5号）
    df['is_month_start'] = df['dt'].dt.day.isin(range(1, 6)).astype(int)
    
    # 是否月末（25号之后）
    df['is_month_end'] = (df['dt'].dt.day > 25).astype(int)
    
    return df

# 应用特征工程
price_df = add_price_features(price_df)
price_df = add_time_features(price_df)

# 计算价格变动标签
price_df['price_change'] = (price_df.groupby('sku_id')['discount_price'].diff() != 0).astype(int)

# 检查并打印数据框的列名
print("\nprice_df的列名：")
print(price_df.columns.tolist())
print("\nfeatures_df的列名：")
print(features_df.columns.tolist())

# 打印合并前的记录数
print("\n合并前的记录数：")
print(f"price_df记录数: {len(price_df)}")
print(f"features_df记录数: {len(features_df)}")

# 由于features_df没有日期列，我们需要先按sku_id合并
merged_df = pd.merge(
    price_df[['sku_id', 'dt', 'price_change'] + 
        [col for col in price_df.columns if col not in ['sku_id', 'dt', 'price_change', 'discount_price']]],
    features_df,
    on=['sku_id'],
    how='inner'
)

# 打印合并后的记录数
print(f"\n合并后的记录数: {len(merged_df)}")

# 检查是否有数据丢失
if len(merged_df) < min(len(price_df), len(features_df)):
    print("\n警告：合并后数据量减少，可能存在数据不匹配的情况")
    print("建议检查以下情况：")
    print("1. sku_id是否完全匹配")
    print("2. 是否存在缺失值")

# 准备训练数据
X = merged_df.drop(['sku_id', 'dt', 'price_change'], axis=1)
y = merged_df['price_change']

print("\n最终使用的特征：")
print(X.columns.tolist())
print(f"\n最终特征数量：{len(X.columns)}")

# 首先划分出测试集
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
for fold, (train_idx, val_idx) in enumerate(kf.split(X_temp), 1):
    print(f"\n训练第 {fold} 折...")
    
    # 划分训练集和验证集
    X_train, X_val = X_temp.iloc[train_idx], X_temp.iloc[val_idx]
    y_train, y_val = y_temp.iloc[train_idx], y_temp.iloc[val_idx]
    
    # 训练模型
    train_data = lgb.Dataset(X_train, label=y_train)
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
        'feature': X.columns,
        'importance': best_model.feature_importance()
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    for _, row in feature_importance.iterrows():
        f.write(f"{row['feature']}: {row['importance']}\n")

# 保存最佳模型的预测结果
results_df = pd.DataFrame({
    'sku_id': merged_df['sku_id'],
    'dt': merged_df['dt'],
    'price_change': merged_df['price_change'],
    'predicted_prob': best_model.predict(X)
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