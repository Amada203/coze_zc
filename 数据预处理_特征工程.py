import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """加载原始数据"""
    print("正在加载数据...")
    df = pd.read_csv('/Users/ruixue.li/lrx/coze_zc/query-impala-1632023.csv')
    df['dt'] = pd.to_datetime(df['dt'])
    
    # 处理价格缺失值
    print("正在处理价格缺失值...")
    # 按sku_id分组，对每个sku的价格进行前向和后向填充
    df = df.sort_values(['sku_id', 'dt'])
    df['discount_price'] = df.groupby('sku_id')['discount_price'].transform(
        lambda x: x.fillna(method='ffill').fillna(method='bfill')
    )
    
    # 检查是否还有缺失值
    missing_count = df['discount_price'].isnull().sum()
    if missing_count > 0:
        print(f"警告：仍有 {missing_count} 条记录的价格为缺失值")
        print("这些记录将被删除")
        df = df.dropna(subset=['discount_price'])
    
    return df

def add_time_features(df):
    """添加时间相关特征"""
    print("正在添加时间特征...")
    
    # 基础时间特征
    df['year'] = df['dt'].dt.year
    df['month'] = df['dt'].dt.month
    df['day'] = df['dt'].dt.day
    df['weekday'] = df['dt'].dt.weekday
    df['quarter'] = df['dt'].dt.quarter
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
    
    # 特殊时间点
    df['is_month_start'] = df['dt'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['dt'].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df['dt'].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df['dt'].dt.is_quarter_end.astype(int)
    
    # 电商大促日期
    promotion_dates = {
        # 双11
        'double_11': [
            '2023-11-01', '2023-11-02', '2023-11-03', '2023-11-04', '2023-11-05',
            '2023-11-06', '2023-11-07', '2023-11-08', '2023-11-09', '2023-11-10',
            '2023-11-11', '2023-11-12'
        ],
        # 双12
        'double_12': [
            '2023-12-01', '2023-12-02', '2023-12-03', '2023-12-04', '2023-12-05',
            '2023-12-06', '2023-12-07', '2023-12-08', '2023-12-09', '2023-12-10',
            '2023-12-11', '2023-12-12'
        ],
        # 618
        '618': [
            '2023-06-01', '2023-06-02', '2023-06-03', '2023-06-04', '2023-06-05',
            '2023-06-06', '2023-06-07', '2023-06-08', '2023-06-09', '2023-06-10',
            '2023-06-11', '2023-06-12', '2023-06-13', '2023-06-14', '2023-06-15',
            '2023-06-16', '2023-06-17', '2023-06-18'
        ],
        # 年货节
        'spring_festival': [
            '2023-01-10', '2023-01-11', '2023-01-12', '2023-01-13', '2023-01-14',
            '2023-01-15', '2023-01-16', '2023-01-17', '2023-01-18', '2023-01-19',
            '2023-01-20', '2023-01-21', '2023-01-22', '2023-01-23', '2023-01-24',
            '2023-01-25', '2023-01-26', '2023-01-27'
        ],
        # 38女神节
        'womens_day': [
            '2023-03-01', '2023-03-02', '2023-03-03', '2023-03-04', '2023-03-05',
            '2023-03-06', '2023-03-07', '2023-03-08'
        ],
        # 五一
        'labor_day': [
            '2023-04-28', '2023-04-29', '2023-04-30', '2023-05-01', '2023-05-02',
            '2023-05-03', '2023-05-04', '2023-05-05'
        ],
        # 国庆
        'national_day': [
            '2023-09-28', '2023-09-29', '2023-09-30', '2023-10-01', '2023-10-02',
            '2023-10-03', '2023-10-04', '2023-10-05', '2023-10-06', '2023-10-07'
        ]
    }
    
    # 添加大促标记
    df['is_promotion_period'] = 0
    for promo_name, dates in promotion_dates.items():
        df[f'is_{promo_name}'] = df['dt'].dt.strftime('%Y-%m-%d').isin(dates).astype(int)
        df['is_promotion_period'] = df['is_promotion_period'] | df[f'is_{promo_name}']
    
    # 添加大促前后标记（大促前7天和后7天）
    df['is_before_promotion'] = 0
    df['is_after_promotion'] = 0
    
    for promo_name, dates in promotion_dates.items():
        # 获取每个大促的开始和结束日期
        dates = pd.to_datetime(dates)
        start_date = dates.min()
        end_date = dates.max()
        
        # 标记大促前7天
        before_start = start_date - pd.Timedelta(days=7)
        mask_before = (df['dt'] >= before_start) & (df['dt'] < start_date)
        df.loc[mask_before, 'is_before_promotion'] = 1
        
        # 标记大促后7天
        after_end = end_date + pd.Timedelta(days=7)
        mask_after = (df['dt'] > end_date) & (df['dt'] <= after_end)
        df.loc[mask_after, 'is_after_promotion'] = 1
    
    return df

def add_price_features(df):
    """添加价格相关特征"""
    print("正在添加价格特征...")
    
    # 按sku_id分组计算特征
    grouped = df.groupby('sku_id')
    
    # 价格变动特征 - 使用滞后值避免数据泄露
    df['price_change'] = grouped['discount_price'].diff()
    df['price_change_pct'] = grouped['discount_price'].pct_change()
    df['price_change_abs'] = df['price_change'].abs()
    
    # 价格统计特征（7天窗口）- 使用shift避免使用未来数据
    df['price_mean_7d'] = grouped['discount_price'].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).mean()
    )
    df['price_std_7d'] = grouped['discount_price'].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).std()
    )
    df['price_min_7d'] = grouped['discount_price'].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).min()
    )
    df['price_max_7d'] = grouped['discount_price'].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).max()
    )
    
    # 价格分位数特征 - 使用shift避免使用未来数据
    df['price_quantile_7d'] = grouped['discount_price'].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).apply(
            lambda y: pd.Series(y).rank(pct=True).iloc[-1]
        )
    )
    
    # 价格趋势特征 - 使用shift避免使用未来数据
    def safe_trend(x):
        try:
            if len(x) < 2:
                return 0
            return np.mean(np.diff(x))
        except:
            return 0
    
    df['price_trend_7d'] = grouped['discount_price'].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).apply(safe_trend)
    )
    
    return df

def add_promotion_features(df):
    """添加促销相关特征"""
    print("正在添加促销特征...")
    
    # 按sku_id分组计算特征
    grouped = df.groupby('sku_id')
    
    # 促销频率特征 - 使用shift避免使用未来数据
    df['promo_freq_7d'] = grouped['is_promotion'].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).mean()
    )
    df['promo_freq_30d'] = grouped['is_promotion'].transform(
        lambda x: x.shift(1).rolling(window=30, min_periods=1).mean()
    )
    
    # 促销持续时间 - 使用shift避免使用未来数据
    df['promo_duration'] = grouped['is_promotion'].transform(
        lambda x: x.shift(1).groupby((x.shift(1) != x.shift(1).shift()).cumsum()).cumsum()
    )
    
    # 促销间隔 - 使用shift避免使用未来数据
    df['days_since_last_promo'] = grouped['is_promotion'].transform(
        lambda x: x.shift(1).groupby((x.shift(1) == 1).cumsum()).cumcount()
    )
    
    # 促销效果特征 - 使用shift避免使用未来数据
    df['price_during_promo'] = df['discount_price'].shift(1) * df['is_promotion'].shift(1)
    df['price_outside_promo'] = df['discount_price'].shift(1) * (1 - df['is_promotion'].shift(1))
    
    # 促销价格变化 - 使用shift避免使用未来数据
    df['promo_price_change'] = df['price_during_promo'].diff()
    df['promo_price_change_pct'] = df['price_during_promo'].pct_change()
    
    return df

def add_interaction_features(df):
    """添加特征交互项"""
    print("正在添加特征交互项...")
    
    # 价格与促销的交互
    df['price_promo_interaction'] = df['discount_price'] * df['is_promotion']
    
    # 时间与促销的交互
    df['weekday_promo_interaction'] = df['weekday'] * df['is_promotion']
    df['month_promo_interaction'] = df['month'] * df['is_promotion']
    
    # 价格趋势与促销的交互
    df['price_trend_promo_interaction'] = df['price_trend_7d'] * df['is_promotion']
    
    return df

def add_lag_features(df):
    """添加滞后特征"""
    print("正在添加滞后特征...")
    
    # 按sku_id分组计算滞后特征
    grouped = df.groupby('sku_id')
    
    # 价格滞后特征
    for lag in [1, 3, 7, 14, 30]:
        df[f'price_lag_{lag}d'] = grouped['discount_price'].shift(lag)
        df[f'price_change_lag_{lag}d'] = grouped['price_change'].shift(lag)
        df[f'is_promotion_lag_{lag}d'] = grouped['is_promotion'].shift(lag)
    
    return df

def main():
    # 加载数据
    df = load_data()
    
    # 添加各类特征
    df = add_time_features(df)
    df = add_price_features(df)
    df = add_promotion_features(df)
    df = add_interaction_features(df)
    df = add_lag_features(df)
    
    # 删除包含NaN的行
    df = df.dropna()
    
    # 按时间分割数据
    print("\n正在分割训练集和测试集...")
    df = df.sort_values(['sku_id', 'dt'])
    split_date = pd.to_datetime('2024-12-31')
    
    # 分割数据集
    train_df = df[df['dt'] < split_date]
    test_df = df[df['dt'] >= split_date]
    
    # 保存处理后的数据
    train_file = '预处理后的商品数据_训练集.csv'
    test_file = '预处理后的商品数据_测试集.csv'
    
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"\n数据预处理完成！")
    print(f"训练集数据已保存到：{train_file}")
    print(f"测试集数据已保存到：{test_file}")
    print(f"\n数据集统计信息：")
    print(f"训练集样本数：{len(train_df)}")
    print(f"测试集样本数：{len(test_df)}")
    print(f"训练集商品数：{train_df['sku_id'].nunique()}")
    print(f"测试集商品数：{test_df['sku_id'].nunique()}")
    print(f"训练集时间范围：{train_df['dt'].min()} 至 {train_df['dt'].max()}")
    print(f"测试集时间范围：{test_df['dt'].min()} 至 {test_df['dt'].max()}")
    print(f"\n价格变动分布：")
    print("训练集：")
    print((train_df['price_change'] != 0).value_counts(normalize=True))
    print("\n测试集：")
    print((test_df['price_change'] != 0).value_counts(normalize=True))
    print(f"\n特征数量：{len(df.columns)}")
    print("\n特征列表：")
    print(df.columns.tolist())

if __name__ == "__main__":
    main() 