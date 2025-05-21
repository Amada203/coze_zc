import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

def safe_pct_change(series):
    prev = series.shift(1)
    pct = (series - prev) / prev.replace(0, np.nan)
    return pct

def load_data():
    print("正在加载数据...")
    df = pd.read_csv('/Users/ruixue.li/lrx/coze_zc/query-impala-1632023.csv')
    df['dt'] = pd.to_datetime(df['dt'])
    # 处理价格缺失值
    print("正在处理价格缺失值...")
    df = df.sort_values(['sku_id', 'dt'])
    df['discount_price'] = df.groupby('sku_id')['discount_price'].transform(
        lambda x: x.fillna(method='ffill').fillna(method='bfill')
    )
    missing_count = df['discount_price'].isnull().sum()
    if missing_count > 0:
        print(f"警告：仍有 {missing_count} 条记录的价格为缺失值")
        print("这些记录将被删除")
        df = df.dropna(subset=['discount_price'])
    return df

def add_time_features(df):
    print("正在添加时间特征...")
    df['year'] = df['dt'].dt.year
    df['month'] = df['dt'].dt.month
    df['day'] = df['dt'].dt.day
    df['weekday'] = df['dt'].dt.weekday
    df['quarter'] = df['dt'].dt.quarter
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
    df['is_month_start'] = df['dt'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['dt'].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df['dt'].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df['dt'].dt.is_quarter_end.astype(int)
    # 电商大促日期
    promotion_dates = {
        'double_11': [
            '2023-11-01', '2023-11-02', '2023-11-03', '2023-11-04', '2023-11-05',
            '2023-11-06', '2023-11-07', '2023-11-08', '2023-11-09', '2023-11-10',
            '2023-11-11', '2023-11-12'
        ],
        'double_12': [
            '2023-12-01', '2023-12-02', '2023-12-03', '2023-12-04', '2023-12-05',
            '2023-12-06', '2023-12-07', '2023-12-08', '2023-12-09', '2023-12-10',
            '2023-12-11', '2023-12-12'
        ],
        '618': [
            '2023-06-01', '2023-06-02', '2023-06-03', '2023-06-04', '2023-06-05',
            '2023-06-06', '2023-06-07', '2023-06-08', '2023-06-09', '2023-06-10',
            '2023-06-11', '2023-06-12', '2023-06-13', '2023-06-14', '2023-06-15',
            '2023-06-16', '2023-06-17', '2023-06-18'
        ],
        'spring_festival': [
            '2023-01-10', '2023-01-11', '2023-01-12', '2023-01-13', '2023-01-14',
            '2023-01-15', '2023-01-16', '2023-01-17', '2023-01-18', '2023-01-19',
            '2023-01-20', '2023-01-21', '2023-01-22', '2023-01-23', '2023-01-24',
            '2023-01-25', '2023-01-26', '2023-01-27'
        ],
        'womens_day': [
            '2023-03-01', '2023-03-02', '2023-03-03', '2023-03-04', '2023-03-05',
            '2023-03-06', '2023-03-07', '2023-03-08'
        ],
        'labor_day': [
            '2023-04-28', '2023-04-29', '2023-04-30', '2023-05-01', '2023-05-02',
            '2023-05-03', '2023-05-04', '2023-05-05'
        ],
        'national_day': [
            '2023-09-28', '2023-09-29', '2023-09-30', '2023-10-01', '2023-10-02',
            '2023-10-03', '2023-10-04', '2023-10-05', '2023-10-06', '2023-10-07'
        ]
    }
    df['is_promotion_period'] = 0
    for promo_name, dates in promotion_dates.items():
        df[f'is_{promo_name}'] = df['dt'].dt.strftime('%Y-%m-%d').isin(dates).astype(int)
        df['is_promotion_period'] = df['is_promotion_period'] | df[f'is_{promo_name}']
    df['is_before_promotion'] = 0
    df['is_after_promotion'] = 0
    for promo_name, dates in promotion_dates.items():
        dates = pd.to_datetime(dates)
        start_date = dates.min()
        end_date = dates.max()
        before_start = start_date - pd.Timedelta(days=7)
        mask_before = (df['dt'] >= before_start) & (df['dt'] < start_date)
        df.loc[mask_before, 'is_before_promotion'] = 1
        after_end = end_date + pd.Timedelta(days=7)
        mask_after = (df['dt'] > end_date) & (df['dt'] <= after_end)
        df.loc[mask_after, 'is_after_promotion'] = 1
    return df

def add_discount_price_features(df):
    print("正在添加价格特征...")
    grouped = df.groupby('sku_id')
    df['discount_price_change'] = grouped['discount_price'].diff().shift(1)
    df['discount_price_change'] = df['discount_price_change'].replace([np.inf, -np.inf], np.nan)
    df['discount_price_change_pct'] = grouped['discount_price'].transform(safe_pct_change).shift(1)
    df['discount_price_change_pct'] = df['discount_price_change_pct'].replace([np.inf, -np.inf], np.nan)
    df['discount_price_change_abs'] = df['discount_price_change'].abs()
    df['discount_price_mean_7'] = grouped['discount_price'].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).mean()
    ).shift(1).replace([np.inf, -np.inf], np.nan)
    df['discount_price_std_7'] = grouped['discount_price'].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).std()
    ).shift(1).replace([np.inf, -np.inf], np.nan)
    df['discount_price_min_7'] = grouped['discount_price'].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).min()
    ).shift(1).replace([np.inf, -np.inf], np.nan)
    df['discount_price_max_7'] = grouped['discount_price'].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).max()
    ).shift(1).replace([np.inf, -np.inf], np.nan)
    def safe_trend(x):
        try:
            if len(x) < 2:
                return 0
            return np.mean(np.diff(x))
        except:
            return 0
    df['discount_price_trend_7'] = grouped['discount_price'].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=2).apply(safe_trend)
    ).shift(1).replace([np.inf, -np.inf], np.nan)
    for lag in [1, 3, 7, 14, 30]:
        df[f'discount_price_lag_{lag}'] = grouped['discount_price'].shift(lag+1).replace([np.inf, -np.inf], np.nan)
    return df

def add_lag_features(df):
    print("正在添加滞后特征...")
    grouped = df.groupby('sku_id')
    for lag in [1, 3, 7, 14, 30]:
        df[f'discount_price_change_lag_{lag}'] = grouped['discount_price_change'].shift(lag+1).replace([np.inf, -np.inf], np.nan)
    return df

def add_promotion_features(df):
    print("正在添加促销特征...")
    grouped = df.groupby('sku_id')
    if 'is_promotion' not in df.columns:
        df['is_promotion'] = 0
    df['discount_price_promo_interaction'] = (df['discount_price'] * df['is_promotion']).shift(1).replace([np.inf, -np.inf], np.nan)
    df['promo_freq_7'] = grouped['is_promotion'].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).mean()
    ).shift(1).replace([np.inf, -np.inf], np.nan)
    df['promo_freq_30'] = grouped['is_promotion'].transform(
        lambda x: x.shift(1).rolling(window=30, min_periods=1).mean()
    ).shift(1).replace([np.inf, -np.inf], np.nan)
    return df

def add_prophet_static_features(df):
    print("正在合并Prophet静态特征...")
    prophet_static = pd.read_csv('商品价格时间序列特征_v2.csv')
    static_cols = ['sku_id', 'trend', 'yearly', 'weekly']
    prophet_static = prophet_static[static_cols].drop_duplicates('sku_id')
    df = pd.merge(df, prophet_static, on='sku_id', how='left')
    return df

def add_prophet_dynamic_features_train(df):
    print("正在生成训练集Prophet动态特征...")
    prophet_predictions = []
    for sku_id, group in df.groupby('sku_id'):
        prophet_df = group[['dt', 'discount_price']].rename(columns={'dt': 'ds', 'discount_price': 'y'})
        model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=True)
        model.fit(prophet_df)
        forecast = model.predict(prophet_df[['ds']])
        forecast['sku_id'] = sku_id
        forecast['dt'] = forecast['ds']
        prophet_predictions.append(forecast[['sku_id', 'dt', 'yhat', 'yhat_lower', 'yhat_upper']])
    prophet_pred = pd.concat(prophet_predictions, ignore_index=True)
    df = pd.merge(df, prophet_pred, on=['sku_id', 'dt'], how='left')
    # shift(1)防止未来信息泄露
    df['yhat'] = df.groupby('sku_id')['yhat'].shift(1)
    df['yhat_lower'] = df.groupby('sku_id')['yhat_lower'].shift(1)
    df['yhat_upper'] = df.groupby('sku_id')['yhat_upper'].shift(1)
    df['yhat_diff'] = df.groupby('sku_id')['yhat'].diff()
    df['yhat_residual'] = df['discount_price'] - df['yhat']
    return df

def add_prophet_dynamic_features_test(df):
    print("正在生成测试集Prophet动态特征（严格无泄露）...")
    result_list = []
    for sku_id, group in df.groupby('sku_id'):
        group = group.sort_values('dt')
        preds = []
        for idx, row in group.iterrows():
            cur_date = row['dt']
            # 只用T-1及更早数据
            history = group[group['dt'] < cur_date]
            if len(history) < 2:
                preds.append({'sku_id': sku_id, 'dt': cur_date, 'yhat': np.nan, 'yhat_lower': np.nan, 'yhat_upper': np.nan})
                continue
            prophet_df = history[['dt', 'discount_price']].rename(columns={'dt': 'ds', 'discount_price': 'y'})
            model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=True)
            model.fit(prophet_df)
            future = pd.DataFrame({'ds': [cur_date]})
            forecast = model.predict(future)
            preds.append({
                'sku_id': sku_id,
                'dt': cur_date,
                'yhat': forecast['yhat'].values[0],
                'yhat_lower': forecast['yhat_lower'].values[0],
                'yhat_upper': forecast['yhat_upper'].values[0]
            })
        preds_df = pd.DataFrame(preds)
        result_list.append(preds_df)
    prophet_pred = pd.concat(result_list, ignore_index=True)
    df = pd.merge(df, prophet_pred, on=['sku_id', 'dt'], how='left')
    df['yhat_diff'] = df.groupby('sku_id')['yhat'].diff()
    df['yhat_residual'] = np.nan  # 测试集不能用真实价格生成残差
    return df

def main():
    df = load_data()
    # 只做基础清洗，不做特征工程
    df = df.sort_values(['sku_id', 'dt'])
    split_date = pd.to_datetime('2024-12-31')
    train_df = df[df['dt'] < split_date].copy()
    test_df = df[df['dt'] >= split_date].copy()
    # 分别做特征工程
    train_df = add_time_features(train_df)
    train_df = add_discount_price_features(train_df)
    train_df = add_lag_features(train_df)
    train_df = add_promotion_features(train_df)
    test_df = add_time_features(test_df)
    test_df = add_discount_price_features(test_df)
    test_df = add_lag_features(test_df)
    test_df = add_promotion_features(test_df)
    # Prophet动态特征
    train_df = add_prophet_dynamic_features_train(train_df)
    test_df = add_prophet_dynamic_features_test(test_df)
    # 归一化前统一处理inf和nan
    train_df = train_df.replace([np.inf, -np.inf], np.nan)
    train_df = train_df.fillna(0)
    test_df = test_df.replace([np.inf, -np.inf], np.nan)
    test_df = test_df.fillna(0)
    # 标签生成：T日价格是否变动
    train_df['label'] = (train_df['discount_price'] != train_df['discount_price_lag_1']).astype(int)
    test_df['label'] = (test_df['discount_pr ice'] != test_df['discount_price_lag_1']).astype(int)
    # 保存处理后的数据
    train_file = '预处理后的商品数据_训练集_全特征.csv'
    test_file = '预处理后的商品数据_测试集_全特征.csv'
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
    print((train_df['discount_price'] != train_df['discount_price_lag_1']).value_counts(normalize=True))
    print("\n测试集：")
    print((test_df['discount_price'] != test_df['discount_price_lag_1']).value_counts(normalize=True))
    print(f"\n特征数量：{len(train_df.columns)}")
    print("\n特征列表：")
    print(train_df.columns.tolist())

if __name__ == "__main__":
    main() 