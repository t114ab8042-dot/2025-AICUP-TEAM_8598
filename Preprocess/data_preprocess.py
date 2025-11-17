#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
資料前處理模組 - 2025 玉山人工智慧公開挑戰賽

此模組負責：
1. 讀取原始資料
2. 資料清洗與轉換
3. 特徵工程
4. 訓練集抽樣
5. 資料標準化
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

# 全域參數設定
ACTIVITY_THRESHOLD = 20  # 活躍帳戶門檻（交易筆數）
SAMPLE_RATIO = 10  # 正常帳戶與警示帳戶的抽樣比例
RANDOM_STATE = 42  # 隨機種子

def time_to_minutes(time_str):
    """
    將時間字串轉換為分鐘數
    
    Parameters
    ----------
    time_str : str
        時間字串，格式為 "HH:MM"
    
    Returns
    -------
    float
        分鐘數，如果無法轉換則返回 NaN
    """
    try:
        if pd.isna(time_str):
            return np.nan
        parts = str(time_str).split(':')
        return int(parts[0]) * 60 + int(parts[1])
    except:
        return np.nan

def load_data():
    """
    讀取所有必要的資料檔案
    
    Returns
    -------
    tuple
        包含 (df_txn, df_alert, df_predict) 的元組
    """
    print("  讀取資料檔案...")
    
    # 讀取交易資料
    df_txn = pd.read_csv('acct_transaction.csv')
    print(f"  ✓ 交易資料：{len(df_txn):,} 筆")
    
    # 讀取警示帳戶資料
    df_alert = pd.read_csv('acct_alert.csv')
    print(f"  ✓ 警示帳戶：{len(df_alert):,} 個")
    
    # 讀取待預測帳戶資料
    df_predict = pd.read_csv('acct_predict.csv')
    print(f"  ✓ 待預測帳戶：{len(df_predict):,} 個")
    
    # 轉換時間欄位
    df_txn['txn_time_min'] = df_txn['txn_time'].apply(time_to_minutes)
    
    return df_txn, df_alert, df_predict

def identify_accounts(df_txn, df_alert, df_predict):
    """
    識別不同類型的帳戶
    
    Parameters
    ----------
    df_txn : pd.DataFrame
        交易資料
    df_alert : pd.DataFrame
        警示帳戶資料
    df_predict : pd.DataFrame
        待預測帳戶資料
    
    Returns
    -------
    dict
        包含各類帳戶集合的字典
    """
    print("  識別帳戶類型...")
    
    # 玉山帳戶
    esun_accounts = set(df_txn[df_txn['from_acct_type'] == 1]['from_acct'].astype(str)) | \
                    set(df_txn[df_txn['to_acct_type'] == 1]['to_acct'].astype(str))
    
    # 警示帳戶字典
    df_alert['acct'] = df_alert['acct'].astype(str)
    alert_dict = dict(zip(df_alert['acct'], df_alert['event_date']))
    
    # 待預測帳戶
    test_set = set(df_predict['acct'].astype(str))
    
    # 訓練帳戶
    train_accounts = esun_accounts - test_set
    train_alert = set(alert_dict.keys()) & train_accounts
    train_normal = train_accounts - set(alert_dict.keys())
    
    print(f"  ✓ 訓練集警示帳戶：{len(train_alert):,}")
    print(f"  ✓ 訓練集正常帳戶：{len(train_normal):,}")
    print(f"  ✓ 測試集帳戶：{len(test_set):,}")
    
    return {
        'train_alert': train_alert,
        'train_normal': train_normal,
        'test_set': test_set,
        'alert_dict': alert_dict
    }

def calculate_account_activity(df_txn, accounts):
    """
    計算帳戶的交易活躍度
    
    Parameters
    ----------
    df_txn : pd.DataFrame
        交易資料
    accounts : set
        要計算的帳戶集合
    
    Returns
    -------
    dict
        帳戶交易數字典
    """
    print("  計算帳戶活躍度...")
    
    # 批次計算交易數
    from_counts = df_txn[df_txn['from_acct_type']==1].groupby('from_acct').size().to_dict()
    to_counts = df_txn[df_txn['to_acct_type']==1].groupby('to_acct').size().to_dict()
    
    account_txn_count = {}
    for acct in accounts:
        count = from_counts.get(acct, 0) + to_counts.get(acct, 0)
        account_txn_count[acct] = count
    
    return account_txn_count

def sample_training_accounts(train_normal, train_alert, account_txn_count):
    """
    智能抽樣訓練帳戶（只選擇活躍帳戶）
    
    Parameters
    ----------
    train_normal : set
        正常帳戶集合
    train_alert : set
        警示帳戶集合
    account_txn_count : dict
        帳戶交易數字典
    
    Returns
    -------
    tuple
        (抽樣的正常帳戶, 全局平均交易數, 全局中位數交易數)
    """
    print("  執行智能抽樣...")
    
    # 篩選活躍的正常帳戶
    active_normal = [acct for acct in train_normal 
                     if account_txn_count.get(acct, 0) >= ACTIVITY_THRESHOLD]
    
    print(f"  ✓ 活躍正常帳戶（>={ACTIVITY_THRESHOLD}筆）：{len(active_normal):,}")
    
    # 抽樣
    np.random.seed(RANDOM_STATE)
    sample_size = min(len(active_normal), len(train_alert) * SAMPLE_RATIO)
    normal_sample = np.random.choice(active_normal, sample_size, replace=False)
    
    print(f"  ✓ 抽樣正常帳戶：{len(normal_sample):,}")
    print(f"  ✓ 最終比例：{len(normal_sample)/len(train_alert):.1f}:1")
    
    # 計算全局統計
    global_mean = np.mean(list(account_txn_count.values()))
    global_median = np.median(list(account_txn_count.values()))
    
    return normal_sample, global_mean, global_median

def calculate_features_enhanced(acct, cutoff_date, df_txn, global_stats):
    """
    計算增強版特徵（33個特徵）
    
    Parameters
    ----------
    acct : str
        帳戶ID
    cutoff_date : int
        截止日期
    df_txn : pd.DataFrame
        交易資料
    global_stats : dict
        全局統計資訊
    
    Returns
    -------
    dict
        特徵字典
    """
    features = {}
    
    # 篩選相關交易
    from_txn = df_txn[(df_txn['from_acct'] == acct) & (df_txn['txn_date'] <= cutoff_date)]
    to_txn = df_txn[(df_txn['to_acct'] == acct) & (df_txn['txn_date'] <= cutoff_date)]
    all_txn = pd.concat([from_txn, to_txn])
    
    total_count = len(all_txn)
    
    # ========== Group 1: 跨行交易特徵 (5個) ==========
    if len(from_txn) > 0:
        cross_from = from_txn[from_txn['from_acct_type'] != from_txn['to_acct_type']]
        features['from_cross_ratio'] = len(cross_from) / len(from_txn)
        features['from_cross_count'] = len(cross_from)
    else:
        features['from_cross_ratio'] = 0
        features['from_cross_count'] = 0
    
    if len(to_txn) > 0:
        cross_to = to_txn[to_txn['from_acct_type'] != to_txn['to_acct_type']]
        features['to_cross_ratio'] = len(cross_to) / len(to_txn)
        features['to_cross_count'] = len(cross_to)
    else:
        features['to_cross_ratio'] = 0
        features['to_cross_count'] = 0
    
    if len(all_txn) > 0:
        cross_all = all_txn[all_txn['from_acct_type'] != all_txn['to_acct_type']]
        features['total_cross_ratio'] = len(cross_all) / len(all_txn)
    else:
        features['total_cross_ratio'] = 0
    
    # ========== Group 2: 快速轉出特徵 (3個) ==========
    if len(to_txn) > 0 and len(from_txn) > 0:
        to_dates = set(to_txn['txn_date'])
        from_dates = set(from_txn['txn_date'])
        same_day_dates = to_dates & from_dates
        active_days = len(set(all_txn['txn_date']))
        features['same_day_inout_ratio'] = len(same_day_dates) / max(active_days, 1)
        features['same_day_count'] = len(same_day_dates)
        
        balance_days = 0
        for date in set(all_txn['txn_date']):
            day_to = to_txn[to_txn['txn_date'] == date]['txn_amt'].sum()
            day_from = from_txn[from_txn['txn_date'] == date]['txn_amt'].sum()
            if day_to > 0 and day_from > 0:
                ratio = day_to / day_from
                if 0.8 <= ratio <= 1.2:
                    balance_days += 1
        features['balance_days'] = balance_days
    else:
        features['same_day_inout_ratio'] = 0
        features['same_day_count'] = 0
        features['balance_days'] = 0
    
    # ========== Group 3: 收款特徵 (5個) ==========
    if len(to_txn) > 0:
        features['to_count'] = len(to_txn)
        features['to_total_amt'] = to_txn['txn_amt'].sum()
        features['to_mean_amt'] = to_txn['txn_amt'].mean()
        features['to_daily_avg'] = len(to_txn) / max(cutoff_date, 1)
        features['to_unique_senders'] = to_txn['from_acct'].nunique()
    else:
        features['to_count'] = 0
        features['to_total_amt'] = 0
        features['to_mean_amt'] = 0
        features['to_daily_avg'] = 0
        features['to_unique_senders'] = 0
    
    # ========== Group 4: 收支平衡特徵 (5個) ==========
    to_amt = features['to_total_amt']
    from_amt = from_txn['txn_amt'].sum() if len(from_txn) > 0 else 0
    
    features['from_total_amt'] = from_amt
    features['inout_ratio'] = to_amt / (from_amt + 1)
    features['net_flow'] = to_amt - from_amt
    features['balance_score'] = 1 - abs(features['net_flow']) / (to_amt + from_amt + 1)
    features['is_balanced'] = 1 if 0.8 <= features['inout_ratio'] <= 1.2 else 0
    
    # ========== Group 5: 匯出特徵 (3個) ==========
    if len(from_txn) > 0:
        features['from_count'] = len(from_txn)
        features['from_mean_amt'] = from_txn['txn_amt'].mean()
        features['from_unique_receivers'] = from_txn['to_acct'].nunique()
    else:
        features['from_count'] = 0
        features['from_mean_amt'] = 0
        features['from_unique_receivers'] = 0
    
    # ========== Group 6: 交易量特徵 (2個) ==========
    features['total_count'] = total_count
    features['txn_diversity'] = features['to_unique_senders'] + features['from_unique_receivers']
    
    # ========== Group 7: 時間窗口特徵 (3個) ==========
    if cutoff_date >= 30:
        recent_start = cutoff_date - 30
        recent_txn = all_txn[all_txn['txn_date'] > recent_start]
        recent_from = from_txn[from_txn['txn_date'] > recent_start]
        recent_to = to_txn[to_txn['txn_date'] > recent_start]
        
        features['last30d_count'] = len(recent_txn)
        
        if len(recent_txn) > 0:
            recent_cross = recent_txn[recent_txn['from_acct_type'] != recent_txn['to_acct_type']]
            features['last30d_cross_ratio'] = len(recent_cross) / len(recent_txn)
        else:
            features['last30d_cross_ratio'] = 0
        
        if len(recent_from) > 0:
            features['last30d_from_ratio'] = len(recent_from) / len(recent_txn)
        else:
            features['last30d_from_ratio'] = 0
    else:
        features['last30d_count'] = 0
        features['last30d_cross_ratio'] = 0
        features['last30d_from_ratio'] = 0
    
    # ========== Group 8: 其他統計特徵 (4個) ==========
    if len(all_txn) > 0:
        features['amt_std'] = all_txn['txn_amt'].std()
        features['amt_max'] = all_txn['txn_amt'].max()
        features['active_days'] = all_txn['txn_date'].nunique()
        features['txn_per_day'] = len(all_txn) / max(features['active_days'], 1)
    else:
        features['amt_std'] = 0
        features['amt_max'] = 0
        features['active_days'] = 0
        features['txn_per_day'] = 0
    
    # ========== Group 9: 相對活躍度特徵 (3個) ==========
    features['relative_activity'] = total_count / max(global_stats['mean'], 1)
    features['above_median'] = 1 if total_count > global_stats['median'] else 0
    
    # 活躍度等級
    if total_count < global_stats['q25']:
        activity_level = 1
    elif total_count < global_stats['median']:
        activity_level = 2
    elif total_count < global_stats['q75']:
        activity_level = 3
    else:
        activity_level = 4
    features['activity_level'] = activity_level
    
    return features

def build_dataset(accounts, df_txn, alert_dict, global_stats, is_training=True, label=None):
    """
    構建資料集
    
    Parameters
    ----------
    accounts : set or list
        帳戶列表
    df_txn : pd.DataFrame
        交易資料
    alert_dict : dict
        警示帳戶字典
    global_stats : dict
        全局統計資訊
    is_training : bool
        是否為訓練集
    label : int or None
        標籤值（僅訓練集使用）
    
    Returns
    -------
    list
        特徵字典列表
    """
    data = []
    
    for i, acct in enumerate(accounts):
        if (i + 1) % 500 == 0:
            print(f"    處理進度：{i+1}/{len(accounts)}")
        
        if is_training:
            if label == 1:  # 警示帳戶
                event_date = alert_dict[acct]
                cutoff_date = event_date - 1
                if cutoff_date <= 0:
                    continue
            else:  # 正常帳戶
                cutoff_date = np.random.randint(30, 122)
        else:  # 測試集
            cutoff_date = 121
        
        features = calculate_features_enhanced(acct, cutoff_date, df_txn, global_stats)
        features['acct'] = acct
        
        if is_training:
            features['label'] = label
        
        data.append(features)
    
    return data

def preprocess_data():
    """
    主要的資料前處理函數
    
    Returns
    -------
    tuple
        (X_train, y_train, X_test, test_accounts, feature_names)
    """
    print("\n執行資料前處理...")
    
    # 載入資料
    df_txn, df_alert, df_predict = load_data()
    
    # 識別帳戶
    account_groups = identify_accounts(df_txn, df_alert, df_predict)
    
    # 計算所有訓練帳戶的活躍度
    all_train = account_groups['train_alert'] | account_groups['train_normal']
    account_txn_count = calculate_account_activity(df_txn, all_train)
    
    # 智能抽樣
    normal_sample, global_mean, global_median = sample_training_accounts(
        account_groups['train_normal'],
        account_groups['train_alert'],
        account_txn_count
    )
    
    # 準備全局統計
    global_stats = {
        'mean': global_mean,
        'median': global_median,
        'q25': np.percentile(list(account_txn_count.values()), 25),
        'q75': np.percentile(list(account_txn_count.values()), 75)
    }
    
    print("\n構建訓練集...")
    # 構建訓練集
    alert_data = build_dataset(
        account_groups['train_alert'],
        df_txn,
        account_groups['alert_dict'],
        global_stats,
        is_training=True,
        label=1
    )
    
    normal_data = build_dataset(
        normal_sample,
        df_txn,
        account_groups['alert_dict'],
        global_stats,
        is_training=True,
        label=0
    )
    
    # 合併訓練資料
    training_data = alert_data + normal_data
    df_train = pd.DataFrame(training_data)
    
    print(f"  ✓ 訓練集構建完成：{len(df_train):,} 筆")
    print(f"    - 警示帳戶：{(df_train['label']==1).sum():,}")
    print(f"    - 正常帳戶：{(df_train['label']==0).sum():,}")
    
    print("\n構建測試集...")
    # 構建測試集
    test_data = build_dataset(
        account_groups['test_set'],
        df_txn,
        account_groups['alert_dict'],
        global_stats,
        is_training=False
    )
    
    df_test = pd.DataFrame(test_data)
    print(f"  ✓ 測試集構建完成：{len(df_test):,} 筆")
    
    # 準備特徵矩陣
    feature_cols = [col for col in df_train.columns if col not in ['acct', 'label']]
    
    X_train = df_train[feature_cols].fillna(0).values
    y_train = df_train['label'].values
    X_test = df_test[feature_cols].fillna(0).values
    test_accounts = df_test['acct'].values
    
    # 資料標準化
    print("\n執行資料標準化...")
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print("  ✓ 標準化完成")
    
    # 保存 scaler（可選）
    import joblib
    joblib.dump(scaler, 'scaler.pkl')
    print("  ✓ 儲存標準化器至 scaler.pkl")
    
    return X_train, y_train, X_test, test_accounts, feature_cols

if __name__ == "__main__":
    """
    獨立執行測試
    """
    print("="*80)
    print("資料前處理模組 - 獨立執行模式")
    print("="*80)
    
    X_train, y_train, X_test, test_accounts, feature_names = preprocess_data()
    
    print("\n處理結果摘要：")
    print(f"  訓練集：{X_train.shape}")
    print(f"  測試集：{X_test.shape}")
    print(f"  特徵數：{len(feature_names)}")
    print("\n✓ 資料前處理完成")