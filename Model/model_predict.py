#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型預測模組 - 2025 玉山人工智慧公開挑戰賽

此模組負責：
1. 載入訓練好的模型
2. 生成預測機率
3. 應用不同閾值產生二元預測
4. 生成符合比賽格式的提交檔案
5. 提供閾值選擇建議
"""

import numpy as np
import pandas as pd
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# 預設閾值設定
DEFAULT_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]

# 預期的警示帳戶比例範圍
EXPECTED_ALERT_RATIO_LOW = 0.5   # 0.5%
EXPECTED_ALERT_RATIO_HIGH = 2.0  # 2.0%

def load_model(model_path='xgboost_model.pkl'):
    """
    載入訓練好的模型
    
    Parameters
    ----------
    model_path : str
        模型檔案路徑
    
    Returns
    -------
    object
        載入的模型物件
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型檔案不存在：{model_path}")
    
    print(f"  載入模型：{model_path}")
    model = joblib.load(model_path)
    print("    ✓ 模型載入成功")
    
    return model

def predict_probabilities(model, X_test):
    """
    生成預測機率
    
    Parameters
    ----------
    model : object
        訓練好的模型
    X_test : np.ndarray
        測試集特徵矩陣
    
    Returns
    -------
    np.ndarray
        預測機率（正類機率）
    """
    print("\n  生成預測機率...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"    預測機率分布：")
    print(f"      平均：{y_pred_proba.mean():.4f}")
    print(f"      中位數：{np.median(y_pred_proba):.4f}")
    print(f"      標準差：{y_pred_proba.std():.4f}")
    print(f"      最小值：{y_pred_proba.min():.4f}")
    print(f"      最大值：{y_pred_proba.max():.4f}")
    
    # 顯示機率分位數
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print(f"\n    機率分位數：")
    for p in percentiles:
        value = np.percentile(y_pred_proba, p)
        print(f"      {p:2d}%: {value:.4f}")
    
    return y_pred_proba

def apply_threshold(probabilities, threshold):
    """
    應用閾值將機率轉換為二元預測
    
    Parameters
    ----------
    probabilities : np.ndarray
        預測機率
    threshold : float
        閾值
    
    Returns
    -------
    np.ndarray
        二元預測結果
    """
    predictions = (probabilities >= threshold).astype(int)
    return predictions

def create_submission_file(test_accounts, predictions, filename):
    """
    建立符合比賽格式的提交檔案
    
    Parameters
    ----------
    test_accounts : np.ndarray
        測試集帳戶ID
    predictions : np.ndarray
        預測結果
    filename : str
        輸出檔案名稱
    
    Returns
    -------
    pd.DataFrame
        提交資料框
    """
    submission_df = pd.DataFrame({
        'acct': test_accounts,
        'label': predictions
    })
    
    # 儲存檔案
    submission_df.to_csv(filename, index=False)
    
    return submission_df

def analyze_threshold_results(probabilities, thresholds, total_accounts):
    """
    分析不同閾值的預測結果
    
    Parameters
    ----------
    probabilities : np.ndarray
        預測機率
    thresholds : list
        閾值列表
    total_accounts : int
        總帳戶數
    
    Returns
    -------
    pd.DataFrame
        分析結果資料框
    """
    results = []
    
    for threshold in thresholds:
        predictions = apply_threshold(probabilities, threshold)
        alert_count = predictions.sum()
        alert_ratio = alert_count / total_accounts * 100
        
        results.append({
            'threshold': threshold,
            'alert_count': alert_count,
            'alert_ratio': alert_ratio,
            'normal_count': total_accounts - alert_count,
            'in_expected_range': EXPECTED_ALERT_RATIO_LOW <= alert_ratio <= EXPECTED_ALERT_RATIO_HIGH
        })
    
    return pd.DataFrame(results)

def recommend_threshold(analysis_df):
    """
    推薦最佳閾值
    
    Parameters
    ----------
    analysis_df : pd.DataFrame
        閾值分析結果
    
    Returns
    -------
    float
        推薦的閾值
    """
    # 優先選擇在預期範圍內的閾值
    in_range = analysis_df[analysis_df['in_expected_range']]
    
    if not in_range.empty:
        # 選擇最接近範圍中心的閾值
        target_ratio = (EXPECTED_ALERT_RATIO_LOW + EXPECTED_ALERT_RATIO_HIGH) / 2
        in_range['distance'] = abs(in_range['alert_ratio'] - target_ratio)
        best_threshold = in_range.loc[in_range['distance'].idxmin(), 'threshold']
    else:
        # 如果沒有在範圍內的，選擇最接近下限的
        analysis_df['distance'] = abs(analysis_df['alert_ratio'] - EXPECTED_ALERT_RATIO_LOW)
        best_threshold = analysis_df.loc[analysis_df['distance'].idxmin(), 'threshold']
    
    return best_threshold

def generate_predictions(model, X_test, test_accounts, thresholds=None):
    """
    主要的預測生成函數
    
    Parameters
    ----------
    model : object
        訓練好的模型
    X_test : np.ndarray
        測試集特徵矩陣
    test_accounts : np.ndarray
        測試集帳戶ID
    thresholds : list or None
        要測試的閾值列表
    
    Returns
    -------
    list
        生成的提交檔案列表
    """
    print("\n開始生成預測...")
    print(f"  測試集大小：{len(X_test):,}")
    
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    
    # 生成預測機率
    probabilities = predict_probabilities(model, X_test)
    
    # 分析不同閾值
    print("\n  分析不同閾值的結果...")
    analysis_df = analyze_threshold_results(probabilities, thresholds, len(test_accounts))
    
    print("\n    【閾值分析結果】")
    print("    " + "-"*70)
    print(f"    {'閾值':^8} | {'預測警示':^10} | {'比例(%)':^10} | {'是否在預期範圍':^15}")
    print("    " + "-"*70)
    
    submission_files = []
    
    for _, row in analysis_df.iterrows():
        threshold = row['threshold']
        alert_count = int(row['alert_count'])
        alert_ratio = row['alert_ratio']
        in_range = row['in_expected_range']
        
        # 生成預測
        predictions = apply_threshold(probabilities, threshold)
        
        # 建立檔名
        filename = f'submission_threshold_{int(threshold*100)}.csv'
        
        # 建立提交檔案
        create_submission_file(test_accounts, predictions, filename)
        submission_files.append(filename)
        
        # 顯示結果
        range_mark = "✓" if in_range else " "
        print(f"    {threshold:^8.2f} | {alert_count:^10d} | {alert_ratio:^10.2f} | {range_mark:^15}")
    
    print("    " + "-"*70)
    
    # 推薦最佳閾值
    best_threshold = recommend_threshold(analysis_df)
    print(f"\n  ✅ 推薦閾值：{best_threshold:.2f}")
    
    best_row = analysis_df[analysis_df['threshold'] == best_threshold].iloc[0]
    print(f"     預測警示數：{int(best_row['alert_count'])}")
    print(f"     預測比例：{best_row['alert_ratio']:.2f}%")
    
    # 基於歷史資料的分析
    print(f"\n  📊 參考資訊：")
    print(f"     歷史警示比例：約 0.3% (1,004/333,768)")
    print(f"     預期測試集警示比例：{EXPECTED_ALERT_RATIO_LOW}-{EXPECTED_ALERT_RATIO_HIGH}%")
    print(f"     預期警示帳戶數：{int(len(test_accounts)*EXPECTED_ALERT_RATIO_LOW/100)}-{int(len(test_accounts)*EXPECTED_ALERT_RATIO_HIGH/100)}")
    
    return submission_files

def load_predict_file():
    """
    載入待預測帳戶清單
    
    Returns
    -------
    pd.DataFrame
        待預測帳戶資料框
    """
    predict_file = 'acct_predict.csv'
    
    if not os.path.exists(predict_file):
        raise FileNotFoundError(f"待預測檔案不存在：{predict_file}")
    
    df_predict = pd.read_csv(predict_file)
    df_predict['acct'] = df_predict['acct'].astype(str)
    
    return df_predict

if __name__ == "__main__":
    """
    獨立執行測試
    """
    print("="*80)
    print("模型預測模組 - 獨立執行模式")
    print("="*80)
    
    # 需要先執行前處理和訓練
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    
    try:
        # 載入模型
        model = load_model()
        
        # 執行前處理獲取測試資料
        from Preprocess.data_preprocess import preprocess_data
        print("\n執行資料前處理...")
        X_train, y_train, X_test, test_accounts, feature_names = preprocess_data()
        
        # 生成預測
        submission_files = generate_predictions(model, X_test, test_accounts)
        
        print("\n" + "="*80)
        print("預測完成摘要")
        print("="*80)
        print(f"  生成檔案數：{len(submission_files)}")
        print(f"  檔案列表：")
        for file in submission_files:
            print(f"    - {file}")
        
        print(f"\n  建議提交順序：")
        print(f"    1. submission_threshold_60.csv （推薦）")
        print(f"    2. submission_threshold_50.csv （基準）")
        print(f"    3. 根據分數調整閾值")
        
    except FileNotFoundError as e:
        print(f"\n❌ 錯誤：{str(e)}")
        print(f"   請先執行 model_train.py 訓練模型")
    except Exception as e:
        print(f"\n❌ 執行錯誤：{str(e)}")
        import traceback
        traceback.print_exc()