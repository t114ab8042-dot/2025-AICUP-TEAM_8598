#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型訓練模組 - 2025 玉山人工智慧公開挑戰賽

此模組負責：
1. 處理不平衡資料（SMOTE）
2. 訓練 XGBoost 模型
3. 交叉驗證評估
4. 特徵重要性分析
5. 模型儲存
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, make_scorer
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

# 模型超參數設定
MODEL_PARAMS = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 300,
    'min_child_weight': 20,
    'gamma': 0.5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 3.0,
    'reg_lambda': 5.0,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss',
    'verbosity': 0
}

# SMOTE 參數
SMOTE_PARAMS = {
    'sampling_strategy': 0.2,
    'random_state': 42
}

# 交叉驗證參數
CV_PARAMS = {
    'n_splits': 5,
    'shuffle': True,
    'random_state': 42
}

def apply_smote(X_train, y_train):
    """
    應用 SMOTE 過採樣處理不平衡資料
    
    Parameters
    ----------
    X_train : np.ndarray
        訓練集特徵矩陣
    y_train : np.ndarray
        訓練集標籤
    
    Returns
    -------
    tuple
        (重採樣後的特徵矩陣, 重採樣後的標籤)
    """
    print("  應用 SMOTE 過採樣...")
    print(f"    原始分布 - 正類：{(y_train==1).sum()}，負類：{(y_train==0).sum()}")
    
    smote = SMOTE(**SMOTE_PARAMS)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"    SMOTE 後 - 正類：{(y_resampled==1).sum()}，負類：{(y_resampled==0).sum()}")
    print(f"    新比例：{(y_resampled==0).sum()/(y_resampled==1).sum():.1f}:1")
    
    return X_resampled, y_resampled

def perform_cross_validation(model, X_train, y_train):
    """
    執行交叉驗證評估模型性能
    
    Parameters
    ----------
    model : XGBClassifier
        XGBoost 模型
    X_train : np.ndarray
        訓練集特徵矩陣
    y_train : np.ndarray
        訓練集標籤
    
    Returns
    -------
    np.ndarray
        各 fold 的 F1 分數
    """
    print("\n  執行 5-Fold 交叉驗證...")
    
    cv = StratifiedKFold(**CV_PARAMS)
    f1_scorer = make_scorer(f1_score)
    
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=cv, scoring=f1_scorer, n_jobs=1
    )
    
    print(f"    ✓ 交叉驗證完成")
    print(f"    平均 F1 Score：{cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    print(f"    各 Fold 結果：")
    for i, score in enumerate(cv_scores):
        print(f"      Fold {i+1}: {score:.4f}")
    
    return cv_scores

def calculate_feature_importance(model, feature_names):
    """
    計算並顯示特徵重要性
    
    Parameters
    ----------
    model : XGBClassifier
        訓練好的模型
    feature_names : list
        特徵名稱列表
    
    Returns
    -------
    pd.DataFrame
        特徵重要性資料框
    """
    print("\n  分析特徵重要性...")
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("    【Top 15 重要特徵】")
    for idx, row in importance_df.head(15).iterrows():
        print(f"    {row['feature']:30s}: {row['importance']:.4f}")
    
    return importance_df

def train_model(X_train, y_train, feature_names):
    """
    主要的模型訓練函數
    
    Parameters
    ----------
    X_train : np.ndarray
        訓練集特徵矩陣
    y_train : np.ndarray
        訓練集標籤
    feature_names : list
        特徵名稱列表
    
    Returns
    -------
    tuple
        (訓練好的模型, 交叉驗證分數)
    """
    print("\n開始模型訓練流程...")
    print(f"  訓練資料維度：{X_train.shape}")
    print(f"  特徵數量：{len(feature_names)}")
    
    # Step 1: SMOTE 過採樣
    X_resampled, y_resampled = apply_smote(X_train, y_train)
    
    # Step 2: 初始化模型
    print("\n  初始化 XGBoost 模型...")
    
    # 動態計算 scale_pos_weight
    scale_pos_weight = (y_resampled==0).sum() / (y_resampled==1).sum()
    
    model_params = MODEL_PARAMS.copy()
    model_params['scale_pos_weight'] = scale_pos_weight
    
    model = XGBClassifier(**model_params)
    
    print("    模型參數：")
    for key, value in model_params.items():
        if key not in ['n_jobs', 'verbosity', 'random_state']:
            print(f"      {key}: {value}")
    
    # Step 3: 交叉驗證
    cv_scores = perform_cross_validation(model, X_resampled, y_resampled)
    
    # Step 4: 訓練最終模型
    print("\n  訓練最終模型...")
    model.fit(X_resampled, y_resampled)
    print("    ✓ 模型訓練完成")
    
    # Step 5: 特徵重要性
    importance_df = calculate_feature_importance(model, feature_names)
    
    # Step 6: 儲存模型
    print("\n  儲存模型...")
    joblib.dump(model, 'xgboost_model.pkl')
    print("    ✓ 模型已儲存至 xgboost_model.pkl")
    
    # 儲存特徵重要性
    importance_df.to_csv('feature_importance.csv', index=False)
    print("    ✓ 特徵重要性已儲存至 feature_importance.csv")
    
    return model, cv_scores

def validate_model_performance(model, X_train, y_train):
    """
    驗證模型在原始資料上的表現
    
    Parameters
    ----------
    model : XGBClassifier
        訓練好的模型
    X_train : np.ndarray
        原始訓練集特徵矩陣（未經 SMOTE）
    y_train : np.ndarray
        原始訓練集標籤
    
    Returns
    -------
    dict
        性能指標字典
    """
    print("\n  驗證模型性能...")
    
    # 預測
    y_pred = model.predict(X_train)
    y_pred_proba = model.predict_proba(X_train)[:, 1]
    
    # 計算指標
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("\n    混淆矩陣：")
    cm = confusion_matrix(y_train, y_pred)
    print(f"      TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"      FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
    
    print("\n    分類報告：")
    report = classification_report(y_train, y_pred, target_names=['正常', '警示'])
    for line in report.split('\n'):
        if line:
            print(f"      {line}")
    
    # 機率分布
    print("\n    預測機率分布：")
    print(f"      平均：{y_pred_proba.mean():.4f}")
    print(f"      中位數：{np.median(y_pred_proba):.4f}")
    print(f"      最小：{y_pred_proba.min():.4f}")
    print(f"      最大：{y_pred_proba.max():.4f}")
    
    return {
        'confusion_matrix': cm,
        'f1_score': f1_score(y_train, y_pred),
        'probabilities': y_pred_proba
    }

if __name__ == "__main__":
    """
    獨立執行測試
    """
    print("="*80)
    print("模型訓練模組 - 獨立執行模式")
    print("="*80)
    
    # 需要先執行資料前處理
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    
    try:
        from Preprocess.data_preprocess import preprocess_data
        
        print("\n執行資料前處理...")
        X_train, y_train, X_test, test_accounts, feature_names = preprocess_data()
        
        # 訓練模型
        model, cv_scores = train_model(X_train, y_train, feature_names)
        
        # 驗證性能
        performance = validate_model_performance(model, X_train, y_train)
        
        print("\n" + "="*80)
        print("訓練完成摘要")
        print("="*80)
        print(f"  交叉驗證 F1：{cv_scores.mean():.4f}")
        print(f"  訓練集 F1：{performance['f1_score']:.4f}")
        print(f"  模型檔案：xgboost_model.pkl")
        print(f"  特徵重要性：feature_importance.csv")
        
    except ImportError as e:
        print(f"\n❌ 錯誤：無法載入前處理模組")
        print(f"   請確保先執行 data_preprocess.py")
        print(f"   錯誤訊息：{str(e)}")
    except Exception as e:
        print(f"\n❌ 執行錯誤：{str(e)}")
        import traceback
        traceback.print_exc()