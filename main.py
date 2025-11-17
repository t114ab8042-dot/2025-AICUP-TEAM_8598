import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

# 添加子目錄到系統路徑
sys.path.append(os.path.join(os.path.dirname(__file__), 'Preprocess'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Model'))

def print_header(title):
    """
    列印格式化的標題
    
    Parameters
    ----------
    title : str
        要顯示的標題文字
    """
    print("\n" + "="*80)
    print(f"🚀 {title}")
    print("="*80)

def check_data_files():
    """
    檢查必要的資料檔案是否存在
    
    Returns
    -------
    bool
        如果所有檔案都存在返回 True，否則返回 False
    """
    required_files = [
        'acct_transaction.csv',
        'acct_alert.csv',
        'acct_predict.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ 錯誤：以下必要檔案不存在：")
        for file in missing_files:
            print(f"   - {file}")
        print("\n請確保所有資料檔案都在專案根目錄中。")
        return False
    
    return True

def main():
    """
    主要執行函數
    
    執行完整的模型訓練與預測流程。
    """
    print_header("2025 玉山人工智慧公開挑戰賽 - 初賽")
    print("版本：1.0.0")
    print("執行時間：" + time.strftime("%Y-%m-%d %H:%M:%S"))
    
    # 步驟 1：檢查資料檔案
    print("\n▶ 步驟 1：檢查資料檔案")
    print("-"*80)
    if not check_data_files():
        sys.exit(1)
    print("✓ 所有資料檔案已就緒")
    
    # 步驟 2：資料前處理
    print("\n▶ 步驟 2：執行資料前處理")
    print("-"*80)
    try:
        from data_preprocess import preprocess_data
        X_train, y_train, X_test, test_accounts, feature_names = preprocess_data()
        print(f"✓ 訓練集樣本數：{len(X_train):,}")
        print(f"✓ 測試集樣本數：{len(X_test):,}")
        print(f"✓ 特徵數量：{len(feature_names)}")
    except Exception as e:
        print(f"❌ 前處理失敗：{str(e)}")
        sys.exit(1)
    
    # 步驟 3：模型訓練
    print("\n▶ 步驟 3：訓練模型")
    print("-"*80)
    try:
        from model_train import train_model
        model, cv_scores = train_model(X_train, y_train, feature_names)
        print(f"✓ 模型訓練完成")
        print(f"✓ 交叉驗證 F1 Score：{cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    except Exception as e:
        print(f"❌ 模型訓練失敗：{str(e)}")
        sys.exit(1)
    
    # 步驟 4：生成預測
    print("\n▶ 步驟 4：生成預測結果")
    print("-"*80)
    try:
        from model_predict import generate_predictions
        submission_files = generate_predictions(model, X_test, test_accounts)
        print(f"✓ 成功生成 {len(submission_files)} 個預測檔案：")
        for file in submission_files:
            print(f"   - {file}")
    except Exception as e:
        print(f"❌ 預測生成失敗：{str(e)}")
        sys.exit(1)
    
    # 完成
    print_header("執行完成！")
    print(f"""
    結果摘要：
    - 訓練樣本：{len(X_train):,} 筆
    - 測試樣本：{len(X_test):,} 筆
    - 特徵數量：{len(feature_names)} 個
    - CV F1 Score：{cv_scores.mean():.4f}
    - 生成檔案：{len(submission_files)} 個
    
    1. submission_threshold_60.csv
    2. submission_threshold_50.csv
    3. 根據分數調整閾值
    """)
    
    print("執行結束時間：" + time.strftime("%Y-%m-%d %H:%M:%S"))
    print("="*80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  程式被使用者中斷")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ 發生未預期的錯誤：{str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)