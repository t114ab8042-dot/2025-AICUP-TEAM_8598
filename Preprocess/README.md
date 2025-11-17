# 資料前處理模組

## 模組說明
本模組負責原始資料的讀取、清洗、特徵工程和訓練集抽樣等前處理工作。

## 主要功能
1. **資料讀取**：載入交易資料、警示帳戶資料和待預測帳戶資料
2. **帳戶識別**：區分訓練集和測試集的帳戶
3. **活躍度計算**：統計每個帳戶的交易數量
4. **智能抽樣**：只選擇活躍帳戶作為訓練集的正常樣本
5. **特徵工程**：計算33個特徵
6. **資料標準化**：使用RobustScaler進行標準化

## 檔案說明
- `data_preprocess.py`：資料前處理主程式

## 特徵列表（33個）

### Group 1: 跨行交易特徵 (5個)
- `from_cross_ratio`：匯出跨行交易比例
- `from_cross_count`：匯出跨行交易數量
- `to_cross_ratio`：收款跨行交易比例
- `to_cross_count`：收款跨行交易數量
- `total_cross_ratio`：總跨行交易比例

### Group 2: 快速轉出特徵 (3個)
- `same_day_inout_ratio`：當日進出比例
- `same_day_count`：當日進出天數
- `balance_days`：收支平衡天數

### Group 3: 收款特徵 (5個)
- `to_count`：收款筆數
- `to_total_amt`：收款總金額
- `to_mean_amt`：平均收款金額
- `to_daily_avg`：日均收款筆數
- `to_unique_senders`：唯一匯款人數

### Group 4: 收支平衡特徵 (5個)
- `from_total_amt`：匯出總金額
- `inout_ratio`：收支比例
- `net_flow`：淨流入金額
- `balance_score`：平衡分數
- `is_balanced`：是否平衡

### Group 5: 匯出特徵 (3個)
- `from_count`：匯出筆數
- `from_mean_amt`：平均匯出金額
- `from_unique_receivers`：唯一收款人數

### Group 6: 交易量特徵 (2個)
- `total_count`：總交易筆數
- `txn_diversity`：交易多樣性

### Group 7: 時間窗口特徵 (3個)
- `last30d_count`：近30天交易筆數
- `last30d_cross_ratio`：近30天跨行比例
- `last30d_from_ratio`：近30天匯出比例

### Group 8: 其他統計特徵 (4個)
- `amt_std`：交易金額標準差
- `amt_max`：最大交易金額
- `active_days`：活躍天數
- `txn_per_day`：日均交易筆數

### Group 9: 相對活躍度特徵 (3個)
- `relative_activity`：相對活躍度
- `above_median`：是否高於中位數
- `activity_level`：活躍度等級 (1-4)

## 使用方式

### 獨立執行
```bash
python Preprocess/data_preprocess.py
```

### 模組引用
```python
from data_preprocess import preprocess_data

# 執行前處理
X_train, y_train, X_test, test_accounts, feature_names = preprocess_data()
```

## 輸出結果
- `X_train`：訓練集特徵矩陣
- `y_train`：訓練集標籤
- `X_test`：測試集特徵矩陣
- `test_accounts`：測試集帳戶列表
- `feature_names`：特徵名稱列表
- `scaler.pkl`：資料標準化器（儲存檔案）

## 參數設定
- **活躍帳戶門檻**：20筆交易
- **抽樣比例**：10:1（正常:警示）
- **隨機種子**：42
- **時間窗口**：30天

## 注意事項
1. 確保所有資料檔案都在專案根目錄
2. 記憶體需求：至少4GB
3. 執行時間：約1-3分鐘（視資料量而定）