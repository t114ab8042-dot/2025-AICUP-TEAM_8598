# 2025 玉山人工智慧公開挑戰賽 - 初賽提交

## 專案簡介
本專案為「2025 玉山人工智慧公開挑戰賽」初賽提交的完整解決方案。任務目標為識別潛在的警示帳戶，透過分析帳戶交易行為特徵，建立機器學習模型進行預測。

## 使用資料來源
- **主辦方提供資料**：
  - `acct_transaction.csv`：帳戶交易資料
  - `acct_alert.csv`：警示帳戶標籤資料
  - `acct_predict.csv`：待預測帳戶清單

## 模型與方法概述
### 核心策略
1. **智能抽樣**：只選擇活躍帳戶（交易數≥20筆）作為訓練集的正常帳戶樣本，確保訓練集與測試集分布一致
2. **增強特徵工程**：設計33個特徵，包含：
   - 跨行交易特徵（5個）
   - 快速轉出特徵（3個）
   - 收款特徵（5個）
   - 收支平衡特徵（5個）
   - 匯出特徵（3個）
   - 交易量特徵（2個）
   - 時間窗口特徵（3個）
   - 相對活躍度特徵（3個）
   - 其他統計特徵（4個）
3. **模型選擇**：XGBoost 分類器
4. **不平衡處理**：SMOTE 過採樣技術（sampling_strategy=0.2）
5. **多閾值策略**：生成多個預測閾值的結果供選擇

### 模型架構
- **預處理**：資料清洗、時間格式轉換、特徵計算
- **特徵標準化**：RobustScaler
- **模型訓練**：XGBoost with 5-fold Stratified Cross Validation
- **後處理**：機率轉換為二元標籤（可調整閾值）

## 執行環境
- **Python 版本**：Python 3.13.5
- **作業系統**：Ubuntu 20.04+ / Windows 10+ / macOS 11+
- **記憶體需求**：至少 8GB RAM
- **儲存空間**：至少 2GB 可用空間

## 專案結構說明
```
.
├── Preprocess/              # 資料前處理模組
│   ├── data_preprocess.py  # 資料前處理主程式
│   └── README.md           # 前處理說明文件
├── Model/                   # 模型訓練與預測模組
│   ├── model_train.py      # 模型訓練程式
│   ├── model_predict.py    # 模型預測程式
│   └── README.md           # 模型說明文件
├── main.py                  # 主執行程式
├── requirements.txt         # 套件依賴清單
└── README.md               # 本文件
```

## 可復現流程

### 步驟 1：環境設置
```bash
# 建立虛擬環境（建議）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安裝依賴套件
pip install -r requirements.txt
```

### 步驟 2：準備資料
將主辦方提供的資料檔案放置於專案根目錄：
- `acct_transaction.csv`
- `acct_alert.csv`
- `acct_predict.csv`

### 步驟 3：執行完整流程
```bash
# 一鍵執行完整流程（預處理 + 訓練 + 預測）
python main.py

# 或分步驟執行：
# 1. 資料前處理
python Preprocess/data_preprocess.py

# 2. 模型訓練
python Model/model_train.py

# 3. 生成預測
python Model/model_predict.py
```

### 步驟 4：查看結果
執行完成後會生成多個不同閾值的預測檔案：
- `submission_threshold_30.csv`（閾值=0.30）
- `submission_threshold_40.csv`（閾值=0.40）
- `submission_threshold_50.csv`（閾值=0.50）
- `submission_threshold_60.csv`（閾值=0.60，推薦）
- `submission_threshold_70.csv`（閾值=0.70）

## 超參數設定
### XGBoost 超參數
```python
{
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
    'n_jobs': -1
}
```

### 資料處理參數
- **活躍帳戶門檻**：20筆交易
- **訓練集抽樣比例**：10:1（正常:警示）
- **SMOTE比例**：0.2
- **交叉驗證折數**：5

## 評估方式與實驗結果
### 評估指標
- **主要指標**：F1 Score
- **評估方法**：5-Fold Stratified Cross Validation

### 實驗結果
- **交叉驗證 F1 Score**：0.4852 (±0.0156)
- **各 Fold 詳細結果**：
  - Fold 1: 0.4721
  - Fold 2: 0.4958
  - Fold 3: 0.4843
  - Fold 4: 0.5012
  - Fold 5: 0.4726

### 特徵重要性 Top 10
1. `relative_activity` - 相對活躍度
2. `from_cross_ratio` - 匯出跨行比例
3. `to_cross_ratio` - 收款跨行比例
4. `total_cross_ratio` - 總跨行比例
5. `same_day_inout_ratio` - 當日進出比例
6. `inout_ratio` - 收支比例
7. `balance_score` - 平衡分數
8. `to_unique_senders` - 唯一匯款人數
9. `last30d_cross_ratio` - 近30天跨行比例
10. `txn_diversity` - 交易多樣性

### 預測結果統計（閾值=0.60）
- 預測警示帳戶數：68 個（1.42%）
- 總預測帳戶數：4,780 個

## 團隊成員資訊
- **隊名**：TEAM_8598
- **隊長**：李泓泯
- **成員**：葉千熏、賴泓瑋

## 授權
本專案僅供 2025 玉山人工智慧公開挑戰賽使用。

---