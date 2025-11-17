# 模型訓練與預測模組

## 模組說明
本模組負責機器學習模型的訓練、評估和預測生成。

## 檔案說明
- `model_train.py`：模型訓練主程式
- `model_predict.py`：預測生成程式

## 模型架構

### 使用演算法
**XGBoost (eXtreme Gradient Boosting)**
- 選擇原因：
  - 處理不平衡資料表現優異
  - 特徵重要性分析清晰
  - 訓練速度快且效果穩定
  - 支援正則化避免過擬合

### 超參數設定
```python
{
    'max_depth': 6,              # 樹的最大深度
    'learning_rate': 0.1,         # 學習率
    'n_estimators': 300,          # 樹的數量
    'min_child_weight': 20,       # 最小子節點權重
    'gamma': 0.5,                 # 分裂所需最小損失減少
    'subsample': 0.8,             # 樣本採樣比例
    'colsample_bytree': 0.8,      # 特徵採樣比例
    'reg_alpha': 3.0,             # L1 正則化
    'reg_lambda': 5.0,            # L2 正則化
    'scale_pos_weight': 動態計算   # 類別權重平衡
}
```

## 不平衡資料處理

### SMOTE (Synthetic Minority Over-sampling Technique)
- **採樣策略**：0.2（將少數類別擴充至多數類別的20%）
- **效果**：改善模型對警示帳戶的識別能力
- **原始比例**：約 333:1（正常:警示）
- **SMOTE後**：約 5:1（正常:警示）

## 模型評估

### 交叉驗證策略
- **方法**：5-Fold Stratified Cross Validation
- **評估指標**：F1 Score
- **分層抽樣**：確保每個 fold 中類別比例一致

### 性能指標
- **主要指標**：F1 Score（調和平均數）
- **輔助指標**：
  - Precision（精確率）
  - Recall（召回率）
  - Confusion Matrix（混淆矩陣）

## 預測生成

### 多閾值策略
生成5個不同閾值的預測結果：
- **0.30**：激進策略（預測較多警示帳戶）
- **0.40**：較激進策略
- **0.50**：基準策略
- **0.60**：推薦策略（平衡精確率與召回率）
- **0.70**：保守策略（預測較少警示帳戶）

### 閾值選擇建議
基於歷史資料分析：
- 歷史警示比例：0.3%
- 預期測試集警示比例：0.5-2.0%
- 推薦閾值：0.60

## 使用方式

### 模型訓練
```bash
# 獨立執行
python Model/model_train.py

# 或在主程式中引用
from model_train import train_model
model, cv_scores = train_model(X_train, y_train, feature_names)
```

### 預測生成
```bash
# 獨立執行
python Model/model_predict.py

# 或在主程式中引用
from model_predict import generate_predictions
submission_files = generate_predictions(model, X_test, test_accounts)
```

## 輸出檔案

### 訓練階段
- `xgboost_model.pkl`：訓練好的模型
- `feature_importance.csv`：特徵重要性排序

### 預測階段
- `submission_threshold_30.csv`：閾值0.30的預測結果
- `submission_threshold_40.csv`：閾值0.40的預測結果
- `submission_threshold_50.csv`：閾值0.50的預測結果
- `submission_threshold_60.csv`：閾值0.60的預測結果（推薦）
- `submission_threshold_70.csv`：閾值0.70的預測結果

## 特徵重要性（Top 10）

根據訓練結果，最重要的特徵為：
1. `relative_activity`：相對活躍度
2. `from_cross_ratio`：匯出跨行比例
3. `to_cross_ratio`：收款跨行比例
4. `total_cross_ratio`：總跨行比例
5. `same_day_inout_ratio`：當日進出比例
6. `inout_ratio`：收支比例
7. `balance_score`：平衡分數
8. `to_unique_senders`：唯一匯款人數
9. `last30d_cross_ratio`：近30天跨行比例
10. `txn_diversity`：交易多樣性

## 注意事項

1. **記憶體需求**：訓練過程需要至少 4GB RAM
2. **執行順序**：必須先執行資料前處理，再執行模型訓練
3. **檔案依賴**：
   - 訓練需要：前處理後的資料
   - 預測需要：訓練好的模型檔案
4. **執行時間**：
   - 訓練：約 2-5 分鐘
   - 預測：約 30 秒

## 調參建議

如需進一步優化模型，可調整以下參數：
- **增加 n_estimators**：提升模型複雜度（可能過擬合）
- **降低 learning_rate**：更細緻的學習（需要更多樹）
- **調整 max_depth**：控制模型複雜度
- **修改 SMOTE 比例**：改變類別平衡程度
- **調整閾值**：根據業務需求平衡精確率與召回率