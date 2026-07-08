# 使用 Restful API 與 Ajax 來進行機器學習模型訓練與預測

在原本的 Titanic 乘客資料 CRUD（RESTful API + 原生 Ajax）專案基礎上，新增「模型訓練」與「乘客生存預測」兩個頁面。

## 安裝套件
（可用 `pip list` 或 `conda list` 檢視實際版本）
- Flask==3.1.3
- pandas==3.0.3
- numpy==2.4.6
- scikit-learn==1.8.0
- joblib==1.5.3

```bash
pip install -r ../requirements.txt
```

## 執行方法
```bash
python init_db.py
python app.py
```

啟動後開啟瀏覽器造訪 `http://127.0.0.1:5000/`，首頁 toolbar 有「模型訓練」「乘客生存預測」兩個連結。

## 說明

### 頁面與 API
- `/ml/train`：模型訓練頁面。按下「開始訓練」後，後端會在**背景執行緒**執行訓練，前端每 1.5 秒輪詢 `GET /api/ml/train/status` 顯示「訓練中 / 已完成 / 失敗」，藉此觀察訓練是否完成。伺服器重啟後也會自動讀回上次訓練的結果與模型。
- `/ml/predict`：乘客生存預測頁面，分為「單筆輸入預測」與「CSV 批次上傳預測」兩個區塊，皆透過 Ajax（`fetch`）呼叫後端 API，並在頁面即時顯示預測結果（是否存活、生存機率），不會整頁重新整理。

### 模型與超參數調整
- 訓練資料：直接從 `my_db.db` 的 `titanic` 資料表讀取（`SELECT * FROM titanic`），共 891 筆。
- 特徵欄位：`Pclass, Sex, Age, SibSp, Parch, Fare, Embarked`；前處理用 `ColumnTransformer`：
  - 數值欄位 `Age, Fare`：以中位數補缺值
  - 類別欄位 `Sex, Embarked`：以眾數補缺值後做 One-Hot Encoding
- 一鍵訓練會**同時**對兩種模型各自用 `GridSearchCV`（5-fold 交叉驗證）調整超參數，最後挑選在測試集（20% holdout）準確率較高的一組整體最佳模型儲存：
  - **RandomForestClassifier**：調整 `n_estimators`（100/200/300）、`max_depth`（None/5/10）、`min_samples_split`（2/5/10），共 27 組
  - **LogisticRegression**：調整 `C`（0.01/0.1/1/10）、`class_weight`（None/balanced），共 8 組

以某次實際訓練結果為例：

| 模型 | 最佳超參數 | CV 準確率 | 測試集準確率 |
|---|---|---|---|
| RandomForest（最佳，已儲存） | `max_depth=10, min_samples_split=10, n_estimators=200` | 82.59% | 80.45% |
| LogisticRegression | `C=0.1, class_weight=None` | 79.92% | 79.33% |

（每次訓練資料切分與 GridSearchCV 結果可能略有差異，實際數字以頁面上顯示的為準。）

### 模型儲存與載入
訓練完成後使用 `joblib` 把整個 `Pipeline`（前處理 + 分類器）存到 repo 根目錄的 `models/titanic_model.joblib`，超參數與準確率等資訊另外存成 `models/model_meta.json`：

```python
joblib.dump(best_pipeline, "models/titanic_model.joblib")
model = joblib.load("models/titanic_model.joblib")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)
```

### 預測功能
- 單筆預測：`POST /api/ml/predict`，輸入 Pclass/Sex/Age/SibSp/Parch/Fare/Embarked，回傳 `survived`（0/1）與 `probability`（生存機率）。
- 批次預測：`POST /api/ml/predict/batch`，上傳 CSV（需包含上述特徵欄位，PassengerId/Name 可有可無方便對照），後端逐列預測後回傳結果，前端直接 render 成表格顯示。

## 成果
![模型訓練頁面](請貼上訓練頁面完成後的截圖)
![乘客生存預測頁面](請貼上預測頁面的截圖)

[Titanic RESTful API + Ajax 機器學習功能展示影片](請貼上你的 YouTube 或 Google Drive 公開影片連結)

## 其它你想要補充的資訊
- 兩個候選模型的比較、GridSearchCV 的參數網格都定義在 `app.py` 的 `MODEL_CONFIGS`，可依需求調整參數範圍或改用其它演算法。
- `models/` 資料夾內容由程式自動產生，不需手動建立；`titanic_model.joblib` 約 1.9 MB，未超過 GitHub 一般檔案大小限制，未使用 Git LFS。
