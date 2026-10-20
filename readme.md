# Predicting Diabetes Web App (Streamlit)

這是一個使用 **Streamlit** 與 **Scikit-learn** 建構的糖尿病預測互動式 Web 應用程式。使用者可以在側邊欄即時調整模型參數、切換不同的機器學習演算法，並即時視覺化評估指標（如混淆矩陣、ROC 曲線與 PR 曲線）。

---

## 📌 功能特色

- **多種分類演算法切換**：
  - **Decision Tree (決策樹)**：可自訂分割品質依據 (`criterion`) 與節點分割策略 (`splitter`)。
  - **Logistic Regression (邏輯迴歸)**：可自訂正則化參數 (`C`) 與最大迭代次數 (`max_iter`)。
  - **Random Forest (隨機森林)**：可自訂樹木數量 (`n_estimators`)、最大樹深 (`max_depth`) 與自助抽樣 (`bootstrap`)。
- **動態評估指標視覺化**：
  - 混淆矩陣 (Confusion Matrix)
  - 接收者操作特徵曲線 (ROC Curve & AUC)
  - 精確率-召回率曲線 (Precision-Recall Curve)
- **模型成效數據**：即時輸出準確率 (Accuracy)、精確率 (Precision) 與召回率 (Recall)。
- **資料檢視**：側邊欄提供查看原始糖尿病數據集 (`diabetes.csv`) 的選項。

---

## 🛠️ 技術與套件需求

- **Python**: 3.8+
- 主要依賴套件：
  - `streamlit`
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`

---

## 🚀 快速開始

### 1. 複製專案庫 (Clone Repository)

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. 建立並啟動虛擬環境 (建議)

- **macOS / Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
- **Windows**:
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```

### 3. 安裝依賴套件

若專案中有提供 `requirements.txt`：
```bash
pip install -r requirements.txt
```

*(或者手動安裝指定套件)*：
```bash
pip install streamlit pandas numpy scikit-learn matplotlib
```

### 4. 準備資料集

請確保專案根目錄下有名為 **`diabetes.csv`** 的資料集（例如常用的 *Pima Indians Diabetes Dataset*），且包含目標欄位 `Outcome`。

### 5. 啟動 Streamlit 應用程式

```bash
streamlit run test.py
```

執行後，終端機將會顯示本地網址（通常為 `http://localhost:8501`），點擊即可在瀏覽器中操作應用。

---

## 📂 專案結構範例

```text
├── diabetes.csv          # 糖尿病資料集
├── test.py               # Streamlit 核心程式碼
├── requirements.txt      # 相依套件清單 (可選)
└── README.md             # 專案說明文件
```

---

## 📄 授權條款 (License)

本專案採用 [MIT License](LICENSE) 進行授權。