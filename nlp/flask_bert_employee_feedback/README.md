
# 員工反饋情感分析

此專案使用 BERT 模型進行員工反饋的情感分析，將反饋分類為三個類別：正面、中立和負面。專案包含兩個主要組件：
- `train_model.py`：用於訓練基於 BERT 的情感分類模型。
- `app.py`：使用 Flask 部署訓練好的模型，並提供 API 服務。

## 環境需求

請按照以下步驟設置虛擬環境：

1. 創建新的虛擬環境（推薦此方式以避免依賴性衝突）：
   ```bash
   conda create -n nlp_env python=3.8
   conda activate nlp_env
   ```

2. 安裝所需的相依套件：
   ```bash
   pip install transformers torch datasets scikit-learn flask
   ```

## 檔案說明

- **`train_model.py`**：此腳本用於訓練 BERT 模型，進行文本情感分類。腳本會加載數據、處理文本、訓練模型並將模型儲存至 `./employee_feedback_model` 資料夾中。

- **`app.py`**：此腳本用於啟動 Flask API 服務，載入訓練好的模型並提供預測 API。該 API 接收員工反饋文本，並返回文本的情感分類及置信度。

## 資料夾說明

### `employee_feedback_model`
- 此資料夾儲存訓練完成的 BERT 模型文件。這些文件包括模型的架構和訓練後的權重。
- 使用此資料夾中的模型文件，可以在推理階段（inference）加載模型並進行情感分類。
- **主要文件**：
  - `pytorch_model.bin` 或 `model.safetensors`：存放模型的權重。
  - `config.json`：包含模型的結構設定，例如隱藏層單元數量、層數等配置。

### `employee_feedback_tokenizer`
- 此資料夾包含 BERT 模型所使用的 tokenizer 文件，用來將文本轉換成模型可理解的數字格式（即 token）。
- 在推理階段，必須使用相同的 tokenizer 配置來確保文本的處理方式與訓練時一致。
- **主要文件**：
  - `vocab.txt` 或 `tokenizer.json`：包含詞彙表或 tokenizer 設定，用來將文字轉為 token ID。
  - `tokenizer_config.json`：描述 tokenizer 的詳細配置，例如最大詞長、特殊符號等設定。

這兩個資料夾必須在推理時保留，因為它們確保了模型的輸入格式和架構與訓練時完全一致。

## 使用說明

### 1. 訓練模型

在命令行中運行以下命令來訓練模型：

```bash
python train_model.py
```

這將會訓練情感分類模型並將模型和 tokenizer 儲存至 `./employee_feedback_model` 和 `./employee_feedback_tokenizer` 資料夾中。

### 2. 啟動 Flask API 服務

訓練完成後，可以運行以下命令來啟動 API 服務：

```bash
python app.py
```

服務啟動後，API 將會在 `localhost:5000` 運行。

### 3. 測試 API

可以使用 [Postman](https://www.postman.com/) 或 `curl` 來測試 API。以下是使用 `curl` 發送測試請求的範例：

```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{"text": "這是一次非常棒的合作經歷"}"
```

或在 Postman 中設置：
- **URL**：`http://localhost:5000/predict`
- **請求方式**：`POST`
- **Body**：選擇 `raw` 和 `JSON`，並輸入測試數據：
  ```json
  {
    "text": "這是一次非常棒的合作經歷"
  }
  ```

範例的回應結果：
```json
{
  "text": "這是一次非常棒的合作經歷",
  "label": "正面",
  "confidence": 0.92
}
```
