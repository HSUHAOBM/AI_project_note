import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from datasets import Dataset

# 初始化 BERT tokenizer 和 model
# 使用預訓練的 BERT 基礎模型，"bert-base-uncased" 表示不區分大小寫的英文模型
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=3)  # num_labels=3 表示分類為三個類別

# 數據範例，這裡的數據為員工反饋及其標籤
data = [
    {"text": "工作環境非常好，合作愉快", "label": "正面"},
    {"text": "經常加班讓人疲憊不堪", "label": "負面"},
    {"text": "辦公室環境一般", "label": "中立"},
    {"text": "同事之間相處融洽，工作氛圍輕鬆", "label": "正面"},
    {"text": "項目壓力大，長時間無法休息", "label": "負面"},
    {"text": "公司的福利制度不錯", "label": "正面"},
    {"text": "每天的通勤時間很長，有些疲憊", "label": "負面"},
    {"text": "工作內容單調無聊", "label": "中立"},
    {"text": "工作非常具有挑戰性，很有成就感", "label": "正面"},
    {"text": "上級領導不夠關心員工的需求", "label": "負面"},
    {"text": "薪資待遇在業界處於中等水平", "label": "中立"},
    {"text": "辦公室設備齊全，環境舒適", "label": "正面"},
    {"text": "晉升機會有限，未來發展堪憂", "label": "負面"},
    {"text": "工作時間靈活，可以自由安排", "label": "正面"},
    {"text": "員工之間缺乏有效的溝通", "label": "負面"},
    {"text": "上下班時間固定，符合預期", "label": "中立"},
    {"text": "公司有很多學習和成長的機會", "label": "正面"},
    {"text": "工作地點離家遠，通勤不便", "label": "負面"},
    {"text": "工作職責明確，但缺乏創新空間", "label": "中立"},
    {"text": "團隊合作良好，經常互相支持", "label": "正面"}
]


# 數據標準化函數：用 tokenizer 將文本轉換為模型可處理的格式
def tokenize_function(example):
    # truncation=True 表示若文本過長會自動截斷
    return tokenizer(example["text"], truncation=True)


# 將數據轉為 Dataset 格式
# 首先使用 train_test_split 將數據劃分為訓練集和驗證集（80%訓練，20%驗證）
train_texts, val_texts = train_test_split(data, test_size=0.2)

# 創建訓練和驗證數據集並進行標籤轉換
train_dataset = Dataset.from_dict({
    "text": [d["text"] for d in train_texts],
    "label": [0 if d["label"] == "負面" else 1 if d["label"] == "中立" else 2 for d in train_texts]
})
val_dataset = Dataset.from_dict({
    "text": [d["text"] for d in val_texts],
    "label": [0 if d["label"] == "負面" else 1 if d["label"] == "中立" else 2 for d in val_texts]
})

# 將文本轉換為 token 格式，batch 處理可提升效率
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# 訓練設定：TrainingArguments 定義模型的訓練配置
training_args = TrainingArguments(
    output_dir="./results",             # 訓練過程中生成的檔案將保存在這個目錄中
    evaluation_strategy="epoch",        # 每個訓練 epoch 結束後進行一次評估
    learning_rate=2e-5,                 # 模型的學習率，通常設定為較小的值以穩定訓練
    per_device_train_batch_size=8,      # 訓練時每個 GPU/CPU 上的批次大小
    per_device_eval_batch_size=8,       # 評估時每個 GPU/CPU 上的批次大小
    num_train_epochs=3,                 # 訓練輪數，即模型將遍歷整個訓練數據集的次數
    weight_decay=0.01,                  # 權重衰減率，防止過擬合的正則化參數
)

# 訓練器初始化：Trainer 用於管理訓練和評估過程
trainer = Trainer(
    model=model,                        # 要訓練的模型
    args=training_args,                 # 訓練參數（例如批次大小、學習率等）
    train_dataset=train_dataset,        # 訓練數據集
    eval_dataset=val_dataset,           # 驗證數據集
    tokenizer=tokenizer,                # 使用的 tokenizer
    data_collator=DataCollatorWithPadding(
        tokenizer=tokenizer),  # 自動對 batch 進行填充，使其相同長度
)

# 訓練模型
trainer.train()

# 儲存模型和 tokenizer，方便之後加載模型進行推理
model.save_pretrained("./employee_feedback_model")
tokenizer.save_pretrained("./employee_feedback_tokenizer")
