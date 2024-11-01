
from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__)

# 加載模型和 tokenizer
model = BertForSequenceClassification.from_pretrained("./employee_feedback_model")
tokenizer = BertTokenizer.from_pretrained("./employee_feedback_tokenizer")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data["text"]
    
    # 將文本轉為模型輸入格式
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    
    # 獲取預測結果
    probs = torch.softmax(outputs.logits, dim=1)
    label = torch.argmax(probs, dim=1).item()
    label_mapping = {0: "負面", 1: "中立", 2: "正面"}
    
    result = {
        "text": text,
        "label": label_mapping[label],
        "confidence": probs[0][label].item()
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
