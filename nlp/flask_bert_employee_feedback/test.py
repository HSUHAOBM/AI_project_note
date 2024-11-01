import requests

url = "http://localhost:5000/predict"
data = {"text": "很慘的經歷"}

response = requests.post(url, json=data)
print(response.json())
