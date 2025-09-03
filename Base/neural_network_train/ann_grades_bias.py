# https://www.youtube.com/watch?v=_PrTJcgKrdk
# csv
import csv
import math

# 1. 讀取 grades-bias.csv，取得每筆資料的 x（時間）和 y（成績）。

read_csv = csv.reader(open('grades-bias.csv'))

x = []  # 輸入：時間
y = []  # 輸出：成績
for row in read_csv:
    x.append(float(row[0]))
    y.append(float(row[1]))

# 2. 設定初始權重（w）和偏差（b），以及學習率（lr）。
# 建立模型 Y=W*X + B*1
lr = 0.001  # 學習率
w = 2  # 隨機一個"權重"
b = 8  # 隨機一個"偏差"

# # 根據第一筆資料做訓練
# output = w * x[0] + b * 1  # 預測的成績
# error = y[0] - output  # 誤差
# gradinet_w = error * x[0]  # 計算權重的梯度 固定的輸入值
# gradinet_b = error * 1  # 計算偏差的梯度 固定的輸入值
# w = w + gradinet_w * lr  # 更新權重
# b = b + gradinet_b * lr  # 更新偏差
# print(f"第一筆 預測的成績：{output}, 誤差：{error}, 更新後的權重：{w}, 更新後的偏差：{b}")

# # 根據第二筆資料做訓練
# output = w * x[1] + b * 1  # 預測的成績
# error = y[1] - output  # 誤差
# gradinet_w = error * x[1]  # 計算權重的梯度 固定的輸入值
# gradinet_b = error * 1  # 計算偏差的梯度 固定的輸入值
# w = w + gradinet_w * lr  # 更新權重
# b = b + gradinet_b * lr  # 更新偏差
# print(f"第二筆 預測的成績：{output}, 誤差：{error}, 更新後的權重：{w}, 更新後的偏差：{b}")

# 3. 用梯度下降法（每筆資料都更新 w 和 b）進行多次訓練（epoch）。
# 迴圈訓練
len(x)  # 總共有幾筆資料
epoch = 10  # 訓練次數
for i in range(epoch):
    for j in range(len(x)):
        output = w * x[j] + b * 1  # 預測的成績
        error = y[j]-output  # 誤差
        gradinet_w = error * x[j]  # 計算權重的梯度
        gradinet_b = error * 1  # 計算偏差的梯度
        w = w+gradinet_w * lr  # 更新權重
        b = b+gradinet_b * lr
        print(
            f"第{i+1}次訓練 第{j+1}筆資料 預測的成績：{output}, 誤差：{error}, 更新後的權重：{w}, 更新後的偏差：{b}")
print(f"訓練完成，最終的權重：{w}, 最終的偏差：{b}")

# # 使用真實資料 評估模型 MSE(Mean Squared Error) 均方誤差
total_error = 0
for i in range(len(x)):
    output = w * x[i] + b * 1  # 預測的成績
    error = y[i] - output  # 誤差
    total_error += error ** 2  # 計算平方誤差
print("Mean Squared error:", total_error / len(x))  # 計算均方誤差 開根號
print("開根號:", math.sqrt(total_error / len(x)))  # 計算均方誤差 開根號


# 測試資料
test_x = 5  # 測試資料：時間
test_output = w * test_x + b * 1  # 預測的成績
print(f"測試資料：{test_x}，預測的成績：{test_output}")
