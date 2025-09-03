# https://youtu.be/XaYL0O-nIOU?si=j9Z_TRHA-YBG_VG3
# csv
import csv
import math

# 1. 讀取 grades-multiple-inputs.csv，取得每筆資料的 x1（時間） x2(狀態) 和 y（成績）。
read_csv = csv.reader(open('grades-multiple-inputs.csv'))

x1 = []  # 輸入：時間
x2 = []  # 輸入：狀態
y = []  # 輸出：成績
for row in read_csv:
    x1.append(float(row[0]))
    x2.append(float(row[1]))
    y.append(float(row[2]))

# 2. 設定初始權重（w）和偏差（b），以及學習率（lr）。
# 建立模型 Y = W1*X1 + W2*X2 + B*1
lr = 0.001  # 學習率
w1 = 2  # 隨機一個"權重" 對應時間
w2 = 10  # 隨機一個"權重" 對應狀態
b = 8  # 隨機一個"偏差"

# # 根據第一筆資料做訓練
output = w1 * x1[0] + w2 * x2[0] + b * 1  # 預測的成績
error = y[0] - output  # 誤差
# 誤差歸因權重 梯度
gradinet_w1 = error * x1[0]  # 計算權重的梯度 固定的輸入值
gradinet_w2 = error * x2[0]  # 計算權重的梯度 固定的輸入值
gradinet_b = error * 1  # 計算偏差的梯度 固定的輸入值
w1 = w1 + gradinet_w1 * lr  # 更新權重
w2 = w2 + gradinet_w2 * lr  # 更新權重
b = b + gradinet_b * lr  # 更新偏差
print(f"第一筆 預測的成績：{output}, 誤差：{error}, 更新後的權重：w1={w1}, w2={w2}, 更新後的偏差：{b}")


# 3. 用梯度下降法（每筆資料都更新 w1 w2 和 b）進行多次訓練（epoch）。
# 迴圈訓練
len(x1)  # 總共有幾筆資料
epoch = 30  # 訓練次數
for i in range(epoch):
    for j in range(len(x1)):
        output = w1 * x1[j] + w2 * x2[j] + b * 1  # 預測的成績
        error = y[j]-output  # 誤差
        gradinet_w1 = error * x1[j]  # 計算權重的梯度
        gradinet_w2 = error * x2[j]  # 計算權重的梯度
        gradinet_b = error * 1  # 計算偏差的梯度
        w1 = w1 + gradinet_w1 * lr  # 更新權重
        w2 = w2 + gradinet_w2 * lr  # 更新權重
        b = b + gradinet_b * lr
        print(
            f"第{i+1}次訓練 第{j+1}筆資料 預測的成績：{output}, 誤差：{error}, 更新後的權重：w1={w1}, w2={w2}, 更新後的偏差：{b}")
print(f"訓練完成，最終的權重：w1={w1}, w2={w2}, 最終的偏差：{b}")

# 使用真實資料 評估模型 MSE(Mean Squared Error) 均方誤差
total_error = 0
for i in range(len(x1)):
    output = w1 * x1[i] + w2 * x2[i] + b * 1  # 預測的成績
    error = y[i] - output  # 誤差
    total_error += error ** 2  # 計算平方誤差
print("Mean Squared error:", total_error / len(x1))  # 計算均方誤差 MSE
print("開根號:", math.sqrt(total_error / len(x1)))  # 計算均方誤差 開根號 RMSE


# 測試資料
test_x = 5  # 測試資料：時間
test_x2 = 1  # 測試資料：狀態
test_output = w1 * test_x + w2 * test_x2 + b * 1  # 預測的成績
print(f"測試資料讀書時間：{test_x}，測試資料狀態：{test_x2}，預測的成績：{test_output}")
