import numpy as np
import pandas as pd
import pylab
import matplotlib.pyplot as plt

def data_read():
    train_data = pd.read_csv("train.csv").values
    test_data = pd.read_csv("test.csv").values

    return train_data, test_data
def compute_error(w, b, data):
    totalError = 0
    x = data[:,0]
    y = data[:,1]
    for i in range(len(data)):
        totalError += (y[i] - w*x[i] - b)**2
    return totalError/len(data)
def optimizer(data, starting_b, starting_w, learning_rate, num_iter):
    '''
    优化器用于进行梯度下降迭代
    starting_b: 初始偏移
    starting_m: 初始权重
    learning_rate: 学习率
    num_iter: 迭代次数
    '''
    w = starting_w
    b = starting_b
    #存cost值，方便图示下降过程
    cost_list = []
    for i in range(num_iter):
        cost_list.append(compute_error(w, b, data))
        w, b = compute_gradient(w, b , data, learning_rate)

    return w, b, cost_list



    return
def compute_gradient(current_w, current_b, data, learing_rate):
    sum_grad_b = 0
    sum_grade_w = 0
    for i in range(len(data)):
        x = data[i, 0]
        y = data[i, 1]
        sum_grade_w += (y - current_w * x - current_b) * (-x)
        sum_grad_b += current_w * x + current_b - y

    #求梯度
    grad_w = 2 / len(data) * sum_grade_w
    gard_b = 2 / len(data) * sum_grad_b

    #梯度更新
    update_w = current_w - learing_rate * grad_w
    update_b = current_b - learing_rate * gard_b

    return update_w, update_b

    return
def plot_data(data, w, b, cost_list):

    plt.scatter(data[:,0], data[:,1], s=50)
    plt.plot(data[:,0], w * data[:,0]+b)
    plt.plot(cost_list, 'r')
    plt.show()
    return
def linear_regression():
    # 初始化超参数
    learning_rate = 0.0008
    starting_b = 0
    starting_w = 0
    num_iter = 40

    # 读取数据
    train_data, test_data = data_read()

    #开始梯度下降求w,b,loss
    w, b, cost_list = optimizer(train_data, starting_b, starting_w, learning_rate, num_iter)
    loss = compute_error(w, b ,train_data)
    print("w is {}, b is {}, loss is {}".format(w, b, loss))
    valid(w*train_data[:, 0] + b, train_data[:, 1])
    #输出测试集结果
    y_pre_test = w * test_data - b
    save_to_csv(y_pre_test, "y_pre_test.csv")
    plot_data(train_data, w, b, cost_list)
    return


def save_to_csv(data, outpath):
    with open(outpath, "w+") as f:
        f.write(str(data))
    return print("结果保存成功")


def valid(y, y_true):
    correct_num = 0
    for i, j in zip(y, y_true):
        if round(i,1)==round(j, 1):
            correct_num += 1

    print("模型正确率：{}%".format(correct_num/len(y) * 100))



if __name__ == '__main__':
    linear_regression()
