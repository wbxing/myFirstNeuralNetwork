import numpy as np
import scipy.special as ssp  # 引入激活函数


class NeuralNetwork:

    # 初始化神经网络
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):

        # 设置节点数目
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # 设置学习率
        self.lr = learningrate

        # 设置权重，w_i_j 表示节点 i 到下一层节点 j 的权重
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # 设置激活函数
        self.activation_function = lambda x: ssp.expit(x)

    # 训练神经网络
    def train(self, inputs_list, targets_list):
        # 原始输入转换为矩阵
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # 计算输入到隐藏层的信号
        hidden_inputs = np.dot(self.wih, inputs)
        # 计算隐藏层输出的信号
        hidden_outputs = self.activation_function(hidden_inputs)

        # 计算输入到输出层的信号
        final_inputs = np.dot(self.who, hidden_outputs)
        # 计算输出层输出的信号
        final_outputs = self.activation_function(final_inputs)

        # 输出误差是 (target - actual)
        output_errors = targets - final_outputs
        # 隐藏层误差 就是对 output_errors 按权重进行分割，之后在隐藏层节点重新组合
        hidden_errors = np.dot(self.who.T, output_errors)

        # 更新隐藏层和输出层之间的权重
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                     np.transpose(hidden_outputs))
        # 更新输入层和隐藏层之间的权重
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                     np.transpose(inputs))

    # 查询神经网络
    def query(self, inputs_list):

        # 原始输入转换为矩阵
        inputs = np.array(inputs_list, ndmin=2).T

        # 计算输入到隐藏层的信号
        hidden_inputs = np.dot(self.wih, inputs)
        # 计算隐藏层输出的信号
        hidden_outputs = self.activation_function(hidden_inputs)

        # 计算输入到输出层的信号
        final_inputs = np.dot(self.who, hidden_outputs)
        # 计算输出层输出的信号
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
