import neuralNetwork as nN


# 设置节点数目
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# 设置学习率
learning_rate = 0.1

# 创建神经网络实例
n = nN.NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
