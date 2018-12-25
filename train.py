import init
import numpy as np

training_n = init.n
# 导入 Mnist 数据
training_data_file = open("mnist_dataset/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# 训练
# 设置迭代次数
epochs = 5
print("Training start!")
for e in range(epochs):
    if e > 0:
        print('\n')
    print(e)
    i = 0
    j = 0
    for record in training_data_list:
        # 删除 ','
        all_values = record.split(',')
        # 归一化输入
        inputs = (np.asfarray(all_values[1:]) / 255 * 0.99) + 0.01
        # 设置真实值（期望节点输出0.99，其余节点输出0.01
        targets = np.zeros(init.output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        training_n.train(inputs, targets)
        i += 1
        if i > 200:
            print('*', end='')
            i = 0
            j += 1
        if j > 100:
            print('\n')
            j = 0
print("Training OK!")
trarned_n = training_n
