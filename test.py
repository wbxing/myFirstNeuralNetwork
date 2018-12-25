import train
import numpy as np

test_n = train.trarned_n
# 读取测试集
test_data_file = open("mnist_dataset/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# 创建得分数组
scorecard = []
print("Testing start!")
i = 1
for record in test_data_list:
    # 去除 ','
    all_values = record.split(',')
    # 正确答案是第一个数
    correct_label = int(all_values[0])
    # 归一化输入
    inputs = (np.asfarray(all_values[1:]) / 255 * 0.99) + 0.01
    # 查询神经网络
    # outputs = test_n.query(inputs)
    outputs = test_n.query(inputs)
    # 取最大的值的下标
    label = np.argmax(outputs)

    print(i, ":", label, correct_label)
    # 比较
    if label == correct_label:
        scorecard.append(1)
    else:
        scorecard.append(0)
    i += 1

# 计算最终得分
scorecard_array = np.asfarray(scorecard)
print("performance = ", scorecard_array.sum() / scorecard_array.size)
print("Testing OK!")

final_nn = test_n

final_inodes = final_nn.inodes
final_hnodes = final_nn.hnodes
final_onodes = final_nn.onodes
final_wih = final_nn.wih
final_who = final_nn.who
print(final_inodes, '\n', final_hnodes, '\n', final_onodes, '\n', final_wih, '\n', final_who)

