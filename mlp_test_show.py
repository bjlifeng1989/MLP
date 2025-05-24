# -*- coding = utf-8 -*-
# @Time : 2025/4/10 13:23
# @Author: Vast
# @File: mlp_test_show.py
# @Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np
from mlp import MLP  # 从mlp模块导入MLP类
import os
import csv

def visualize_predictions(model_path, X_test, y_test, img_shape=(28, 28)):
    """
    可视化模型预测结果
    :param model_path: 模型文件路径
    :param X_test: 测试集特征数据
    :param y_test: 测试集真实标签（one-hot编码）
    :param img_shape: 图像尺寸
    """
    # 加载模型
    model = MLP.load_model(model_path)

    # 随机选择20个样本（不重复）
    # np.random.seed(42)
    sample_indices = np.random.choice(len(X_test), 20, replace=False)
    X_test_samples = X_test[sample_indices]
    y_test_samples = y_test[sample_indices]

    # 生成预测结果
    y_pred = model.predict(X_test_samples)
    y_true = np.argmax(y_test_samples, axis=1)

    # 创建可视化窗口
    plt.figure(num="MLP 手写数字识别 MNIST 图像可视化",figsize=(12, 10))
    for i, (img, true_label, pred_label) in enumerate(zip(X_test_samples, y_true, y_pred)):
        plt.subplot(5, 4, i + 1)
        plt.imshow(img.reshape(img_shape), cmap='gray')
        plt.axis('off')

        # 使用不同颜色标注正确/错误预测
        color = 'green' if true_label == pred_label else 'blue'
        plt.title(f"Index: {sample_indices[i]}\n True: {true_label}\nPred: {pred_label}", fontweight='bold', color=color)

    plt.tight_layout()
    plt.show()

# 加载MNIST数据集
def load_mnist_data(file_path):
    """
    加载MNIST数据集
    :param file_path: 数据集文件路径
    :return: 特征和标签
    """
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过标题行
        for row in reader:
            data.append([int(x) for x in row])
    data = np.array(data)
    X = data[:, 1:] / 255.0
    y = np.eye(10)[data[:, 0]]
    return X, y
# 在主函数中添加调用示例
if __name__ == "__main__":
    # ...原有代码...                                                                           random_state=42)
    test_path = 'MNIST_data/mnist_test.csv'

    if os.path.exists(test_path):
        X_test_mnist, y_test_mnist = load_mnist_data(test_path)
    # 可视化预测结果（添加在模型训练之后）
    visualize_predictions('mlp_classification.npz',
                          X_test_mnist,
                          y_test_mnist)