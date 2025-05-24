import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from v1_torch_mlp_implementation import MLP


class MLPTest:
    def __init__(self, model_path, test_data_path):
        """
        初始化函数，设置模型路径和测试数据路径。
        :param model_path: 保存的模型文件路径
        :param test_data_path: 测试数据集文件路径
        """
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.model = self.load_model()
        self.X_test, self.y_test = self.load_mnist_data()

    def load_model(self):
        """
        加载训练好的模型。
        :return: 加载后的模型
        """
        input_size = 28 * 28
        hidden_sizes = [128, 64]
        output_size = 10
        activation ='relu'
        model = MLP(input_size, hidden_sizes, output_size, activation)
        model.load_state_dict(torch.load(self.model_path))
        model.eval()
        return model

    def load_mnist_data(self):
        """
        加载MNIST测试数据集。
        :return: 测试集特征数据和真实标签
        """
        data = []
        with open(self.test_data_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # 跳过标题行
            for row in reader:
                data.append([int(x) for x in row])
        data = np.array(data)
        X = data[:, 1:] / 255.0
        y = np.eye(10)[data[:, 0]]
        return X, y

    def visualize_predictions(self, img_shape=(28, 28)):
        """
        可视化模型预测结果。
        :param img_shape: 图像尺寸，默认为(28, 28)
        """
        # 随机选择20个样本（不重复）
        sample_indices = np.random.choice(len(self.X_test), 20, replace=False)
        X_test_samples = self.X_test[sample_indices]
        y_test_samples = self.y_test[sample_indices]

        # 将数据转换为PyTorch张量
        X_test_samples_tensor = torch.tensor(X_test_samples, dtype=torch.float32)

        # 生成预测结果
        with torch.no_grad():
            outputs = self.model(X_test_samples_tensor)
        _, y_pred = torch.max(outputs, 1)
        y_true = np.argmax(y_test_samples, axis=1)

        # 创建可视化窗口
        plt.figure(num="MLP Torch版手写数字识别 MNIST 图像可视化", figsize=(12, 10))
        for i, (img, true_label, pred_label) in enumerate(zip(X_test_samples, y_true, y_pred.numpy())):
            plt.subplot(5, 4, i + 1)
            plt.imshow(img.reshape(img_shape), cmap='gray')
            plt.axis('off')

            # 使用不同颜色标注正确/错误预测
            color = 'green' if true_label == pred_label else 'blue'
            plt.title(f"Index: {sample_indices[i]}\n True: {true_label}\nPred: {pred_label}", fontweight='bold',
                      color=color)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    model_path ='v1_torch_mlp_model.pth'  # 替换为你的模型保存路径
    test_data_path = 'MNIST_data/mnist_test.csv'  # 替换为你的测试数据路径
    mlp_tester = MLPTest(model_path, test_data_path)
    mlp_tester.visualize_predictions()
