import numpy as np
# from sklearn.datasets import load_boston
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
import csv
import os
import threading
import logging
import struct

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# 定义激活函数及其导数
class Activation:
    @staticmethod
    def sigmoid(x):
        """
        Sigmoid激活函数
        :param x: 输入数据
        :return: 经过sigmoid激活后的数据
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        """
        Sigmoid激活函数的导数
        :param x: 输入数据
        :return: sigmoid导数计算结果
        """
        return Activation.sigmoid(x) * (1 - Activation.sigmoid(x))

    @staticmethod
    def relu(x):
        """
        ReLU激活函数
        :param x: 输入数据
        :return: 经过ReLU激活后的数据
        """
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        """
        ReLU激活函数的导数
        :param x: 输入数据
        :return: ReLU导数计算结果
        """
        return (x > 0).astype(float)

    @staticmethod
    def softmax(x):
        """
        Softmax激活函数，用于多分类问题
        :param x: 输入数据
        :return: 经过Softmax激活后的数据
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# 定义权重初始化方法
class WeightInitializer:
    @staticmethod
    def random_init(input_size, output_size):
        """
        随机初始化权重
        :param input_size: 输入层大小
        :param output_size: 输出层大小
        :return: 初始化后的权重矩阵
        """
        return np.random.randn(input_size, output_size) * 0.01

    @staticmethod
    def xavier_init(input_size, output_size):
        """
        Xavier初始化权重
        :param input_size: 输入层大小
        :param output_size: 输出层大小
        :return: 初始化后的权重矩阵
        """
        limit = np.sqrt(6 / (input_size + output_size))
        return np.random.uniform(-limit, limit, (input_size, output_size))

# 定义优化器
class Optimizer:
    def __init__(self, learning_rate):
        """
        优化器基类构造函数
        :param learning_rate: 学习率
        """
        self.learning_rate = learning_rate

    def update(self, weights, gradients):
        """
        更新权重的抽象方法，需要在子类中实现
        :param weights: 当前权重矩阵
        :param gradients: 梯度矩阵
        :return: 更新后的权重矩阵
        """
        raise NotImplementedError

class SGD(Optimizer):
    def update(self, weights, gradients):
        """
        随机梯度下降更新权重
        :param weights: 当前权重矩阵
        :param gradients: 梯度矩阵
        :return: 更新后的权重矩阵
        """
        return weights - self.learning_rate * gradients

class Momentum(Optimizer):
    def __init__(self, learning_rate, momentum=0.9):
        """
        动量优化器构造函数
        :param learning_rate: 学习率
        :param momentum: 动量系数
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.v = None

    def update(self, weights, gradients):
        """
        使用动量更新权重
        :param weights: 当前权重矩阵
        :param gradients: 梯度矩阵
        :return: 更新后的权重矩阵
        """
        if self.v is None:
            self.v = np.zeros_like(weights)
        self.v = self.momentum * self.v + self.learning_rate * gradients
        return weights - self.v

class RMSProp(Optimizer):
    def __init__(self, learning_rate, rho=0.9, epsilon=1e-8):
        """
        RMSProp优化器构造函数
        :param learning_rate: 学习率
        :param rho: 衰减率
        :param epsilon: 防止除零的小常数
        """
        super().__init__(learning_rate)
        self.rho = rho
        self.epsilon = epsilon
        self.s = None

    def update(self, weights, gradients):
        """
        使用RMSProp更新权重
        :param weights: 当前权重矩阵
        :param gradients: 梯度矩阵
        :return: 更新后的权重矩阵
        """
        if self.s is None:
            self.s = np.zeros_like(weights)
        self.s = self.rho * self.s + (1 - self.rho) * gradients ** 2
        return weights - self.learning_rate * gradients / (np.sqrt(self.s) + self.epsilon)

# class Adam(Optimizer):
#     def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
#         """
#         Adam优化器构造函数
#         :param learning_rate: 学习率
#         :param beta1: 一阶矩估计的指数衰减率
#         :param beta2: 二阶矩估计的指数衰减率
#         :param epsilon: 防止除零的小常数
#         """
#         super().__init__(learning_rate)
#         self.beta1 = beta1
#         self.beta2 = beta2
#         self.epsilon = epsilon
#         self.m = {}
#         self.v = {}
#         self.t = 0
#
#     def update(self, weights, gradients):
#         """
#         使用Adam更新权重
#         :param weights: 当前权重矩阵
#         :param gradients: 梯度矩阵
#         :return: 更新后的权重矩阵
#         """
#         self.t += 1
#         layer_index = id(weights)
#         if layer_index not in self.m:
#             self.m[layer_index] = np.zeros_like(gradients)
#             self.v[layer_index] = np.zeros_like(gradients)
#             logging.info(f"Initialized m and v for layer {layer_index} with shape {self.m[layer_index].shape}")
#         logging.info(f"Layer {layer_index}: weights shape {weights.shape}, gradients shape {gradients.shape}, m shape {self.m[layer_index].shape}, v shape {self.v[layer_index].shape}")
#         self.m[layer_index] = self.beta1 * self.m[layer_index] + (1 - self.beta1) * gradients
#         self.v[layer_index] = self.beta2 * self.v[layer_index] + (1 - self.beta2) * gradients ** 2
#         m_hat = self.m[layer_index] / (1 - self.beta1 ** self.t)
#         v_hat = self.v[layer_index] / (1 - self.beta2 ** self.t)
#         return weights - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
class Adam(Optimizer):
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, weights, gradients):
        self.t += 1
        layer_index = id(weights)
        if layer_index not in self.m:
            # 确保m和v的形状与梯度形状一致
            self.m[layer_index] = np.zeros_like(gradients)
            self.v[layer_index] = np.zeros_like(gradients)
            logging.info(f"Initialized m and v for layer {layer_index} with shape {self.m[layer_index].shape}")
        logging.info(f"Layer {layer_index}: weights shape {weights.shape}, gradients shape {gradients.shape}, m shape {self.m[layer_index].shape}, v shape {self.v[layer_index].shape}")
        # 确保更新m和v时使用正确的梯度形状
        if gradients.shape != self.m[layer_index].shape:
            raise ValueError(f"Gradients shape {gradients.shape} does not match m shape {self.m[layer_index].shape}")
        self.m[layer_index] = self.beta1 * self.m[layer_index] + (1 - self.beta1) * gradients
        self.v[layer_index] = self.beta2 * self.v[layer_index] + (1 - self.beta2) * gradients ** 2
        m_hat = self.m[layer_index] / (1 - self.beta1 ** self.t)
        v_hat = self.v[layer_index] / (1 - self.beta2 ** self.t)
        return weights - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


# 定义正则化方法
class Regularizer:
    @staticmethod
    def l1(weights, lambda_):
        """
        L1正则化
        :param weights: 权重矩阵
        :param lambda_: 正则化系数
        :return: L1正则化项
        """
        return lambda_ * np.sum(np.abs(weights))

    @staticmethod
    def l1_derivative(weights, lambda_):
        """
        L1正则化的导数
        :param weights: 权重矩阵
        :param lambda_: 正则化系数
        :return: L1正则化导数
        """
        return lambda_ * np.sign(weights)

    @staticmethod
    def l2(weights, lambda_):
        """
        L2正则化
        :param weights: 权重矩阵
        :param lambda_: 正则化系数
        :return: L2正则化项
        """
        return 0.5 * lambda_ * np.sum(weights ** 2)

    @staticmethod
    def l2_derivative(weights, lambda_):
        """
        L2正则化的导数
        :param weights: 权重矩阵
        :param lambda_: 正则化系数
        :return: L2正则化导数
        """
        return lambda_ * weights

    @staticmethod
    def elastic(weights, lambda_, alpha):
        """
        弹性网络正则化
        :param weights: 权重矩阵
        :param lambda_: 正则化系数
        :param alpha: L1正则化比例
        :return: 弹性网络正则化项
        """
        return alpha * Regularizer.l1(weights, lambda_) + (1 - alpha) * Regularizer.l2(weights, lambda_)

    @staticmethod
    def elastic_derivative(weights, lambda_, alpha):
        """
        弹性网络正则化的导数
        :param weights: 权重矩阵
        :param lambda_: 正则化系数
        :param alpha: L1正则化比例
        :return: 弹性网络正则化导数
        """
        return alpha * Regularizer.l1_derivative(weights, lambda_) + (1 - alpha) * Regularizer.l2_derivative(weights, lambda_)

# 定义MLP模型
class MLP:
    def __init__(self, layers, activations, weight_init='random', optimizer='sgd', learning_rate=0.01,
                 regularization=None, lambda_=0.01, alpha=0.5, stop_criteria=1e-6):
        """
        MLP模型构造函数
        :param layers: 各层神经元数量列表
        :param activations: 各层激活函数列表
        :param weight_init: 权重初始化方法
        :param optimizer: 优化器类型
        :param learning_rate: 学习率
        :param regularization: 正则化方法
        :param lambda_: 正则化系数
        :param alpha: 弹性网络正则化中L1的比例
        :param stop_criteria: 停止训练的标准
        """
        self.layers = layers
        self.activations = activations
        self.weights = []
        self.biases = []
        self.activation_functions = {
            'sigmoid': (Activation.sigmoid, Activation.sigmoid_derivative),
            'relu': (Activation.relu, Activation.relu_derivative),
            'softmax': (Activation.softmax, None)
        }
        self.weight_init = weight_init
        self.regularization = regularization
        self.lambda_ = lambda_
        self.alpha = alpha
        self.stop_criteria = stop_criteria

        # 初始化权重和偏置
        for i in range(len(layers) - 1):
            if weight_init == 'random':
                self.weights.append(WeightInitializer.random_init(layers[i], layers[i + 1]))
            elif weight_init == 'xavier':
                self.weights.append(WeightInitializer.xavier_init(layers[i], layers[i + 1]))
            self.biases.append(np.zeros((1, layers[i + 1])))

        # 选择优化器
        if optimizer == 'sgd':
            self.optimizer = SGD(learning_rate)
        elif optimizer == 'momentum':
            self.optimizer = Momentum(learning_rate)
        elif optimizer == 'rmsprop':
            self.optimizer = RMSProp(learning_rate)
        elif optimizer == 'adam':
            self.optimizer = Adam(learning_rate)

    def forward(self, X):
        """
        前向传播
        :param X: 输入数据
        :return: 输出结果和各层激活值
        """
        activations = [X]
        logging.info(f"输入数据形状: {X.shape}")
        for i in range(len(self.layers) - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            activation_func = self.activation_functions[self.activations[i]][0]
            a = activation_func(z)
            activations.append(a)
            logging.info(f"第 {i + 1} 层输入形状: {activations[-2].shape}, 输出形状: {a.shape}")
        return activations[-1], activations

    # def backward(self, X, y, activations):
    #     """
    #     反向传播
    #     :param X: 输入数据
    #     :param y: 真实标签
    #     :param activations: 各层激活值
    #     :return: 权重和偏置的梯度
    #     """
    #     num_samples = X.shape[0]
    #     weight_gradients = []
    #     bias_gradients = []
    #     output = activations[-1]
    #
    #     # 计算输出层误差
    #     if self.activations[-1] == 'softmax':
    #         delta = output - y
    #     else:
    #         activation_derivative = self.activation_functions[self.activations[-1]][1]
    #         delta = (output - y) * activation_derivative(output)
    #
    #     # 反向传播计算梯度
    #     for i in range(len(self.layers) - 2, -1, -1):
    #         weight_gradient = np.dot(activations[i].T, delta) / num_samples
    #         bias_gradient = np.sum(delta, axis=0, keepdims=True) / num_samples
    #         # 打印梯度形状，用于调试
    #         logging.info(f"Layer {i} weight gradient shape: {weight_gradient.shape}")
    #         logging.info(f"Layer {i} bias gradient shape: {bias_gradient.shape}")
    #
    #         # 添加正则化项
    #         if self.regularization == 'l1':
    #             weight_gradient += Regularizer.l1_derivative(self.weights[i], self.lambda_)
    #         elif self.regularization == 'l2':
    #             weight_gradient += Regularizer.l2_derivative(self.weights[i], self.lambda_)
    #         elif self.regularization == 'elastic':
    #             weight_gradient += Regularizer.elastic_derivative(self.weights[i], self.lambda_, self.alpha)
    #
    #         weight_gradients.insert(0, weight_gradient)
    #         bias_gradients.insert(0, bias_gradient)
    #
    #         # 计算前一层的delta
    #         if i > 0:
    #             activation_derivative = self.activation_functions[self.activations[i - 1]][1]
    #         else:
    #             activation_derivative = lambda x: np.ones_like(x)
    #         delta = np.dot(delta, self.weights[i].T) * activation_derivative(activations[i])
    #
    #     return weight_gradients, bias_gradients

    def backward(self, X, y, activations):
        num_samples = X.shape[0]
        weight_gradients = []
        bias_gradients = []
        output = activations[-1]

        # 计算输出层误差
        if self.activations[-1] == 'softmax':
            delta = output - y
        else:
            activation_derivative = self.activation_functions[self.activations[-1]][1]
            delta = (output - y) * activation_derivative(output)

        # 反向传播计算梯度
        for i in range(len(self.layers) - 2, -1, -1):
            weight_gradient = np.dot(activations[i].T, delta) / num_samples
            bias_gradient = np.sum(delta, axis=0, keepdims=True) / num_samples
            # 打印梯度形状，用于调试
            logging.info(f"Layer {i} weight gradient shape: {weight_gradient.shape}")
            logging.info(f"Layer {i} bias gradient shape: {bias_gradient.shape}")

            # 确保权重梯度形状与权重矩阵形状匹配
            if weight_gradient.shape != self.weights[i].shape:
                raise ValueError(
                    f"Weight gradient shape {weight_gradient.shape} does not match weights shape {self.weights[i].shape}")
            # 确保偏置梯度形状与偏置向量形状匹配
            if bias_gradient.shape != self.biases[i].shape:
                raise ValueError(
                    f"Bias gradient shape {bias_gradient.shape} does not match biases shape {self.biases[i].shape}")

            # 添加正则化项
            if self.regularization == 'l1':
                weight_gradient += Regularizer.l1_derivative(self.weights[i], self.lambda_)
            elif self.regularization == 'l2':
                weight_gradient += Regularizer.l2_derivative(self.weights[i], self.lambda_)
            elif self.regularization == 'elastic':
                weight_gradient += Regularizer.elastic_derivative(self.weights[i], self.lambda_, self.alpha)

            weight_gradients.insert(0, weight_gradient)
            bias_gradients.insert(0, bias_gradient)

            # 计算前一层的delta
            if i > 0:
                activation_derivative = self.activation_functions[self.activations[i - 1]][1]
            else:
                activation_derivative = lambda x: np.ones_like(x)
            delta = np.dot(delta, self.weights[i].T) * activation_derivative(activations[i])

        return weight_gradients, bias_gradients

    def train(self, X, y, epochs=100, batch_size=32, parallel=False):
        """
        训练模型
        :param X: 输入数据
        :param y: 真实标签
        :param epochs: 训练轮数
        :param batch_size: 批次大小
        :param parallel: 是否并行训练
        :return: 每轮的损失值
        """
        losses = []
        num_samples = X.shape[0]
        num_batches = num_samples // batch_size

        for epoch in range(epochs):
            epoch_loss = 0
            if parallel:
                threads = []
                for batch in range(num_batches):
                    start = batch * batch_size
                    end = start + batch_size
                    X_batch = X[start:end]
                    y_batch = y[start:end]
                    thread = threading.Thread(target=self._train_batch, args=(X_batch, y_batch))
                    threads.append(thread)
                    thread.start()
                for thread in threads:
                    thread.join()
            else:
                for batch in range(num_batches):
                    start = batch * batch_size
                    end = start + batch_size
                    X_batch = X[start:end]
                    y_batch = y[start:end]
                    loss = self._train_batch(X_batch, y_batch)
                    epoch_loss += loss

            epoch_loss /= num_batches
            losses.append(epoch_loss)
            logging.info(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss}')

            # 停止条件
            if epoch > 0 and abs(losses[-1] - losses[-2]) < self.stop_criteria:
                logging.info(f"达到停止条件，提前结束训练，在第 {epoch + 1} 轮")
                break

        return losses

    def _train_batch(self, X_batch, y_batch):
        """
        训练一个批次的数据
        :param X_batch: 批次输入数据
        :param y_batch: 批次真实标签
        :return: 该批次的损失值
        """
        output, activations = self.forward(X_batch)
        weight_gradients, bias_gradients = self.backward(X_batch, y_batch, activations)

        # 更新权重和偏置
        for i in range(len(self.weights)):
            self.weights[i] = self.optimizer.update(self.weights[i], weight_gradients[i])
            self.biases[i] = self.optimizer.update(self.biases[i], bias_gradients[i])

        # 计算损失
        loss = self._compute_loss(output, y_batch)
        return loss

    def _compute_loss(self, output, y):
        """
        计算损失
        :param output: 模型输出
        :param y: 真实标签
        :return: 损失值
        """
        num_samples = y.shape[0]
        if self.activations[-1] == 'softmax':
            loss = -np.sum(y * np.log(output + 1e-8)) / num_samples
        else:
            loss = np.mean((output - y) ** 2)

        # 添加正则化项
        if self.regularization == 'l1':
            reg_loss = sum([Regularizer.l1(w, self.lambda_) for w in self.weights])
            loss += reg_loss
        elif self.regularization == 'l2':
            reg_loss = sum([Regularizer.l2(w, self.lambda_) for w in self.weights])
            loss += reg_loss
        elif self.regularization == 'elastic':
            reg_loss = sum([Regularizer.elastic(w, self.lambda_, self.alpha) for w in self.weights])
            loss += reg_loss

        return loss

    def predict(self, X):
        """
        预测
        :param X: 输入数据
        :return: 预测结果
        """
        output, _ = self.forward(X)
        if self.activations[-1] == 'softmax':
            return np.argmax(output, axis=1)
        return output

    def save_model(self, file_path):
        """
        保存模型
        :param file_path: 保存文件路径
        """
        model_data = {
            'layers': self.layers,
            'activations': self.activations,
            'weights': self.weights,
            'biases': self.biases,
            'weight_init': self.weight_init,
            'regularization': self.regularization,
            'lambda_': self.lambda_,
            'alpha': self.alpha,
            'stop_criteria': self.stop_criteria
        }
        np.savez(file_path, **model_data)

    @staticmethod
    def load_model(file_path):
        """
        加载模型
        :param file_path: 模型文件路径
        :return: 加载后的模型
        """
        data = np.load(file_path)
        model = MLP(data['layers'], data['activations'], data['weight_init'], 'sgd',
                    regularization=data['regularization'], lambda_=data['lambda_'], alpha=data['alpha'],
                    stop_criteria=data['stop_criteria'])
        model.weights = list(data['weights'])
        model.biases = list(data['biases'])
        return model

    def confusion_matrix(self, X, y):
        """
        计算混淆矩阵
        :param X: 输入数据
        :param y: 真实标签
        :return: 混淆矩阵
        """
        y_pred = self.predict(X)
        if self.activations[-1] == 'softmax':
            y_true = np.argmax(y, axis=1)
        else:
            y_true = y.flatten()
        return sk_confusion_matrix(y_true, y_pred)

# 加载MNIST二进制数据集
def load_mnist_images(file_path):
    with open(file_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows * cols)
    return images / 255.0


def load_mnist_labels(file_path):
    with open(file_path, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return np.eye(10)[labels]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 分类任务（MNIST数据集）
    logging.info("分类任务（MNIST数据集）")
    train_images_path = 'MNIST_data/train-images-idx3-ubyte'
    train_labels_path = 'MNIST_data/train-labels-idx1-ubyte'
    test_images_path = 'MNIST_data/t10k-images-idx3-ubyte'
    test_labels_path = 'MNIST_data/t10k-labels-idx1-ubyte'

    if all([os.path.exists(path) for path in [train_images_path, train_labels_path, test_images_path, test_labels_path]]):
        X_train_mnist = load_mnist_images(train_images_path)
        y_train_mnist = load_mnist_labels(train_labels_path)
        X_test_mnist = load_mnist_images(test_images_path)
        y_test_mnist = load_mnist_labels(test_labels_path)

        # 定义MLP模型
        mlp_classification = MLP(layers=[784, 128, 10], activations=['relu', 'softmax'], weight_init='xavier',
                                 optimizer='adam', learning_rate=0.001, regularization='l2', lambda_=0.001)

        # 训练模型
        losses_classification = mlp_classification.train(X_train_mnist, y_train_mnist, epochs=10, batch_size=32)

        # 保存模型
        mlp_classification.save_model('mlp_classification.npz')

        # 预测
        y_pred_mnist = mlp_classification.predict(X_test_mnist)
        y_true_mnist = np.argmax(y_test_mnist, axis=1)
        accuracy_mnist = np.mean(y_pred_mnist == y_true_mnist)
        logging.info(f"MNIST 分类准确率: {accuracy_mnist * 100:.2f}%")

        # 混淆矩阵
        conf_matrix_mnist = mlp_classification.confusion_matrix(X_test_mnist, y_test_mnist)
        logging.info("MNIST 混淆矩阵:")
        logging.info(conf_matrix_mnist)
    else:
        logging.info("MNIST数据集文件未找到，请检查路径。")

    # # 回归任务（波士顿房价数据集）
    # logging.info("\n回归任务（波士顿房价数据集）")
    # boston = load_boston()
    # X_boston = boston.data
    # y_boston = boston.target
    # scaler = StandardScaler()
    # X_boston = scaler.fit_transform(X_boston)
    # X_train_boston, X_test_boston, y_train_boston, y_test_boston = train_test_split(X_boston, y_boston, test_size=0.2, random_state=42)
    # y_train_boston = y_train_boston.reshape(-1, 1)
    # y_test_boston = y_test_boston.reshape(-1, 1)
    #
    # # 定义MLP模型
    # mlp_regression = MLP(layers=[13, 64, 1], activations=['relu', 'linear'], weight_init='xavier',
    #                      optimizer='adam', learning_rate=0.001, regularization='l2', lambda_=0.001)
    #
    # # 训练模型
    # losses_regression = mlp_regression.train(X_train_boston, y_train_boston, epochs=10, batch_size=32)
    #
    # # 保存模型
    # mlp_regression.save_model('mlp_regression.npz')
    #
    # # 预测
    # y_pred_boston = mlp_regression.predict(X_test_boston)
    # mse_boston = np.mean((y_pred_boston - y_test_boston) ** 2)
    # logging.info(f"波士顿房价回归均方误差: {mse_boston:.2f}")