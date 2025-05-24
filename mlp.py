# -*- coding = utf-8 -*-
# @Time : 2025/4/10 10:07
# @Author: Vast
# @File: mlp.py
# @Software: PyCharm
import numpy as np
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
import csv
import os
import threading
import logging


# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


# 定义激活函数及其导数
class Activation:
    @staticmethod
    def sigmoid(x):
        """
        Sigmoid激活函数（作用是将输入映射到0-1之间，常用于二分类输出）
        :param x: 输入数据
        :return: 经过sigmoid激活后的数据
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        """
        Sigmoid激活函数的导数（Sigmoid函数的导数可表示为sigmoid(x)*(1-sigmoid(x))）
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
        随机初始化权重（使用np.random.randn生成正态分布的随机数，然后乘以0.01进行缩放，这是常见的权重初始化方法，防止梯度爆炸或消失。 ）
        :param input_size: 输入层大小
        :param output_size: 输出层大小
        :return: 初始化后的权重矩阵
        """
        return np.random.randn(input_size, output_size) * 0.01

    @staticmethod
    def xavier_init(input_size, output_size):
        """
        Xavier初始化权重（首先计算limit值，然后生成均匀分布的矩阵）
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


class Adam(Optimizer):
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # 格式：{ (层索引, 参数类型): 矩阵 }
        self.v = {}
        self.t = 0

    def update(self, layer_idx, param_type, weights, gradients):
        """
        :param layer_idx: 层索引（从0开始）
        :param param_type: 'weight' 或 'bias'
        """
        self.t += 1
        key = (layer_idx, param_type)  # 唯一键

        if key not in self.m:
            self.m[key] = np.zeros_like(weights)
            self.v[key] = np.zeros_like(weights)

        # 更新矩估计
        self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * gradients
        self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (gradients ** 2)

        # 偏置修正
        m_hat = self.m[key] / (1 - self.beta1 ** self.t)
        v_hat = self.v[key] / (1 - self.beta2 ** self.t)

        return weights - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


# 定义正则化方法
class Regularizer:
    @staticmethod
    def l1(weights, lambda_):
        """
        L1正则化（计算L1正则化项，通过对权重矩阵所有元素取绝对值求和后乘以正则化系数lambda_。用于在损失函数中增加权重稀疏性惩罚，降低模型复杂度防止过拟合）
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
        L2正则化（2正则化通常用于防止过拟合，通过惩罚大的权重值，这里的计算是0.5乘以lambda参数，再乘以权重的平方和）
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
        弹性网络正则化（结合了L1和L2正则化，使用alpha作为权重参数，lambda_作为正则化系数。具体来说，计算alpha乘以L1正则化项加上(1-alpha)乘以L2正则化项，返回两者的加权和。）
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
        return alpha * Regularizer.l1_derivative(weights, lambda_) + (1 - alpha) * Regularizer.l2_derivative(weights,
                                                                                                             lambda_)


# 定义MLP模型
class MLP:
    def __init__(self, layers, activations, weight_init='random', optimizer='sgd', learning_rate=0.01,
                 regularization=None, lambda_=0.01, alpha=0.5, stop_criteria=1e-6):
        logging.info("开始模型初始化过程   Initializing MLP model...")
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

        # 在初始化最后添加结构输出
        self.print_model_summary()

    def print_model_summary(self):
        """打印模型结构摘要"""
        logging.info("\n{:=^60}".format(" 输出模型结构摘要信息    Model Summary "))
        total_params = 0
        layers_info = []

        # 输入层
        layers_info.append({
            'Layer': 0,
            'Type': 'Input',
            'Neurons': self.layers[0],
            'Activation': '-',
            'Weights Shape': '-',
            'Bias Shape': '-',
            'Params': 0
        })

        # 隐藏层和输出层
        for i in range(len(self.layers) - 1):
            layer_type = 'Hidden' if i < len(self.layers) - 2 else 'Output'
            weights_shape = self.weights[i].shape if i < len(self.weights) else '-'
            bias_shape = self.biases[i].shape if i < len(self.biases) else '-'
            params = np.prod(weights_shape) + np.prod(bias_shape) if i < len(self.weights) else 0

            layers_info.append({
                'Layer': i + 1,
                'Type': layer_type,
                'Neurons': self.layers[i + 1],
                'Activation': self.activations[i],
                'Weights Shape': weights_shape,
                'Bias Shape': bias_shape,
                'Params': params
            })
            total_params += params

        # 打印表格
        logging.info("{:<6} {:<8} {:<8} {:<10} {:<15} {:<12} {:<10}".format(
            'Layer', 'Type', 'Neurons', 'Activation', 'Weights Shape', 'Bias Shape', 'Parameters'
        ))
        logging.info("-" * 70)
        for info in layers_info:
            logging.info("{:<6} {:<8} {:<8} {:<10} {:<15} {:<12} {:<10,}".format(
                info['Layer'],
                info['Type'],
                info['Neurons'],
                info['Activation'],
                str(info['Weights Shape']),
                str(info['Bias Shape']),
                info['Params']
            ))

        # 打印汇总信息
        logging.info("\n{:=^60}".format(" Summary "))
        logging.info(f"Total layers: {len(self.layers)} (input + {len(self.layers) - 2} hidden + output)")
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Weight Initialization: {self.weight_init}")
        logging.info(f"Regularization: {self.regularization if self.regularization else 'None'}")
        if self.regularization:
            logging.info(
                f"Lambda: {self.lambda_}{' Alpha: ' + str(self.alpha) if self.regularization == 'elastic' else ''}")
        logging.info(f"Optimizer: {type(self.optimizer).__name__} (lr={self.optimizer.learning_rate})")
        logging.info("=" * 60 + "\n")

    def forward(self, X):
        """
        前向传播
        :param X: 输入数据
        :return: 输出结果和各层激活值
        """
        logging.info("开始前向传播过程   Forward propagation...")
        # activations列表的作用是存储每一层的输出，包括输入层的数据。输入层的数据X作为第一个元素，之后每一层的输出都是基于前一层的激活值计算得到的
        activations = [X]
        logging.info(f"输入数据形状: {X.shape}")
        for i in range(len(self.layers) - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            activation_func = self.activation_functions[self.activations[i]][0]
            a = activation_func(z)
            activations.append(a)
            logging.info(f"第 {i + 1} 层输入形状: {activations[-2].shape}, 输出形状: {a.shape}")
        logging.info("前向传播完成   Forward propagation completed.")
        return activations[-1], activations

    def backward(self, X, y, activations):
        """
        反向传播
        :param X: 输入数据
        :param y: 真实标签
        :param activations: 各层激活值
        :return: 权重和偏置的梯度
        """
        logging.info("开始反向传播过程   Backward propagation...")
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

            # 添加正则化项
            if self.regularization == 'l1':
                weight_gradient += Regularizer.l1_derivative(self.weights[i], self.lambda_)
            elif self.regularization == 'l2':
                weight_gradient += Regularizer.l2_derivative(self.weights[i], self.lambda_)
            elif self.regularization == 'elastic':
                weight_gradient += Regularizer.elastic_derivative(self.weights[i], self.lambda_, self.alpha)

            weight_gradients.insert(0, weight_gradient)
            bias_gradients.insert(0, bias_gradient)

            if i > 0:
                activation_derivative = self.activation_functions[self.activations[i - 1]][1]
                delta = np.dot(delta, self.weights[i].T) * activation_derivative(activations[i])
        logging.info("反向传播完成   Backward propagation completed.")
        return weight_gradients, bias_gradients

    def train(self, X, y, epochs=100, batch_size=32, parallel=False):
        logging.info(f"开始模型训练过程，共 {epochs} 轮，批次大小 {batch_size}")
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
                # 并行训练过程
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
                # 单线程训练过程
                for batch in range(num_batches):
                    start = batch * batch_size
                    end = start + batch_size
                    X_batch = X[start:end]
                    y_batch = y[start:end]
                    loss = self._train_batch(X_batch, y_batch)
                    epoch_loss += loss

            epoch_loss /= num_batches
            losses.append(epoch_loss)
            if parallel:
                logging.info(f'######################并行模型训练过程:Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss}')
            else:
                logging.info(f'#####################单线程模型训练过程:Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss}')

            # 停止条件
            if epoch > 0 and abs(losses[-1] - losses[-2]) < self.stop_criteria:
                logging.info(f"达到停止条件，提前结束训练，在第 {epoch + 1} 轮")
                break

        return losses

    # def _train_batch(self, X_batch, y_batch):
    #     """
    #     训练一个批次的数据
    #     :param X_batch: 批次输入数据
    #     :param y_batch: 批次真实标签
    #     :return: 该批次的损失值
    #     """
    #     output, activations = self.forward(X_batch)
    #     weight_gradients, bias_gradients = self.backward(X_batch, y_batch, activations)
    #
    #     # 更新权重和偏置
    #     for i in range(len(self.weights)):
    #         self.weights[i] = self.optimizer.update(self.weights[i], weight_gradients[i])
    #         self.biases[i] = self.optimizer.update(self.biases[i], bias_gradients[i])
    #
    #     # 计算损失
    #     loss = self._compute_loss(output, y_batch)
    #     return loss
    def _train_batch(self, X_batch, y_batch):
        output, activations = self.forward(X_batch)
        weight_gradients, bias_gradients = self.backward(X_batch, y_batch, activations)

        # 更新权重和偏置
        for i in range(len(self.weights)):
            # 更新权重
            self.weights[i] = self.optimizer.update(
                layer_idx=i,
                param_type='weight',
                weights=self.weights[i],
                gradients=weight_gradients[i]
            )
            # 更新偏置
            self.biases[i] = self.optimizer.update(
                layer_idx=i,
                param_type='bias',
                weights=self.biases[i],
                gradients=bias_gradients[i]
            )

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
        logging.info("计算训练损失中...")
        num_samples = y.shape[0]
        if self.activations[-1] == 'softmax':
            #计算交叉熵损失。通过取模型输出概率的对数值，与真实标签逐元素相乘求和后取负，最后除以样本数得到平均损失。添加1e-8防止对零取对数导致数值错误。
            loss = -np.sum(y * np.log(output + 1e-8)) / num_samples
        else:
            # 计算均方差损失
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
        logging.info(f"训练损失为: {loss}")
        return loss

    def predict(self, X):
        """
        预测
        :param X: 输入数据
        :return: 预测结果
        """
        logging.info("开始模型预测过程...")
        output, _ = self.forward(X)
        if self.activations[-1] == 'softmax':
            return np.argmax(output, axis=1)
        logging.info("模型预测完成，返回预测结果...")
        return output

    def save_model(self, file_path):
        logging.info(f"开始保存模型到文件: {file_path}")
        model_data = {
            'layers': self.layers,
            'activations': self.activations,
            'weight_init': self.weight_init,
            'regularization': self.regularization,
            'lambda_': self.lambda_,
            'alpha': self.alpha,
            'stop_criteria': self.stop_criteria
        }

        # 按层保存权重和偏置
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            model_data[f'weight_{i}'] = w
            model_data[f'bias_{i}'] = b
        logging.info(f"模型数据保存完成，共保存了 {len(model_data)} 个参数。")
        np.savez(file_path, **model_data)

    @staticmethod
    def load_model(file_path):
        logging.info(f"开始加载模型文件: {file_path}")
        data = np.load(file_path)
        model = MLP(
            layers=data['layers'],
            activations=data['activations'],
            weight_init=data['weight_init'],
            optimizer='sgd',  # 优化器类型需重新指定
            regularization=data['regularization'],
            lambda_=data['lambda_'],
            alpha=data['alpha'],
            stop_criteria=data['stop_criteria']
        )

        # 按层加载权重和偏置
        model.weights = [data[f'weight_{i}'] for i in range(len(model.weights))]
        model.biases = [data[f'bias_{i}'] for i in range(len(model.biases))]
        logging.info("模型加载完成，返回模型对象...")
        return model

    def confusion_matrix(self, X, y):
        """
        计算混淆矩阵(计算混淆矩阵以评估分类的准确性。)
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


# 加载MNIST数据集
def load_mnist_data(file_path):
    """
    加载MNIST数据集
    :param file_path: 数据集文件路径
    :return: 特征和标签
    """
    logging.info(f"开始加载MNIST数据集，文件路径：{file_path}")
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过标题行
        for row in reader:
            data.append([int(x) for x in row])
    data = np.array(data)
    # 特征归一化处理：将MNIST图像的像素值（data第2列及之后）除以255实现归一化
    X = data[:, 1:] / 255.0
    # 标签独热编码：用np.eye生成10维单位矩阵，根据data首列标签值索引对应one-hot向量
    y = np.eye(10)[data[:, 0]]
    logging.info(f"开始加载MNIST数据集，文件路径：{file_path}")
    return X, y


# 示例使用
if __name__ == "__main__":
    # 分类任务（MNIST数据集）
    logging.info("分类任务（加载MNIST数据集）loading...............")

    train_path = 'MNIST_data/mnist_train.csv'   # 训练数据集路径
    test_path = 'MNIST_data/mnist_test.csv'     # 测试数据集路径

    if os.path.exists(train_path) and os.path.exists(test_path):
        X_train_mnist, y_train_mnist = load_mnist_data(train_path)
        X_test_mnist, y_test_mnist = load_mnist_data(test_path)

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
        logging.info(f'MNIST 混淆矩阵:\n {conf_matrix_mnist}')
    else:
        logging.info("MNIST数据集文件未找到，请检查路径。")

