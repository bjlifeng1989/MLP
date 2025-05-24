# -*- coding = utf-8 -*-
# @Time : 2025/3/28 18:09
# @Author: Vast
# @File: mlp1.py
# @Software: PyCharm
"""
MLP框架核心实现（PyTorch风格）
包含：优化器(Adam/Momentum/RMSprop)、早停机制、并行训练
"""

import numpy as np
from abc import ABC, abstractmethod
from multiprocessing import Pool, cpu_count
from typing import List, Callable, Optional

from function import ReLU, Softmax, cross_entropy_loss


class Tensor(np.ndarray):
    """ 自定义张量类型，用于类型标注 """
    pass


class Module(ABC):
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor: ...

    @abstractmethod
    def backward(self, grad: Tensor) -> Tensor: ...

    def parameters(self) -> List[Tensor]:
        return []


class Linear(Module):
    def __init__(self, in_features: int, out_features: int,
                 init_method: str = 'he'):
        """
        全连接层
        :param init_method: 初始化方法 (he/xavier/normal)
        """
        super().__init__()
        self.w = self._init_weights((out_features, in_features), init_method)
        self.b = np.zeros(out_features)
        self.x = None  # 前向传播缓存

    def _init_weights(self, shape, method: str) -> Tensor:
        if method == 'he':
            std = np.sqrt(2.0 / shape[1])
        elif method == 'xavier':
            std = np.sqrt(2.0 / (shape[0] + shape[1]))
        else:  # normal
            std = 0.01
        return np.random.normal(0, std, shape).astype(np.float32)

    def forward(self, x: Tensor) -> Tensor:
        self.x = x  # 缓存输入用于反向传播
        return self.w @ x + self.b

    def backward(self, grad: Tensor) -> Tensor:
        # 计算参数梯度
        self.dw = np.outer(grad, self.x)  # outer product
        self.db = grad.copy()
        # 返回输入的梯度
        return self.w.T @ grad

    def parameters(self) -> List[Tensor]:
        return [self.w, self.b]


class Optimizer:
    """ 优化器基类（类似torch.optim） """

    def __init__(self, params: List[Tensor], lr: float = 0.01):
        self.params = params
        self.lr = lr

    @abstractmethod
    def step(self) -> None: ...


class SGD(Optimizer):
    """ 普通SGD优化器 """

    def step(self) -> None:
        for p in self.params:
            p -= self.lr * p.grad


class Momentum(Optimizer):
    """ 动量优化器 (类似torch.optim.SGD with momentum) """

    def __init__(self, params, lr=0.01, momentum=0.9):
        super().__init__(params, lr)
        self.momentum = momentum
        self.velocities = [np.zeros_like(p) for p in params]

    def step(self) -> None:
        for i, p in enumerate(self.params):
            self.velocities[i] = self.momentum * self.velocities[i] + self.lr * p.grad
            p -= self.velocities[i]


class RMSprop(Optimizer):
    """ RMSprop优化器 (类似torch.optim.RMSprop) """

    def __init__(self, params, lr=0.001, alpha=0.99, eps=1e-8):
        super().__init__(params, lr)
        self.alpha = alpha
        self.eps = eps
        self.avg_sq_grad = [np.zeros_like(p) for p in params]

    def step(self) -> None:
        for i, p in enumerate(self.params):
            self.avg_sq_grad[i] = self.alpha * self.avg_sq_grad[i] + \
                                  (1 - self.alpha) * p.grad ** 2
            p -= self.lr * p.grad / (np.sqrt(self.avg_sq_grad[i]) + self.eps)


class Adam(Optimizer):
    """ Adam优化器 (类似torch.optim.Adam) """

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [np.zeros_like(p) for p in params]  # 一阶矩
        self.v = [np.zeros_like(p) for p in params]  # 二阶矩
        self.t = 0  # 时间步

    def step(self) -> None:
        self.t += 1
        for i, p in enumerate(self.params):
            # 更新矩估计
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * p.grad ** 2
            # 偏差修正
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            # 更新参数
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class EarlyStopping:
    """ 早停机制 (类似PyTorch的EarlyStopping回调) """

    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = np.inf

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # 触发停止
        return False


class ParallelSGD:
    """ 并行SGD实现（使用多进程） """

    def __init__(self, model: 'MLP', n_workers: int = None):
        self.model = model
        self.n_workers = n_workers or cpu_count()
        self.pool = Pool(self.n_workers)

    def _parallel_backward(self, batch: tuple) -> List[Tensor]:
        """ 每个worker计算小批量梯度 """
        X_batch, y_batch = batch
        grads = []
        # 前向传播
        output = self.model(X_batch)
        loss = self.model.loss_fn(output, y_batch)
        # 反向传播
        grad = self.model.loss_derivative(output, y_batch)
        for layer in reversed(self.model.layers):
            grad = layer.backward(grad)
            grads.append([param.grad.copy() for param in layer.parameters()])
        return grads  # 梯度按层倒序存储

    def step(self, dataloader: 'DataLoader') -> None:
        """ 并行训练步骤 """
        all_grads = self.pool.map(self._parallel_backward, dataloader.batches())
        # 平均梯度
        for layer_idx in range(len(self.model.layers)):
            layer = self.model.layers[layer_idx]
            for param_idx in range(len(layer.parameters())):
                # 汇总所有worker的梯度
                avg_grad = np.mean([grads[layer_idx][param_idx]
                                    for grads in all_grads], axis=0)
                # 更新参数
                layer.parameters()[param_idx] -= self.model.optimizer.lr * avg_grad


class MLP:
    """ 神经网络模型（类似torch.nn.Module） """

    def __init__(self, layers: List[Module]):
        self.layers = layers
        self.optimizer = None
        self.loss_fn = None
        self.parallel_enabled = False

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: Tensor) -> None:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def compile(self, optimizer: Optimizer,
                loss_fn: Callable[[Tensor, Tensor], float],
                parallel: bool = False) -> None:
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.parallel_enabled = parallel

    def fit(self, X: Tensor, y: Tensor, epochs: int,
            batch_size: int = 32, val_data: tuple = None,
            early_stopping: EarlyStopping = None) -> None:
        # 数据预处理
        dataset = DataLoader(X, y, batch_size)

        for epoch in range(epochs):
            # 并行训练分支
            if self.parallel_enabled:
                psgd = ParallelSGD(self, 4)
                psgd.step(dataset)
            else:
                # 普通训练
                for X_batch, y_batch in dataset:
                    # 前向传播
                    output = self.forward(X_batch)
                    loss = self.loss_fn(output, y_batch)

                    # 反向传播
                    grad = self.loss_derivative(output, y_batch)
                    self.backward(grad)

                    # 优化器更新
                    self.optimizer.step()

            # 早停检查
            if val_data and early_stopping:
                val_loss = self.evaluate(*val_data)
                if early_stopping(val_loss):
                    print(f"Early stopping at epoch {epoch}")
                    break

    def evaluate(self, X: Tensor, y: Tensor) -> float:
        # 评估逻辑
        outputs = np.array([self.forward(x) for x in X])
        return self.loss_fn(outputs, y)


class DataLoader:
    """ 数据加载器（类似torch.utils.data.DataLoader） """

    def __init__(self, X: Tensor, y: Tensor, batch_size: int = 32):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.n_samples = X.shape[0]

    def batches(self):
        indices = np.random.permutation(self.n_samples)
        for i in range(0, self.n_samples, self.batch_size):
            batch_idx = indices[i:i + self.batch_size]
            yield self.X[batch_idx], self.y[batch_idx]


if __name__ == "__main__":
    # 示例用法（MNIST分类）
    # 定义模型结构
    model = MLP([
        Linear(784, 256, 'he'),
        ReLU(),
        Linear(256, 10),
        Softmax()
    ])

    # 定义优化器和损失函数
    optimizer = Adam(model.parameters(), lr=0.001)
    loss_fn = cross_entropy_loss

    # 编译模型（启用并行训练）
    model.compile(optimizer,loss_fn,True)

    # 初始化早停机制
    early_stop = EarlyStopping(patience=3, delta=0.001)

    # 开始训练（假设X_train, y_train已加载）
    model.fit(X_train, y_train, epochs=100,
              val_data=(X_val, y_val),
              early_stopping=early_stop)
