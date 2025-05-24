import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


class MNISTDataset(Dataset):
    """
    自定义数据集类，用于从 CSV 文件中加载 MNIST 数据。
    该类继承自 torch.utils.data.Dataset，需要实现 __len__ 和 __getitem__ 方法。
    """

    def __init__(self, csv_file):
        """
        初始化函数，读取 CSV 文件并将数据转换为 PyTorch 张量。
        :param csv_file: CSV 文件的路径
        """
        self.data = pd.read_csv(csv_file)
        self.labels = torch.tensor(self.data.iloc[:, 0].values, dtype=torch.long)
        self.images = torch.tensor(self.data.iloc[:, 1:].values, dtype=torch.float32) / 255.0

    def __len__(self):
        """
        返回数据集的长度。
        :return: 数据集的长度
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        根据索引返回图像和对应的标签。
        :param idx: 索引
        :return: 图像和对应的标签
        """
        image = self.images[idx].view(1, 28, 28)
        label = self.labels[idx]
        return image, label


class MLP(nn.Module):
    """
    多层感知机（MLP）模型类，继承自 nn.Module。
    该模型由多个全连接层和激活函数组成。
    """

    def __init__(self, input_size, hidden_sizes, output_size, activation='relu'):
        """
        初始化函数，定义模型的结构。
        :param input_size: 输入层的大小
        :param hidden_sizes: 隐藏层大小的列表
        :param output_size: 输出层的大小
        :param activation: 激活函数的类型，默认为 'relu'
        """
        super(MLP, self).__init__()
        layers = []
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'sigmoid':
                    layers.append(nn.Sigmoid())
                elif activation == 'tanh':
                    layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播函数，定义模型的前向计算过程。
        :param x: 输入数据
        :return: 模型的输出
        """
        # logging.info(f"Input shape to MLP: {x.shape}")
        output = self.model(x)
        # logging.info(f"Output shape from MLP: {output.shape}")
        return output


def initialize_weights(model, init_type='xavier'):
    """
    初始化模型的权重。
    :param model: 要初始化的模型
    :param init_type: 初始化方法的类型，默认为 'xavier'
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if init_type == 'xavier':
                nn.init.xavier_uniform_(m.weight)
            elif init_type == 'kaiming':
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif init_type == 'zeros':
                nn.init.zeros_(m.weight)


def get_optimizer(model, optimizer_type='adam', lr=0.001, momentum=0.9):
    """
    根据指定的优化器类型和学习率返回优化器。
    :param model: 要优化的模型
    :param optimizer_type: 优化器的类型，默认为 'adam'
    :param lr: 学习率，默认为 0.001
    :param momentum: 动量，仅在使用 SGD 优化器时有效，默认为 0.9
    :return: 优化器对象
    """
    if optimizer_type == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer_type == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=lr)
    elif optimizer_type == 'adam':
        return optim.Adam(model.parameters(), lr=lr)


def get_regularization_loss(model, reg_type='l2', reg_lambda=0.001):
    """
    计算正则化损失。
    :param model: 模型
    :param reg_type: 正则化类型，默认为 'l2'
    :param reg_lambda: 正则化系数，默认为 0.001
    :return: 正则化损失
    """
    if reg_type == 'l1':
        return reg_lambda * sum(p.abs().sum() for p in model.parameters())
    elif reg_type == 'l2':
        return reg_lambda * sum(p.pow(2).sum() for p in model.parameters())
    elif reg_type == 'elastic':
        l1_loss = sum(p.abs().sum() for p in model.parameters())
        l2_loss = sum(p.pow(2).sum() for p in model.parameters())
        return reg_lambda * (0.5 * l1_loss + 0.5 * l2_loss)
    return 0


def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device,
                reg_type='l2', reg_lambda=0.001, writer=None):
    """
    训练模型的函数。
    :param model: 要训练的模型
    :param train_loader: 训练数据加载器
    :param test_loader: 测试数据加载器
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param num_epochs: 训练的轮数
    :param device: 设备（CPU 或 GPU）
    :param reg_type: 正则化类型，默认为 'l2'
    :param reg_lambda: 正则化系数，默认为 0.001
    :param writer: TensorBoard 写入器，默认为 None
    """
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            images = images.view(images.size(0), -1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            reg_loss = get_regularization_loss(model, reg_type, reg_lambda)
            total_loss = loss + reg_loss
            total_loss.backward()

            # 记录各层梯度变化
            for name, param in model.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

            optimizer.step()
            running_loss += total_loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # 计算训练集准确率
        train_accuracy = 100 * correct / total
        avg_loss = running_loss / len(train_loader)

        # 记录训练损失和准确率
        if writer:
            writer.add_scalar('Training Loss', avg_loss, epoch)
            writer.add_scalar('Training Accuracy', train_accuracy, epoch)

        # 记录学习率
        lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning Rate', lr, epoch)

        # 记录各层权重变化
        for name, param in model.named_parameters():
            writer.add_histogram(f'Weights/{name}', param, epoch)

        # 在测试集上评估模型
        test_accuracy , cm= evaluate_model(model, test_loader, device, writer, epoch)

        logging.info(
            f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%')


def evaluate_model(model, test_loader, device, writer=None, epoch=None):
    """
    评估模型在测试集上的性能。

    Args:
        model (nn.Module): 要评估的模型。
        test_loader (DataLoader): 测试数据加载器。
        device (torch.device): 评估设备（如'cuda'或'cpu'）。
        writer (SummaryWriter, optional): TensorBoard的SummaryWriter对象，用于记录测试指标。
        epoch (int, optional): 当前训练轮数，用于在TensorBoard中记录不同轮数的测试指标。

    Returns:
        float: 模型在测试集上的准确率
        np.ndarray: 混淆矩阵
    """
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            images = images.view(images.size(0), -1)
            # logging.info(f"Test batch input image shape: {images.shape}, label shape: {labels.shape}")
            outputs = model(images)
            # logging.info(f"Test batch model output shape: {outputs.shape}")
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    accuracy = 100 * correct / total
    cm = confusion_matrix(all_labels, all_preds)
    if writer and epoch is not None:
        writer.add_scalar('Test Accuracy', accuracy, epoch)
    return accuracy, cm


def plot_confusion_matrix(cm, classes):

    """
     以日志形式输出混淆矩阵
     :param cm: 混淆矩阵数据
     :param classes: 类别标签列表
     """
    logging.info("Confusion Matrix:")
    header = "    " + " ".join(f"{c:5}" for c in classes)
    logging.info(header)
    for i, row in enumerate(cm):
        row_str = f"{classes[i]:5}" + " ".join(f"{v:5}" for v in row)
        logging.info(row_str)


if __name__ == "__main__":
    train_csv_path = 'MNIST_data/mnist_train.csv'
    test_csv_path = 'MNIST_data/mnist_test.csv'

    train_dataset = MNISTDataset(train_csv_path)
    test_dataset = MNISTDataset(test_csv_path)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_size = 28 * 28
    hidden_sizes = [128, 64]
    output_size = 10
    activation = 'relu'
    init_type = 'xavier'
    optimizer_type = 'adam'
    lr = 0.001
    num_epochs = 10
    reg_type = 'l2'
    reg_lambda = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLP(input_size, hidden_sizes, output_size, activation).to(device)
    logging.info(f"Model structure:\n{model}")

    initialize_weights(model, init_type)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, optimizer_type, lr)

    writer = SummaryWriter('runs/mlp_experiment')

    train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device,
                reg_type, reg_lambda, writer)

    torch.save(model.state_dict(), 'v1_torch_mlp_model.pth')

    test_accuracy, cm = evaluate_model(model, test_loader, device)

    classes = [str(i) for i in range(10)]
    plot_confusion_matrix(cm, classes)

    writer.close()
