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

# 自定义数据集类，用于从 CSV 文件中加载 MNIST 数据
class MNISTDataset(Dataset):
    def __init__(self, csv_file):
        # 读取 CSV 文件
        self.data = pd.read_csv(csv_file)
        # 提取标签
        self.labels = torch.tensor(self.data.iloc[:, 0].values, dtype=torch.long)
        # 提取图像数据，并进行归一化处理
        self.images = torch.tensor(self.data.iloc[:, 1:].values, dtype=torch.float32) / 255.0

    def __len__(self):
        # 返回数据集的长度
        return len(self.data)

    def __getitem__(self, idx):
        # 根据索引返回图像和对应的标签
        image = self.images[idx].view(1, 28, 28)
        label = self.labels[idx]
        return image, label

#  定义一个简单的多层感知机
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu'):
        super(MLP, self).__init__()
        layers = []
        sizes = [input_size] + hidden_sizes + [output_size]
        # 根据隐藏层大小列表，创建多层感知器的层
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
    # 定义前向传播方法，返回模型输出
    def forward(self, x):
        return self.model(x)

#  初始化权重
def initialize_weights(model, init_type='xavier'):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if init_type == 'xavier':
                nn.init.xavier_uniform_(m.weight)
            elif init_type == 'kaiming':
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif init_type == 'zeros':
                nn.init.zeros_(m.weight)

#  获取优化器
def get_optimizer(model, optimizer_type='adam', lr=0.001, momentum=0.9):
    if optimizer_type == 'sgd':
        # 创建SGD优化器
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer_type == 'rmsprop':
        # 创建RMSprop优化器
        return optim.RMSprop(model.parameters(), lr=lr)
    elif optimizer_type == 'adam':
        # 创建Adam优化器
        return optim.Adam(model.parameters(), lr=lr)

#  计算正则化损失
def get_regularization_loss(model, reg_type='l2', reg_lambda=0.001):
    if reg_type == 'l1':
        #  L1 正则化  计算L1正则化损失：遍历模型所有参数，取绝对值求和后乘以正则化系数reg_lambda，用于约束模型参数稀疏性。
        return reg_lambda * sum(p.abs().sum() for p in model.parameters())
    elif reg_type == 'l2':
        #  L2 正则化  计算L2正则化损失：遍历模型所有参数，取平方求和后乘以正则化系数reg_lambda，用于约束模型参数的权重大小。
        return reg_lambda * sum(p.pow(2).sum() for p in model.parameters())
    elif reg_type == 'elastic':
        #  弹性正则化  计算弹性正则化损失：同时使用L1和L2正则化，取L1和L2正则化损失之和乘以正则化系数reg_lambda。
        l1_loss = sum(p.abs().sum() for p in model.parameters())
        l2_loss = sum(p.pow(2).sum() for p in model.parameters())
        return reg_lambda * (0.5 * l1_loss + 0.5 * l2_loss)
    return 0

# 训练模型
def train_model(model, train_loader, criterion, optimizer, num_epochs, device,
                reg_type='l2', reg_lambda=0.001, writer=None):
    # 训练模型
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            images = images.view(images.size(0), -1)
            optimizer.zero_grad()
            outputs = model(images)
            # 计算损失和正则化损失，并计算总损失
            loss = criterion(outputs, labels)
            reg_loss = get_regularization_loss(model, reg_type, reg_lambda)
            total_loss = loss + reg_loss
            # 反向传播和优化
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
        avg_loss = running_loss / len(train_loader)
        if writer:
            writer.add_scalar('Training Loss', avg_loss, epoch)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')

# 测试模型
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            images = images.view(images.size(0), -1)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    cm = confusion_matrix(all_labels, all_preds)
    return cm

# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


if __name__ == "__main__":
    # 请根据实际情况修改文件路径
    train_csv_path = 'MNIST_data/mnist_train.csv'
    test_csv_path = 'MNIST_data/mnist_test.csv'

    # 创建训练集和测试集的数据集对象
    train_dataset = MNISTDataset(train_csv_path)
    test_dataset = MNISTDataset(test_csv_path)

    # 创建训练数据加载器和测试数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 定义模型参数
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

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')

    model = MLP(input_size, hidden_sizes, output_size, activation).to(device)
    initialize_weights(model, init_type)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, optimizer_type, lr)

    writer = SummaryWriter('runs/mlp_experiment')

    # 训练模型
    train_model(model, train_loader, criterion, optimizer, num_epochs, device,
                reg_type, reg_lambda, writer)

    # 保存模型
    torch.save(model.state_dict(), 'v_torch_mlp_model.pth')

    # 评估模型
    cm = evaluate_model(model, test_loader, device)

    # 绘制混淆矩阵
    classes = [str(i) for i in range(10)]
    plot_confusion_matrix(cm, classes)

    writer.close()
    