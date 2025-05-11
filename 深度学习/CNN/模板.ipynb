#%% [markdown]
# ## 1. 环境准备与数据加载
# 注意：确保已安装torch==2.0.0+cu118和pandas>=1.5.3
# 理论补充：图像数据归一化可以加速模型收敛，防止梯度爆炸

import torch
import torch.nn as nn
import torch.utils.data as data
import pandas as pd
from torch.utils.data import DataLoader, random_split, TensorDataset

#%% [markdown]
# ### 1.1 加载原始数据
# 异常情况处理：当文件路径错误时会抛出FileNotFoundError
try:
    # 加载训练图像（假设是未标准化的[0,255]范围）
    # 文件格式说明：train-images.pt应为FloatTensor类型
    images_raw = torch.load('train-images.pt')  # shape: [60000, 28, 28]
    
    # 加载标签数据（CSV文件第一列为标签）
    labels_raw = pd.read_csv('train-labels.csv').values[:, 0]  # shape: [60000]
except FileNotFoundError as e:
    print(f"文件加载失败：{str(e)}")
    raise

#%% [markdown]
# ### 1.2 数据预处理
# 理论说明：归一化使数据分布均值为0，标准差为1，加速训练收敛
def normalize_images(images):
    """将[0,255]范围的图像归一化到[-1,1]范围"""
    # 计算原理：归一化公式 x' = (x - μ)/σ
    # 此处使用MNIST通用参数，若自定义数据集需重新计算
    mean = 0.5
    std = 0.5
    return (images/255.0 - mean) / std

# 增加通道维度（从[60000,28,28]变为[60000,1,28,28]）
images = images_raw.unsqueeze(1).float()  # 添加通道维度
images = normalize_images(images)         # 归一化处理

# 转换标签为LongTensor类型
labels = torch.tensor(labels_raw, dtype=torch.long)

#%% [markdown]
# ### 1.3 创建数据集
# 注意事项：使用TensorDataset确保数据对齐
# 异常处理：检查数据尺寸是否匹配
if len(images) != len(labels):
    raise ValueError("图像与标签数量不匹配！")

dataset = TensorDataset(images, labels)

# 8:2分割训练集和验证集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(
    dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)  # 固定随机种子保证可重复性
)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)

#%% [markdown]
# ## 2. CNN模型定义
# 网络结构说明：经典LeNet改进结构，适合28x28图像
# 调整建议：可通过增加卷积层深度提升性能

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 卷积层组
        self.conv_layers = nn.Sequential(
            # 输入通道1，输出32，卷积核3x3，padding保持尺寸
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 输出：32x28x28
            nn.ReLU(),
            nn.MaxPool2d(2),  # 下采样：32x14x14
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 输出：64x14x14
            nn.ReLU(),
            nn.MaxPool2d(2)   # 输出：64x7x7
        )
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),  # 特征展开
            nn.ReLU(),
            nn.Dropout(0.5),         # 防止过拟合
            nn.Linear(128, 10)       # 输出10个类别
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)    # 展平特征图
        x = self.fc_layers(x)
        return x

#%% [markdown]
# ## 3. 模型初始化
# 设备选择：自动检测GPU加速
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
print("模型结构：\n", model)

#%% [markdown]
# ## 4. 训练配置
# 理论说明：交叉熵损失适合分类任务，Adam优化器具有自适应学习率
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

#%% [markdown]
# ## 5. 训练与验证循环
# 注意事项：train/eval模式切换影响BN和Dropout行为
# 扩展功能：可添加学习率调度器（如ReduceLROnPlateau）

for epoch in range(5):
    # === 训练阶段 ===
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计指标
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # 每50个batch打印进度
        if batch_idx % 50 == 0:
            train_loss = running_loss / (batch_idx+1)
            train_acc = correct / total
            print(f"Epoch {epoch+1}, Batch {batch_idx}, "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Train Accuracy: {train_acc*100:.2f}%")
    
    # === 验证阶段 ===
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()
    
    val_loss /= len(val_loader)
    val_acc = val_correct / val_total
    print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc*100:.2f}%")

#%% [markdown]
# ## 6. 测试与提交
# 注意事项：测试数据需使用相同的归一化参数
test_images = torch.load('test_images.pt')  # shape: [10000,28,28]

# 预处理（与训练数据一致）
test_images = test_images.unsqueeze(1).float()
test_images = normalize_images(test_images).to(device)

# 推理预测
model.eval()
with torch.no_grad():
    outputs = model(test_images)
    predictions = outputs.argmax(dim=1)

# 生成提交文件
df_test = pd.DataFrame({"label": predictions.cpu().numpy()})
df_test.to_csv("submission.csv", index_label="id")
print("预测结果已保存至submission.csv")
