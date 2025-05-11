import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd

# ================================================
# 小題1：數據讀取與預處理
# ================================================

# 讀取數據集
train_images = torch.load('train-images.pt')  # [60000,28,28]
train_labels = pd.read_csv('train-labels.csv').values.squeeze()  # [60000]
test_images = torch.load('test-images.pt')    # [10000,28,28]

# 歸一化處理 (均值0.5，標準差0.5)
transform = torch.nn.Sequential(
    transforms.Lambda(lambda x: x.float() / 255.0),  # 轉換為0-1範圍
    transforms.Normalize((0.5,), (0.5,))             # 歸一化到[-1,1]
)

# 自定義數據集類
class MNISTDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = torch.from_numpy(labels).long()
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx].unsqueeze(0)  # 增加通道維度 [1,28,28]
        return transform(img), self.labels[idx]

# 創建完整訓練集並拆分
full_dataset = MNISTDataset(train_images, train_labels)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# 創建數據加載器
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)

# ================================================
# 小題2：構建CNN模型
# ================================================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 卷積層配置
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 保持尺寸不變
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 池化層配置
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全連接層
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 兩次池化後尺寸28→14→7
        self.fc2 = nn.Linear(128, 10)
        # 激活函數
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 卷積層1
        x = self.conv1(x)    # [b,1,28,28] → [b,32,28,28]
        x = self.relu(x)
        x = self.pool(x)     # → [b,32,14,14]
        
        # 卷積層2
        x = self.conv2(x)    # → [b,64,14,14]
        x = self.relu(x)
        x = self.pool(x)      # → [b,64,7,7]
        
        # 展平特徵圖
        x = x.view(x.size(0), -1)
        
        # 全連接層
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
print(model)

# ================================================
# 小題3：模型訓練
# ================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    # 訓練階段
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # 前向傳播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向傳播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 統計數據
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 每50個batch打印進度
        if batch_idx % 50 == 0:
            batch_acc = 100 * correct / total
            print(f"Epoch {epoch+1}, Batch {batch_idx}, Train Loss: {loss.item():.4f}, Train Acc: {batch_acc:.2f}%")
    
    # 驗證階段
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    # 打印epoch結果
    avg_loss = total_loss / len(train_loader)
    train_acc = 100 * correct / total
    val_acc = 100 * val_correct / val_total
    print(f"Epoch {epoch+1} Result:")
    print(f"Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.2f}%")
    print("-"*50)

# ================================================
# 測試集預測
# ================================================
# 加載並處理測試數據
test_images = transform(test_images.unsqueeze(1))  # 增加通道維度並歸一化

model.eval()
with torch.no_grad():
    test_images = test_images.to(device)
    outputs = model(test_images)
    predictions = outputs.argmax(dim=1)
    
# 保存結果
df_test = pd.DataFrame({"label": predictions.cpu().numpy()})
df_test.to_csv("submission.csv", index_label="id")
