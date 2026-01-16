import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
import joblib


# ==========================================
# 1. 定义神经网络结构 (Your "GNN" Node Embedding)
# ==========================================
class PricingPredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PricingPredictor, self).__init__()
        # 一个简单的 3 层全连接网络
        # 在论文里这可以被称为 "Node Embedding MLP"
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),  # 加速收敛
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.network(x)


# ==========================================
# 2. 数据加载与预处理
# ==========================================
# ==========================================
# 2. 数据加载与预处理 (修复版)
# ==========================================
def load_and_process_data(csv_path="gnn_training_data.csv"):
    print(f">>> [Data] 正在读取 {csv_path}...")
    df = pd.read_csv(csv_path)

    # 🔍 调试：打印所有列名，看看我们手里有什么牌
    print(f">>> [Debug] CSV 文件包含的列: {list(df.columns)}")

    # 1. 过滤掉无效数据
    df = df.dropna()

    # 2. 定义特征 X (Features)
    # ❌ 原来的写法 (导致报错):
    # feature_cols = ['t_cargo', 't_shore', 'cost_diff', 'dual_pi']

    # ✅ 现在的修正写法:
    # 根据你的截图，你肯定有 't_cargo', 't_shore' (或者叫 t_sp?), 'dual_pi', 'dual_mu_avg'
    # 咱们先用下面这个最保险的组合。
    # 如果代码报错说找不到 't_cargo'，请看控制台打印出来的列名，改成对应的名字即可。

    # 猜测你的列名可能是这些 (根据你的截图推断):
    candidate_cols = ['t_cargo', 't_shore', 'dual_pi', 'dual_mu_avg']

    # 自动筛选：只保留 CSV 里真正存在的列
    feature_cols = [col for col in candidate_cols if col in df.columns]

    print(f">>> [Feature] 最终使用的特征列: {feature_cols}")

    if not feature_cols:
        raise ValueError("❌ 找不到有效的特征列！请检查 CSV 列名是否和代码匹配。")

    X = df[feature_cols].values

    # 3. 定义标签 Y (Labels)
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label_mode'])
    Y = df['label_encoded'].values

    print(f">>> [Label Map] 标签对应关系: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    return X, Y, le


# ==========================================
# 3. 训练主程序
# ==========================================
def train_ai():
    # A. 准备数据
    X_raw, Y_raw, label_encoder = load_and_process_data()

    # 归一化 (对神经网络非常重要！)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # 划分训练集和测试集 (80% 训练, 20% 验证)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, Y_raw, test_size=0.2, random_state=42
    )

    # 转为 PyTorch Tensor
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)

    # B. 初始化模型
    input_dim = X_train.shape[1]
    output_dim = len(label_encoder.classes_)  # 也就是分类的数量 (通常是 2 或 3)

    model = PricingPredictor(input_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print(f"\n>>> [Train] 开始训练 (Input dim={input_dim}, Output dim={output_dim})...")

    # C. 训练循环
    epochs = 50  # 训练 50 轮
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # 前向传播
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)

        # 反向传播
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch + 1}/{epochs} | Loss: {loss.item():.4f}")

    # D. 评估模型
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t)
        _, predicted = torch.max(test_outputs, 1)
        acc = (predicted == y_test_t).sum().item() / len(y_test_t)
        print(f"\n>>> [Result] 测试集准确率: {acc * 100:.2f}%")
        print("-" * 60)
        print(classification_report(y_test_t, predicted, target_names=label_encoder.classes_))
        print("-" * 60)

    # E. 保存“大脑” (以便在 main.py 里调用)
    torch.save(model.state_dict(), "pricing_model.pth")
    joblib.dump(scaler, "scaler.pkl")  # 别忘了保存归一化参数，预测时要用
    joblib.dump(label_encoder, "label_encoder.pkl")

    print("✅ 模型已保存为 'pricing_model.pth' (及配套 scaler.pkl, label_encoder.pkl)")


if __name__ == "__main__":
    train_ai()