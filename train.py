import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from models.gaze_mobilenetv3 import GazeMobileNetV3

def load_dataset(train_path, val_ratio=0.2):
    data = np.load(train_path)
    X = torch.tensor(data['images'], dtype=torch.float32)  # shape: [N, 2, 3, H, W]
    y = torch.tensor(data['gaze'], dtype=torch.float32)    # shape: [N, 2]
    
    dataset = TensorDataset(X, y)
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    return random_split(dataset, [train_size, val_size])

def train_model(model, train_loader, val_loader, epochs, device):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_losses = []
    val_losses = []

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # 🟡 2枚の画像（過去・現在）→ 1枚の6ch画像に結合（過去に0.3倍）
            past = inputs[:, 0] * 0.3      # [B, 3, H, W]
            present = inputs[:, 1]         # [B, 3, H, W]
            inputs = torch.cat([past, present], dim=1)  # [B, 6, H, W]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                # 🟡 同様にバリデーションでも加工
                past = inputs[:, 0] * 0.3
                present = inputs[:, 1]
                inputs = torch.cat([past, present], dim=1)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch}/{epochs}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        # モデル保存
        torch.save(model.state_dict(), f"checkpoints/model_epoch{epoch}.pth")

        # グラフ保存（2エポック目から毎回更新）
        if epoch >= 2:
            plt.figure()
            plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
            plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
            plt.xlabel("Epoch")
            plt.ylabel("Loss (MSE)")
            plt.legend()
            plt.grid(True)
            plt.savefig("checkpoints/loss.png")
            plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--data", type=str, default="data/processed/gtea_train.npz", help="Path to npz dataset")
    args = parser.parse_args()

    train_set, val_set = load_dataset(args.data)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)

    model = GazeMobileNetV3()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_model(model, train_loader, val_loader, args.epochs, device)
