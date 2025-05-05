from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import PairedTyphoonDataset, TytModel, EarlyStopping, plot_loss_curves

import argparse
def train_model(model, train_dataset, val_dataset, epochs, batch_size, lr, device):
    early_stopping = EarlyStopping(patience=args.early_stop_patience, mode='min')  # 监控验证损失
    # 初始化记录容器
    train_losses = []
    val_losses = []
    best_loss = float('inf')
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[Epoch {epoch + 1}/{epochs}] Training")

        for x, y, z in pbar:
            x, y, z = x.to(device), y.to(device), z.to(device)  # x: [B, 5,9,40,40], y: [B,5, 4], z: [B, 4]
            optimizer.zero_grad()
            outputs = model(x, y)  # [B, 4]
            loss = criterion(outputs, z)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            pbar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f}")
        train_losses.append(avg_train_loss)

        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_val, y_val, z_val in val_loader:
                    x_val, y_val, z_val = x_val.to(device), y_val.to(device), z_val.to(device)
                    outputs = model(x_val, y_val)
                    loss = criterion(outputs, z_val)
                    val_loss += loss.item() * x_val.size(0)
            avg_val_loss = val_loss / len(val_loader.dataset)
            val_losses.append(avg_val_loss)
            print(f"          | Val   Loss: {avg_val_loss:.4f}")
            # 早停判断
            if early_stopping(val_loss):
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

            # 保存最佳模型
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), './checkpoints/best_model_' + str(epoch) + '.pth')
    # 绘制曲线
    plot_loss_curves(train_losses, val_losses)

def parse_args():
    parser = argparse.ArgumentParser(description='Typhoon Trajectory Prediction Training')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs (default: 30)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Input batch size for training (default: 16)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device for training (cuda/cpu)')

    # 数据路径参数
    parser.add_argument('--data_dir', type=str, default='./dataset',
                        help='Path to dataset directory (default: ./dataset)')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                        help='Path to checkpoints directory (default: ./checkpoints)')
    # 早停参数
    parser.add_argument('--early_stop_patience', type=int, default=30,
                        help='Number of epochs to wait before stopping when no improvement (default: 3)')
    parser.add_argument('--early_stop_delta', type=float, default=0.0,
                        help='Minimum change to qualify as improvement (default: 0.0)')
    # 添加是否为训练的参数
    parser.add_argument('--is_train', action='store_true', default=0,
                        help='是否为训练模式，设置该参数则表示处于训练模式，默认为否')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    train_era5 = np.load(f"{args.data_dir}/train.npy")
    val_era5 = np.load(f"{args.data_dir}/val.npy")
    test_era5 = np.load(f"{args.data_dir}/test.npy")
    train_bst = np.load(f"{args.data_dir}/dataset_train.npy")
    val_bst = np.load(f"{args.data_dir}/dataset_val.npy")
    test_bst = np.load(f"{args.data_dir}/dataset_test.npy")
    train_dataset = PairedTyphoonDataset(train_era5, train_bst, seq_len=5, transform='minmax')
    val_dataset = PairedTyphoonDataset(val_era5, val_bst, seq_len=5, transform='minmax')
    test_dataset = PairedTyphoonDataset(test_era5, test_bst, seq_len=5, transform='minmax')

    model = TytModel().to(args.device)
    if args.is_train:
        train_model(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device
        )
    else:
        model.load_state_dict(torch.load('./checkpoints/best_model_115.pth', map_location=args.device))
        model.to(args.device)
        model.eval()
        with torch.no_grad():
            # 创建测试数据加载器
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

            # 存储预测结果和真实值
            predictions = []
            true_values = []

            # 进行测试
            for x_test, y_test, z_test in test_loader:
                x_test = x_test.to(args.device)
                y_test = y_test.to(args.device)

                outputs = model(x_test, y_test)

                # 收集结果（移回CPU并转numpy）
                predictions.append(outputs.cpu().numpy())
                true_values.append(z_test.numpy())  # z_test本身在CPU上

            # 合并所有batch的结果
            predictions = np.concatenate(predictions, axis=0)
            true_values = np.concatenate(true_values, axis=0)

            # 保存结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            save_path = f"./results/predictions.npy"
            np.save(save_path, predictions)
            save_path = f"./results/true_values.npy"
            np.save(save_path,  true_values)
            print(f"测试结果已保存至 {save_path},时间为{timestamp}")


