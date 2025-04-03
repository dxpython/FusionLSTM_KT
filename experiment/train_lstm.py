import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from model.LSTM import LSTM_Attention
from data_loader import load_data
from sklearn.metrics import mean_squared_error, mean_absolute_error


def create_sliding_windows(x_data, window_size):
    windows = []
    for i in range(x_data.size(0) - window_size + 1):
        windows.append(x_data[i:i + window_size])
    return torch.stack(windows)


# 训练参数
epochs = 200
learning_rate = 0.001
batch_size = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

time_stamp, w_228, v_228 = load_data()
x_data = torch.tensor(w_228.values, dtype=torch.float32)
x_data = (x_data - x_data.mean(dim=0)) / x_data.std(dim=0)
x_data = x_data.to(device)

window_size = 10
x_data_windows = create_sliding_windows(x_data, window_size)


def create_batches(x_data, batch_size):
    for i in range(0, x_data.size(0), batch_size):
        yield x_data[i:i + batch_size]


input_dim = 228
hidden_dim = 128
num_layers = 2
output_dim = 228
model = LSTM_Attention(input_dim, hidden_dim, num_layers, output_dim).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

criterion = nn.SmoothL1Loss()


def compute_metrics(y_true, y_pred):
    acc = ((y_true - y_pred).abs() < 0.1).float().mean().item()
    mse = mean_squared_error(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
    mae = mean_absolute_error(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
    return acc, mse, mae


losses, accuracies, mses, maes = [], [], [], []
for epoch in range(epochs):
    model.train()
    epoch_loss, epoch_acc, epoch_mse, epoch_mae = 0, 0, 0, 0
    batch_count = 0

    for batch in create_batches(x_data_windows, batch_size):
        optimizer.zero_grad()
        # print(f"Batch shape: {batch.shape}")
        outputs, _ = model(batch)
        loss = criterion(outputs, batch[:, -1, :])
        loss.backward()
        optimizer.step()

        acc, mse, mae = compute_metrics(batch[:, -1, :], outputs)
        epoch_loss += loss.item()
        epoch_acc += acc
        epoch_mse += mse
        epoch_mae += mae
        batch_count += 1

    losses.append(epoch_loss / batch_count)
    accuracies.append(epoch_acc / batch_count)
    mses.append(epoch_mse / batch_count)
    maes.append(epoch_mae / batch_count)

    print(
        f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / batch_count}, Acc: {epoch_acc / batch_count}, MSE: {epoch_mse / batch_count}, MAE: {epoch_mae / batch_count}')


def plot_metrics(metric_values, metric_name):
    plt.figure()
    plt.plot(metric_values)
    plt.title(f'{metric_name} over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.grid(True)
    plt.savefig(f'./LSTM/{metric_name}.png')
    plt.show()


# 绘制评价指标图
plot_metrics(losses, 'Loss')
plot_metrics(accuracies, 'Accuracy')
plot_metrics(mses, 'MSE')
plot_metrics(maes, 'MAE')
