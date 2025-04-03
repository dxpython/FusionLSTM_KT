import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from data_loader import load_data
from sklearn.metrics import mean_squared_error, mean_absolute_error
from model.lstm_Transformer_KAN import LSTM_KAN_Transformer

def create_sliding_windows(x_data, window_size):
    windows = []
    for i in range(x_data.size(0) - window_size + 1):
        windows.append(x_data[i:i + window_size])
    return torch.stack(windows)

epochs = 200 
learning_rate = 0.01  
batch_size = 64  
knowledge_dim = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

time_stamp, w_228, v_228 = load_data()
x_data = torch.tensor(w_228.values, dtype=torch.float32)
x_data = (x_data - x_data.mean(dim=0)) / x_data.std(dim=0)
x_data = x_data.to(device)

knowledge_data = torch.randn(x_data.size(0), knowledge_dim).to(device)

window_size = 10
x_data_windows = create_sliding_windows(x_data, window_size)

def create_batches(x_data, knowledge_data, batch_size):
    for i in range(0, x_data.size(0), batch_size):
        end_idx = i + batch_size
        if end_idx > x_data.size(0):
            end_idx = x_data.size(0)
        yield x_data[i:end_idx], knowledge_data[i:end_idx]

input_dim = 228
hidden_dim = 128  
lstm_layers = 1  
transformer_layers = 1  
num_heads = 4
output_dim = 228
model = LSTM_KAN_Transformer(input_dim, hidden_dim, lstm_layers, transformer_layers, num_heads, output_dim, knowledge_dim).to(device)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # 增加weight decay
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.8)

criterion = nn.SmoothL1Loss()

def compute_metrics(y_true, y_pred):
    acc = ((y_true - y_pred).abs() < 0.1).float().mean().item()
    mse = mean_squared_error(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
    mae = mean_absolute_error(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
    return acc, mse, mae

output_dir = './LSTM_Transformer_KAN'
os.makedirs(output_dir, exist_ok=True)

losses, accuracies, mses, maes = [], [], [], []
for epoch in range(epochs):
    model.train()
    epoch_loss, epoch_acc, epoch_mse, epoch_mae = 0, 0, 0, 0
    batch_count = 0

    for batch, knowledge_batch in create_batches(x_data_windows, knowledge_data, batch_size):
        optimizer.zero_grad()
        outputs = model(batch, knowledge_batch)

        loss = criterion(outputs, batch[:, -1, :])
        loss.backward()
        optimizer.step()

        acc, mse, mae = compute_metrics(batch[:, -1, :], outputs)
        epoch_loss += loss.item()
        epoch_acc += acc
        epoch_mse += mse
        epoch_mae += mae
        batch_count += 1

    scheduler.step(epoch_loss / batch_count)

    losses.append(epoch_loss / batch_count)
    accuracies.append(epoch_acc / batch_count)
    mses.append(epoch_mse / batch_count)
    maes.append(epoch_mae / batch_count)

    print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / batch_count}, Acc: {epoch_acc / batch_count}, MSE: {epoch_mse / batch_count}, MAE: {epoch_mae / batch_count}')

def plot_metrics(metric_values, metric_name):
    plt.figure()
    plt.plot(metric_values)
    plt.title(f'{metric_name} over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{metric_name}.png'))
    plt.show()

plot_metrics(losses, 'Loss')
plot_metrics(accuracies, 'Accuracy')
plot_metrics(mses, 'MSE')
plot_metrics(maes, 'MAE')
