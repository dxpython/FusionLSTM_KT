# train_electricity.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from electricity_dataloader import get_dataloader
from Fusion import LSTM_Attention_KAN_Transformer

# 超参数设置
csv_file = "/root/autodl-tmp/fusion/dataset/electricity.csv"
batch_size = 32
seq_length = 192
knowledge_dim = 3


input_dim = 320  
hidden_dim = 64
lstm_layers = 2
transformer_layers = 2
num_heads = 4
output_dim = 1
dropout = 0.2
num_epochs = 50
learning_rate = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


dataloader = get_dataloader(csv_file, batch_size=batch_size, seq_length=seq_length, knowledge_dim=knowledge_dim, shuffle=True)

model = LSTM_Attention_KAN_Transformer(input_dim, hidden_dim, lstm_layers, transformer_layers, num_heads, output_dim, knowledge_dim, dropout)
model.to(device)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0


    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
        x, knowledge, y = batch
        x = x.to(device)
        knowledge = knowledge.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        output, attn_weights = model(x, knowledge)

        loss = criterion(output.squeeze(), y)
        loss.backward()
        optimizer.step()

        batch_size_curr = x.size(0)
        total_loss += loss.item() * batch_size_curr
        total_samples += batch_size_curr

        batch_mse = nn.functional.mse_loss(output.squeeze(), y, reduction='sum').item()
        batch_mae = nn.functional.l1_loss(output.squeeze(), y, reduction='sum').item()
        total_mse += batch_mse
        total_mae += batch_mae

    avg_loss = total_loss / total_samples
    avg_mse = total_mse / total_samples
    avg_mae = total_mae / total_samples

    print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}, MSE = {avg_mse:.4f}, MAE = {avg_mae:.4f}")
