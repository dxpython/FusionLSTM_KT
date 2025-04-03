import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + ff_output)
        return x

class LSTM_Transformer_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, lstm_layers, transformer_layers, num_heads, output_dim, dropout=0.2):
        super(LSTM_Transformer_Model, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, lstm_layers, batch_first=True, dropout=dropout)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout) for _ in range(transformer_layers)
        ])
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_output, (h_n, c_n) = self.lstm(x)
        transformer_input = lstm_output.permute(1, 0, 2)  # [seq_len, batch_size, hidden_dim]
        for transformer_block in self.transformer_blocks:
            transformer_input = transformer_block(transformer_input)
        transformer_output = transformer_input.permute(1, 0, 2)  # [batch_size, seq_len, hidden_dim]
        final_output = self.fc(transformer_output[:, -1, :])  # 使用最后一个时间步的输出
        return final_output
