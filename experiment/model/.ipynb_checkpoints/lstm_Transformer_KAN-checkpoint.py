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


class KANLayer(nn.Module):
    def __init__(self, input_dim, knowledge_dim, hidden_dim):
        super(KANLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim + knowledge_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x, knowledge):
        combined_input = torch.cat((x, knowledge), dim=-1)
        x = self.relu(self.fc1(combined_input))
        x = self.relu(self.fc2(x))
        return x

class LSTM_KAN_Transformer(nn.Module):  
    def __init__(self, input_dim, hidden_dim, lstm_layers, transformer_layers, num_heads, output_dim, knowledge_dim, dropout=0.2):
        super(LSTM_KAN_Transformer, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, lstm_layers, batch_first=True, dropout=dropout)
        self.kan = KANLayer(hidden_dim, hidden_dim, hidden_dim)
        self.fc_transformer_input = nn.Linear(hidden_dim * 2, hidden_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout) for _ in range(transformer_layers)
        ])
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.knowledge_embedding = nn.Linear(knowledge_dim, hidden_dim)

    def forward(self, x, knowledge):
        lstm_output, (h_n, c_n) = self.lstm(x)
        knowledge_embedded = self.knowledge_embedding(knowledge)
        kan_output = self.kan(lstm_output[:, -1, :], knowledge_embedded)  
        kan_output = kan_output.unsqueeze(1).repeat(1, x.size(1), 1)
        transformer_input = torch.cat((lstm_output, kan_output), dim=2)
        transformer_input = self.fc_transformer_input(transformer_input)
        transformer_output = transformer_input.permute(1, 0, 2)
        for transformer_block in self.transformer_blocks:
            transformer_output = transformer_block(transformer_output)
        transformer_output = transformer_output.permute(1, 0, 2)
        final_output = self.fc(transformer_output[:, -1, :])
        return final_output
