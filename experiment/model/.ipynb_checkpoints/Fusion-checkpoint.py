import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_output):
        attn_weights = torch.softmax(self.attention(lstm_output), dim=1)
        attn_output = torch.sum(lstm_output * attn_weights, dim=1)
        return attn_output, attn_weights

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
        # print(f"KANLayer combined input shape: {combined_input.shape}")
        x = self.relu(self.fc1(combined_input))
        # print(f"KANLayer after fc1 shape: {x.shape}")
        x = self.relu(self.fc2(x))
        # print(f"KANLayer after fc2 shape: {x.shape}")
        return x

class LSTM_Attention_KAN_Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, lstm_layers, transformer_layers, num_heads, output_dim, knowledge_dim, dropout=0.2):
        super(LSTM_Attention_KAN_Transformer, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.transformer_layers = transformer_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, lstm_layers, batch_first=True, dropout=dropout)
        self.attention = Attention(hidden_dim)
        self.kan = KANLayer(hidden_dim, hidden_dim, hidden_dim)  
        self.fc_transformer_input = nn.Linear(hidden_dim * 2, hidden_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout) for _ in range(transformer_layers)
        ])
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.knowledge_embedding = nn.Linear(knowledge_dim, hidden_dim)

    def forward(self, x, knowledge):
        lstm_output, (h_n, c_n) = self.lstm(x)
        # print(f"LSTM output shape: {lstm_output.shape}")
        attn_output, attn_weights = self.attention(lstm_output)
        # print(f"Attention output shape: {attn_output.shape}")
        
        # Knowledge embedding
        knowledge_embedded = self.knowledge_embedding(knowledge)
        # print(f"Knowledge embedded shape: {knowledge_embedded.shape}")
        kan_output = self.kan(attn_output, knowledge_embedded)
        # print(f"KAN output shape: {kan_output.shape}")
        
        # 将KAN输出添加到Transformer输入
        kan_output = kan_output.unsqueeze(1).repeat(1, x.size(1), 1)  # [batch_size, seq_len, hidden_dim]
        transformer_input = torch.cat((lstm_output, kan_output), dim=2)
        # print(f"Transformer input shape: {transformer_input.shape}")

        # 调整transformer_input的维度以匹配TransformerBlock的期望输入
        transformer_input = self.fc_transformer_input(transformer_input)  
        # print(f"Transformed Transformer input shape: {transformer_input.shape}")
        
        transformer_output = transformer_input.permute(1, 0, 2)  # [seq_len, batch_size, hidden_dim]
        for transformer_block in self.transformer_blocks:
            transformer_output = transformer_block(transformer_output)
        transformer_output = transformer_output.permute(1, 0, 2)  # [batch_size, seq_len, hidden_dim]
        
        final_output = self.fc(transformer_output[:, -1, :])
        return final_output, attn_weights
