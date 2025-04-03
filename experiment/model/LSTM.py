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

class LSTM_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(LSTM_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_output, (h_n, c_n) = self.lstm(x)
        attn_output, attn_weights = self.attention(lstm_output)
        output = self.fc(attn_output)
        return output, attn_weights
