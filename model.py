import torch
import torch.nn as nn
from load_dataset import ds_smarteye
from torch.nn.utils.rnn import pad_sequence
import math

def collate_fn(batch):
    # 从 batch 中获取 data 和 label
    data = [item['data'] for item in batch]
    labels = [item['label'] for item in batch]
    seq_lengths = [item['seq_length'] for item in batch]
    padded_data = pad_sequence(data, batch_first=True)
    
    labels = torch.stack(labels)
    
    return {
        'data': padded_data,
        'label': labels,
        'seq_lengths': torch.tensor(seq_lengths)
    }

class LSTMModel(nn.Module):
    def __init__(self, input_size=19, hidden_size=64, num_layers=1):
        super(LSTMModel, self).__init__()
        # self.fc1 = nn.Linear()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, seq_lengths):
        packed_input = nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hn, cn) = self.lstm(packed_input)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        batch_size = x.size(0)
        idx = torch.arange(batch_size)
        last_valid_outputs = lstm_out[idx, seq_lengths - 1]

        output = self.fc(last_valid_outputs)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * (-math.log(10000.0) / hidden_size))
        pe = torch.zeros(max_len, hidden_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe.shape[1] // 2])

        pe = pe.unsqueeze(1)  # Shape: [max_len, 1, hidden_size]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, hidden_size]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, input_size=19, hidden_size=512, num_layers=1, nhead=64, dim_feedforward=2048, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, seq_lengths):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, input_size]
            seq_lengths: Tensor of shape [batch_size], containing the lengths of each sequence
        """
        # Transpose x to shape [seq_len, batch_size, input_size]
        x = x.transpose(0, 1)

        # Embed and apply positional encoding
        x = self.embedding(x) * math.sqrt(self.hidden_size)
        x = self.pos_encoder(x)

        # Create key padding mask
        seq_len = x.size(0)
        batch_size = x.size(1)
        device = x.device

        # Move seq_lengths to the same device as x
        seq_lengths = seq_lengths.to(device)

        # Mask has True where we want to mask (i.e., padding positions)
        mask = torch.arange(seq_len, device=device).unsqueeze(1) >= seq_lengths.unsqueeze(0)
        mask = mask.transpose(0, 1)  # Shape: [batch_size, seq_len]

        # Pass through the Transformer encoder
        output = self.transformer_encoder(x, src_key_padding_mask=mask)

        # Get the outputs at the last valid time steps
        last_outputs = output[seq_lengths - 1, torch.arange(batch_size)]

        # Final linear layer
        output = self.fc(last_outputs)
        return output
    

if __name__ == '__main__':
    for participant_id in range(1, 38):

        json_path = ".\\label_1.json"

        print(f"Training model for participant {participant_id}")
        
        train_dataset = ds_smarteye(json_path, participant_id, train=True)
        test_dataset = ds_smarteye(json_path, participant_id, train=False)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
        
        model = LSTMModel()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
