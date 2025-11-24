import torch
from torch import nn
import config


class TranslationEncoder(nn.Module):
    def __init__(self, vocab_size, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.EMBEDDING_DIM, padding_idx=padding_idx)
        self.gru = nn.GRU(input_size=config.EMBEDDING_DIM,
                          hidden_size=config.ENCODER_HIDDEN_DIM,
                          num_layers=config.ENCODER_LAYERS,
                          batch_first=True, bidirectional=True)

    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)
        # [batch_size, seq_len, embedding_dim]
        outputs, hidden = self.gru(embedded)
        # outputs: [batch_size, seq_len, hidden_dim*2], hidden: [num_layers*2, batch_size, hidden_dim]
        context_vector = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        # context_vector: [batch_size, 2 * hidden_dim]
        return context_vector


class TranslationDecoder(nn.Module):
    def __init__(self, vocab_size, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.EMBEDDING_DIM, padding_idx=padding_idx)
        self.gru = nn.GRU(input_size=config.EMBEDDING_DIM,
                          hidden_size=config.DECODER_HIDDEN_DIM,
                          batch_first=True)
        self.linear = nn.Linear(config.DECODER_HIDDEN_DIM, vocab_size)

    def forward(self, x, hidden_0):
        # x: [batch_size, 1] hidden_0: [1, batch_size, hidden_dim]
        embedded = self.embedding(x)
        # [batch_size, 1, embedding_dim]
        outputs, hidden_n = self.gru(embedded, hidden_0)
        # outputs: [batch_size, 1, hidden_dim], hidden_n: [1, batch_size, hidden_dim]
        outputs = self.linear(outputs)
        # outputs: [batch_size, 1, vocab_size]
        return outputs, hidden_n
