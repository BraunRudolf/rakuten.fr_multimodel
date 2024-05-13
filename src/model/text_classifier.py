import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.rnn2 = nn.LSTM(hidden_dim, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64, 128)  # First additional fully connected layer
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        output = self.dropout(output)
        output, _ = self.rnn2(output)
        output = self.dropout2(output)
        output = output[:, -1, :]  # Get the last hidden state
        output = nn.ReLU()(self.fc1(output))  # Apply ReLU activation
        logits = self.fc3(output)
        return logits
