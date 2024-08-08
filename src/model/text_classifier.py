from torch import nn


class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.rnn2 = nn.LSTM(hidden_dim, 64, batch_first=True)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64, 64)  # First additional fully connected layer
        self.fc3 = nn.Linear(64, num_classes)
        # TODO: Still needed?
        self.config = TextConfig(hidden_dim)

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


# TODO: Still needed?
class TextConfig:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size


class HeadlessTextClassifier(TextClassifier):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(HeadlessTextClassifier, self).__init__(
            vocab_size, embedding_dim, hidden_dim, num_classes
        )
        del self.fc3

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        output = self.dropout(output)
        output, _ = self.rnn2(output)
        output = self.dropout2(output)
        output = output[:, -1, :]  # Get the last hidden state

        return output
