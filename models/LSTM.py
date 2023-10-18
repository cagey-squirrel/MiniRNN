import torch


class LSTM(torch.nn.Module):


    def __init__(self, vocab_size, hidden_size, num_layers, dropout, batch_first):
        super().__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.dropout = dropout
        #batch_first = False
        # LSTM
        self.lstm = torch.nn.LSTM(input_size=vocab_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=batch_first)

        # Transforming outputs to vocab logits
        self.fc = torch.nn.Linear(hidden_size, vocab_size)

    
    def forward(self, input, hidden_state):
        
        lstm_output, hidden_state = self.lstm(input, hidden_state)
        word_prediction_logits = self.fc(lstm_output)

        return word_prediction_logits, hidden_state




