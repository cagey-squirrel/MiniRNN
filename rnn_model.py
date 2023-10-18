import torch


class RNN_Parallel(torch.nn.DataParallel):

    def __init__(self, module,  dim=0, device_ids=None, output_device=None):
        super().__init__(module, device_ids, output_device, dim)
        self.module = module
    
    def reset_hidden_state(self):
        self.module.reset_hidden_state()



class RNN(torch.nn.Module):

    def __init__(self, char_to_id, device, hidden_size=1, vocab_size=1):
        super().__init__()
        self.char_to_id = char_to_id
        self.device = device
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        # Weights
        self.Wxh = torch.nn.Parameter(data=torch.randn(size=(hidden_size, vocab_size)).to(device) * 0.01, requires_grad=True)
        self.Whh = torch.nn.Parameter(data=torch.randn(size=(hidden_size, hidden_size)).to(device) * 0.01, requires_grad=True)
        self.Why = torch.nn.Parameter(data=torch.randn(size=(vocab_size, hidden_size)).to(device) * 0.01, requires_grad=True)

        # Biases
        self.bh = torch.nn.Parameter(data=torch.zeros((hidden_size, 1)).to(device), requires_grad=True)
        self.by = torch.nn.Parameter(data=torch.zeros((vocab_size, 1)).to(device), requires_grad=True)

        
    
    def forward(self, input_char_id, hidden_state):
        # Input vector has all 0s except index of input char which has a value of 1
        # Basically, its a one-hot encoding
        num_encodings = hidden_state.shape[0]
        input_vector = torch.zeros((num_encodings, self.vocab_size, 1)).to(self.device)

        
        input_indices = input_char_id.flatten()
        input_vector[torch.arange(num_encodings), input_indices, 0] = 1

        # Hidden state update
        first_dot = self.Wxh @ input_vector
        second_dot = self.Whh @ hidden_state
        hidden_state = torch.tanh(first_dot + second_dot + self.bh)

        # Logits of next char probabilities
        char_predictions = (self.Why @ hidden_state) + self.by

        # Normalized probabilities
        # Removed: now returns logits
        # char_predictions = torch.softmax(char_predictions, dim=1)

        return char_predictions, hidden_state
