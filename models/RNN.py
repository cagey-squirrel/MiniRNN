import torch
import numpy as np

class RNN_Parallel(torch.nn.DataParallel):

    def __init__(self, module,  dim=0, device_ids=None, output_device=None):
        super().__init__(module, device_ids, output_device, dim)
        self.module = module
    
    def reset_hidden_state(self):
        self.module.reset_hidden_state()



class RNN(torch.nn.Module):

    def __init__(self, char_to_id, id_to_char, device, hidden_size, vocab_size, num_sequences):
        super().__init__()
        self.char_to_id = char_to_id
        self.id_to_char = id_to_char
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_sequences = num_sequences
        self.device = device

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
    

    def sample_data(self, sample_length, temperature, output_file):

        self.eval()

        seed_char = 'a'  # TODO get_random_init_char
        seed_char_index = self.char_to_id[seed_char]
        previous_char_index = seed_char_index
        

        hidden_state = self.init_hidden_state()
        num_chars = len(self.char_to_id)

        for _ in range(sample_length):

  
            next_char_logits, hidden_state = self.forward(torch.Tensor([previous_char_index]).long(), hidden_state)
            

            if temperature == -1:  # No random sampling
                next_char_probabilities = torch.softmax(next_char_logits, dim=1)
                next_char_index = torch.argmax(next_char_probabilities).item()
        
            else:  # Random sampling with temperature
                next_char_logits /= temperature
                next_char_probabilities = torch.softmax(next_char_logits, dim=1)
                np_probs = next_char_probabilities.detach().cpu().numpy()
                next_char_index = np.random.choice(num_chars, p=np_probs.flatten())

            next_char = self.id_to_char[next_char_index]

            output_file.write(next_char)
            previous_char_index = next_char_index

        self.train()
    

    def init_hidden_state(self):

        hidden_state = torch.zeros((self.num_sequences, self.hidden_size, 1)).to(self.device)
        return hidden_state



    def detach_hidden_state(self, hidden_state):
    
        # Hidden state is a single torch.tensor
        hidden_state = hidden_state.detach()
        return hidden_state
