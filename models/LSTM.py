import torch
import numpy as np

class LSTM(torch.nn.Module):


    def __init__(self, char_to_id, id_to_char, vocab_size, hidden_size, num_layers, num_sequences, dropout, device, batch_first):
        super().__init__()

        self.char_to_id = char_to_id
        self.id_to_char = id_to_char
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.num_sequences = num_sequences
        self.device = device

        # LSTM
        self.lstm = torch.nn.LSTM(input_size=vocab_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=batch_first)

        # Transforming outputs to vocab logits
        self.fc = torch.nn.Linear(hidden_size, vocab_size)

    
    def forward(self, input, hidden_state):
        
        lstm_output, hidden_state = self.lstm(input, hidden_state)
        word_prediction_logits = self.fc(lstm_output)

        return word_prediction_logits, hidden_state
    

    def sample_data(self, sample_length, temperature, output_file):

        self.eval()

        seed_char = 'a'  # TODO get_random_init_char
        seed_char_index = self.char_to_id[seed_char]
        previous_char_index = seed_char_index

        hidden_state = self.init_hidden_state_for_sampling()

        num_chars = len(self.char_to_id)
        for _ in range(sample_length):


            input_char_tensor = torch.zeros((1, 1, self.vocab_size), device=self.device)
            input_char_tensor[0, 0, previous_char_index] = 1
            next_char_logits, hidden_state = self.forward(input_char_tensor, hidden_state)


            if temperature == -1:  # No random sampling, just choose highest probability char
                next_char_probabilities = torch.softmax(next_char_logits, dim=2)
                next_char_index = torch.argmax(next_char_probabilities).item()
                
            else:  # Random sampling with temperature: scale probabilities then randomly sample a char based on those probabilities
                next_char_logits /= temperature
                next_char_probabilities = torch.softmax(next_char_logits, dim=2)
                np_probs = next_char_probabilities.detach().cpu().numpy()
                next_char_index = np.random.choice(num_chars, p=np_probs.flatten())
            
            next_char = self.id_to_char[next_char_index]

            output_file.write(next_char)
            previous_char_index = next_char_index

        self.train()


    def init_hidden_state(self):

        hidden_state = (torch.zeros(self.num_layers, self.num_sequences, self.hidden_size).to(self.device),
                        torch.zeros(self.num_layers, self.num_sequences, self.hidden_size).to(self.device))
        return hidden_state
    

    def init_hidden_state_for_sampling(self):
        '''
        In training the input is in batches, when sampling we just use a single input (num_sequences = 1)
        '''
        hidden_state = (torch.zeros(self.num_layers, 1, self.hidden_size).to(self.device),
                        torch.zeros(self.num_layers, 1, self.hidden_size).to(self.device))
        return hidden_state


    def detach_hidden_state(self, hidden_state):

        # Hidden state is a tuple of two torch.Tensors
        hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())
        return hidden_state

