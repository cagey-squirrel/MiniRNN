import torch 
from torch.utils.data import DataLoader
import numpy as np

class CharDataset(torch.utils.data.Dataset):

    def __init__(self, char_index_data, sequence_length, data_size):
        self.char_index_data = char_index_data
        self.sequence_length = sequence_length
        self.data_size = data_size
        self.num_sequences = (len(self.char_index_data) - 1) // self.sequence_length
        print(f'self.data_size = {self.data_size}')
        print(f'self.num_sequences = {self.num_sequences}')
        print(f'self.sequence_length = {self.sequence_length}')

    def __len__(self):
        return self.sequence_length
    
    
    def __getitem__(self, index):
        
        inputs  = [char_index for char_index in self.char_index_data[index :  : self.sequence_length]][:self.num_sequences]
        targets = [char_index for char_index in self.char_index_data[(index + 1) :  : self.sequence_length]][:self.num_sequences]

        return torch.Tensor(inputs).long(), torch.Tensor(targets).long()


def get_char_data_loaders(char_index_data, data_size, sequence_length):

    if sequence_length == -1 or sequence_length >= data_size:
        sequence_length = data_size - 1

    char_dataset = CharDataset(char_index_data, sequence_length, data_size)
    char_dataloader = DataLoader(char_dataset, batch_size=1, shuffle=False)

    return char_dataloader, char_dataset.num_sequences


def load_data(path):
    
    txt_file = open(path, 'r', encoding='utf-8')
    char_data = txt_file.read()
    unique_chars = list(set(char_data))

    data_size, vocab_size = len(char_data), len(unique_chars)
    char_to_ix = { ch:i for i,ch in enumerate(unique_chars) }
    ix_to_char = { i:ch for i,ch in enumerate(unique_chars) }

    char_index_data = [char_to_ix[char] for char in char_data]

    return char_index_data, data_size, vocab_size, char_to_ix, ix_to_char


def sample_data(network, seed, length, output_file, char_to_id, id_to_char, hidden_size, device, temperature):
    '''
    Samples a string of length 'length' from model and prints it to output file
    Seed 'seed' is used to prompt the first iteration of sampling 
    Char_to_id is mapping from char ('c') to index (5)
    id_to_char is mapping from char (5) to char ('c')
    '''
    network.eval()

    input_char = seed
    input_char_index = char_to_id[input_char]
    previous_char_index = input_char_index
    hidden_state = torch.zeros((1, hidden_size, 1)).to(device)

    num_chars = len(char_to_id)
    for _ in range(length):

        next_char_logits, hidden_state = network(torch.Tensor([previous_char_index]).long(), hidden_state)

        if temperature == -1:  # No random sampling
            next_char_probabilities = torch.softmax(next_char_logits, dim=1)
            next_char_index = torch.argmax(next_char_probabilities).item()
        else:  # Random sampling with temperature
            next_char_logits /= temperature
            next_char_probabilities = torch.softmax(next_char_logits, dim=1)
            np_probs = next_char_probabilities.detach().cpu().numpy()
            next_char_index = np.random.choice(num_chars, p=np_probs.flatten())
        next_char = id_to_char[next_char_index]

        output_file.write(next_char)
        previous_char_index = next_char_index

    network.train()

