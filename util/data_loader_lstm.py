import torch
from torch.utils.data import DataLoader

class CharLstmDataset(torch.utils.data.Dataset):

    def __init__(self, char_index_data, sequence_length, data_size, vocab_size, device):
        self.char_index_data = char_index_data
        self.sequence_length = sequence_length
        self.data_size = data_size
        self.num_sequences = (len(self.char_index_data) - 1) // self.sequence_length
        self.vocab_size = vocab_size
        self.device = device
        print(f'self.data_size = {self.data_size}')
        print(f'self.num_sequences = {self.num_sequences}')
        print(f'self.sequence_length = {self.sequence_length}')
        print(f'self.vocab_size = {self.vocab_size}')


    def __len__(self):
        return self.num_sequences
    
    
    def __getitem__(self, index):
        
        start_index =  index      * self.sequence_length
        end_index   = (index + 1) * self.sequence_length
        inputs  = [get_one_hot_vector(self.vocab_size, char_index, self.device) for char_index in self.char_index_data[start_index     : end_index]]
        targets = [get_one_hot_vector(self.vocab_size, char_index, self.device) for char_index in self.char_index_data[start_index + 1 : end_index + 1]]

        return torch.stack(inputs), torch.stack(targets)


def get_one_hot_vector(shape, index, device):

    one_hot_vector = torch.zeros(shape).to(device)
    one_hot_vector[index] = 1

    return one_hot_vector


def get_char_lstm_data_loaders(char_index_data, data_size, sequence_length, vocab_size, device):

    if sequence_length == -1 or sequence_length >= data_size:
        sequence_length = data_size - 1

    char_dataset = CharLstmDataset(char_index_data, sequence_length, data_size, vocab_size, device)
    char_dataloader = DataLoader(char_dataset, batch_size=6, shuffle=False)

    return char_dataloader, char_dataset.num_sequences