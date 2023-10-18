import torch
from torch.utils.data import DataLoader

class CharRnnDataset(torch.utils.data.Dataset):

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


def get_char_rnn_data_loaders(char_index_data, data_size, sequence_length):

    if sequence_length == -1 or sequence_length >= data_size:
        sequence_length = data_size - 1

    char_dataset = CharRnnDataset(char_index_data, sequence_length, data_size)
    char_dataloader = DataLoader(char_dataset, batch_size=1, shuffle=False)

    return char_dataloader, char_dataset.num_sequences