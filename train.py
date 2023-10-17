from rnn_model import RNN, RNN_Parallel
from loss import CharacterPredictionLoss
import torch
from util import load_data, sample_data, get_char_data_loaders
from time import time
import sys
from IPython.core.ultratb import ColorTB

sys.excepthook = ColorTB()


def train(data_path, num_epochs=1000, lr=0.01, sequence_length=1024, hidden_size=100, output_file_name='output.txt'):

    device = torch.device('cpu' if not torch.cuda.is_available() else "cuda:0")
    char_index_data, data_size, vocab_size, char_to_id, id_to_char = load_data(data_path)
    char_dataloader, num_sequences = get_char_data_loaders(char_index_data, data_size, sequence_length)
    
    network = RNN(char_to_id, device, vocab_size=vocab_size, hidden_size=hidden_size)
    network.to(device)
    network.train()
    loss_function = CharacterPredictionLoss(char_to_id, device)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    output_file = open(output_file_name, 'w', encoding='utf-8')
    
    for epoch in range(num_epochs):
        
        optimizer.zero_grad()
        network.train()
        time_start = time()
        total_loss = 0
        loads = 0

        hidden_state = torch.zeros((num_sequences, hidden_size, 1)).to(device)

        for inputs, targets in char_dataloader:
            optimizer.zero_grad()
            loads += 1

            outputs, hidden_state = network(inputs, hidden_state)
            hidden_state = hidden_state.detach()
            
            loss = loss_function(outputs, targets)
            total_loss += loss
            
        total_loss /= loads
       
        print(f'finished epoch {epoch} with loss = {total_loss} in {time() - time_start} seconds')
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
        optimizer.step()

        if (epoch + 0) % 10 == 0:
            sample_data(network, 'a', 100, output_file, char_to_id, id_to_char, hidden_size, device)
            output_file.write('\n\n')
            output_file.flush()




def main():
    path = 'aca_chat.txt'
    #path = 'mini_example.txt'
    #path = 'test.txt'
    train(path)


if __name__ == "__main__":
    main()