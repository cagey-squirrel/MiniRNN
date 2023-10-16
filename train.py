from rnn_model import RNN, RNN_Parallel
from loss import CharacterPredictionLoss
import torch
from util import load_data, sample_data
from time import time


def train(data_path, num_epochs=1000, lr=0.01, seq_len=25, hidden_size=100, output_file_name='output.txt'):

    device = torch.device('cpu' if not torch.cuda.is_available() else "cuda:0")
    data, data_size, vocab_size, char_to_id, id_to_char = load_data(data_path)
    network = RNN(char_to_id, device, vocab_size=vocab_size, hidden_size=hidden_size)
    network.to(device)
    network.train()
    loss_function = CharacterPredictionLoss(char_to_id)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    output_file = open(output_file_name, 'w')
    
    for epoch in range(num_epochs):
        
        time_start = time()
        loss = 0
        network.reset_hidden_state()
        p = 0


        while p+seq_len+1 <= data_size:
            
            inputs  = [char_to_id[char] for char in data[p:p+seq_len]]
            targets = [char_to_id[char] for char in data[p+1:p+seq_len+1]]
            
            

            for input_char, target_char in zip(inputs, targets):
                char_predictions = network(input_char)
                loss += loss_function(char_predictions, target_char)

            p += seq_len

        loss /= seq_len
        print(f'finished epoch {epoch} with loss = {loss} in {time() - time_start} seconds')
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
        optimizer.step()

        if (epoch + 0) % 10 == 0:
            sample_data(network, 'a', 100, output_file, char_to_id, id_to_char)
            output_file.write('\n\n')
            output_file.flush()




def main():
    path = 'aca_chat_mini.txt'
    path = 'mini_example.txt'
    train(path)


if __name__ == "__main__":
    main()