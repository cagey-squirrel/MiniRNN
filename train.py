from rnn_model import RNN
from loss import CharacterPredictionLoss
import torch
from util import load_data





def train(data_path, num_epochs=100, lr=0.1, seq_len=25, hidden_size=100):

    data, data_size, vocab_size, char_to_id, id_to_char = load_data(data_path)
    network = RNN(char_to_id, vocab_size=vocab_size, hidden_size=hidden_size)
    loss_function = CharacterPredictionLoss(char_to_id)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    network.train()

    for epoch in range(num_epochs):
        
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
        print(f'loss = {loss}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



def main():
    path = 'aca_chat.txt'
    train(path)


if __name__ == "__main__":
    main()