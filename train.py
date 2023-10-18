from models.RNN import RNN, RNN_Parallel
from models.LSTM import LSTM
from loss import CharacterPredictionLoss
import torch
from util.util import load_data, sample_data, init_hidden_state
from util.data_loader_rnn import get_char_rnn_data_loaders
from util.data_loader_lstm import get_char_lstm_data_loaders
from time import time
import sys
import argparse
from IPython.core.ultratb import ColorTB

sys.excepthook = ColorTB()


def train(args):

    data_path = args.txt_file_path
    num_epochs = args.num_epochs
    lr = args.learning_rate
    sequence_length = args.sequence_length
    hidden_size = args.hidden_size
    output_file_name = args.output_file_name
    sample_freq = args.sample_freq
    temperature = args.temperature
    architecture = args.architecture
    num_layers = args.num_layers
    dropout = 0.2

    device = torch.device('cpu' if not torch.cuda.is_available() else "cuda:0")
    char_index_data, data_size, vocab_size, char_to_id, id_to_char = load_data(data_path)
    
    
    if architecture == 'RNN':
        network = RNN(char_to_id, device, vocab_size=vocab_size, hidden_size=hidden_size)
        char_dataloader, num_sequences = get_char_rnn_data_loaders(char_index_data, data_size, sequence_length)
        args = num_sequences, hidden_size, device
    elif architecture == 'LSTM':
        network = LSTM(vocab_size=vocab_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        char_dataloader, num_sequences = get_char_lstm_data_loaders(char_index_data, data_size, sequence_length, vocab_size, device)
        args = num_layers, num_sequences, hidden_size, device


    network.to(device)
    network.train()
    loss_function = CharacterPredictionLoss(char_to_id, device, architecture)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    output_file = open(output_file_name, 'w', encoding='utf-8')
    #output_file.writelines(vars(args))
    
    for epoch in range(num_epochs):
        
        optimizer.zero_grad()
        network.train()
        time_start = time()
        total_loss = 0
        loads = 0

        hidden_state = init_hidden_state(architecture, args)
        #hidden_state = torch.zeros((num_sequences, hidden_size, 1)).to(device)

        for inputs, targets in char_dataloader:
            optimizer.zero_grad()
            loads += 1

            #print(f'inputs.shape = {inputs.shape}')
            #exit(-1)
            outputs, hidden_state = network(inputs, hidden_state)

            if architecture == 'RNN':
                hidden_state = hidden_state.detach()
            elif architecture == 'LSTM':
                hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())
            
            loss = loss_function(outputs, targets)
            total_loss += loss
            
        total_loss /= loads
       
        print(f'Finished epoch {epoch} with loss = {total_loss} in {time() - time_start} seconds')
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
        optimizer.step()

        if (epoch + 0) % sample_freq == 0:
            sample_data(network, 'a', 100, output_file, char_to_id, id_to_char, hidden_size, device, temperature, architecture, args, vocab_size)
            output_file.write('\n\n')
            output_file.flush()




def main(args):
    train(args)


def arg_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument("--txt_file_path", type=str, help="Path to txt file for network to train on", default="wa_chat.txt")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs model trains for", default=1000)
    parser.add_argument("--learning_rate", type=float, help="Learning rate for model", default=0.01)
    parser.add_argument("--sequence_length", type=int, help="Txt file is split into sequences for training. This parameter determines the length of those sequences. Smaller length grants faster training with lower performance. -1 means whole file is used as a sequence."
                        , default=256)
    parser.add_argument("--hidden_size", type=int, help="Number of neurons in hidden layer.", default=100)
    parser.add_argument("--output_file_name", type=str, help="Path to txt file where output will be printed.", default="output.txt")
    parser.add_argument("--temperature", type=float, help="Temperature used for scaling logits before softmax. Used only for random sampling. Value of -1 means deterministic argmax sampling.", default=-1)
    parser.add_argument("--sample_freq", type=int, help="After how many epochs to sample output.", default=10)
    parser.add_argument("--architecture", type=str, help="Type of architecture to use.", choices=['LSTM', 'RNN'], default='RNN')
    parser.add_argument("--num_layers", type=int, help="Number of hidden layers for LSTM model.", default=2)

    args = parser.parse_args()

    return args


def run(**kwargs):
    # Usage: from train import run; run(txt_file_path='songs.txt', num_epochs=10)
    args = arg_parser()

    for k, v in kwargs.items():
        setattr(args, k, v)
    
    main(args)



if __name__ == "__main__":

    args = arg_parser()
    print(vars(args))
    main(args)