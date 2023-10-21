from models.RNN import RNN, RNN_Parallel
from models.LSTM import LSTM
from loss import CharacterPredictionLoss
import torch
from util.util import load_data
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
    sample_length = args.sample_length
    batch_size = args.batch_size
    dropout = args.dropout

    device = torch.device('cpu' if not torch.cuda.is_available() else "cuda:0")
    char_index_data, data_size, vocab_size, char_to_id, id_to_char = load_data(data_path)
    
    
    if architecture == 'RNN':
        
        char_dataloader, num_sequences = get_char_rnn_data_loaders(char_index_data, data_size, sequence_length)
        network = RNN(char_to_id, id_to_char, vocab_size=vocab_size, hidden_size=hidden_size, num_sequences=num_sequences, device=device)
        args = num_sequences, hidden_size, device
    elif architecture == 'LSTM':
        char_dataloader, num_sequences, batch_size = get_char_lstm_data_loaders(char_index_data, data_size, sequence_length, vocab_size, batch_size, device)
        network = LSTM(char_to_id, id_to_char, vocab_size=vocab_size, hidden_size=hidden_size, num_layers=num_layers, num_sequences=num_sequences, sequence_length=sequence_length, batch_size=batch_size, dropout=dropout, device=device, batch_first=True)
        args = num_layers, num_sequences, hidden_size, device


    network.to(device)
    network.train()
    loss_function = CharacterPredictionLoss(char_to_id, device, architecture)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    output_file = open(output_file_name, 'w', encoding='utf-8')
    #output_file.writelines(vars(args))  # TODO: write all arguments in txt file
    
    for epoch in range(num_epochs):
        
        optimizer.zero_grad()
        network.train()
        time_start = time()
        total_loss = 0
        loads = 0

        hidden_state = network.init_hidden_state()

        for inputs, targets in char_dataloader:
    
            optimizer.zero_grad()
            loads += 1

            inputs, targets = inputs.to(device), targets.to(device)
            outputs, hidden_state = network(inputs, hidden_state, return_probs=True)
            hidden_state = network.detach_hidden_state(hidden_state)  # detaching hidden state so it can be used in next iteration with no gradient

            targets = network.one_hot_encoding(targets)
            loss = loss_function(outputs, targets)
            total_loss += loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
            optimizer.step()
            
        total_loss /= loads
       
        print(f'Finished epoch {epoch} with loss = {total_loss} in {time() - time_start} seconds')
        
        

        if (epoch + 0) % sample_freq == 0:
            network.sample_data(sample_length, temperature, output_file)
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
    parser.add_argument("--dropout", type=float, help="Dropout prob for LSTM model.", default=0.2)
    parser.add_argument("--sample_length", type=int, help="Length of generated sample", default=100)
    parser.add_argument("--batch_size", type=int, help="Batch size to use for LSTM model", default=8)

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