import torch 
import numpy as np


def init_hidden_state(architecture, args):

    if architecture == 'RNN':
        num_sequences, hidden_size, device = args
        hidden_state = torch.zeros((num_sequences, hidden_size, 1)).to(device)
    elif architecture == 'LSTM':
        num_layers, sequence_length, lstm_size, device = args
        hidden_state = (torch.zeros(num_layers, sequence_length, lstm_size).to(device),
                        torch.zeros(num_layers, sequence_length, lstm_size).to(device))
    
    return hidden_state


def load_data(path):
    
    txt_file = open(path, 'r', encoding='utf-8')
    char_data = txt_file.read()
    unique_chars = list(set(char_data))

    data_size, vocab_size = len(char_data), len(unique_chars)
    char_to_ix = { ch:i for i,ch in enumerate(unique_chars) }
    ix_to_char = { i:ch for i,ch in enumerate(unique_chars) }

    char_index_data = [char_to_ix[char] for char in char_data]

    return char_index_data, data_size, vocab_size, char_to_ix, ix_to_char


def sample_data(network, seed, length, output_file, char_to_id, id_to_char, hidden_size, device, temperature, architecture, args, vocab_size):
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
    
    # Changing sequence_length to 1 for sampling
    if architecture == 'LSTM':
        num_layers, sequence_length, lstm_size, device = args
        args = num_layers, 1, lstm_size, device


    hidden_state = init_hidden_state(architecture, args)

    num_chars = len(char_to_id)
    for _ in range(length):
        
        if architecture == 'RNN':
            next_char_logits, hidden_state = network(torch.Tensor([previous_char_index]).long(), hidden_state)
        elif architecture == 'LSTM':
            

            input_char_tensor = torch.zeros((1, 1, vocab_size), device=device)
            input_char_tensor[0, 0, previous_char_index] = 1
            next_char_logits, hidden_state = network(input_char_tensor, hidden_state)

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

