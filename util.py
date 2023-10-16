import torch 




def load_data(path):
    
    txt_file = open(path, 'r', encoding='utf-8')
    char_data = txt_file.read()
    unique_chars = list(set(char_data))

    data_size, vocab_size = len(char_data), len(unique_chars)
    char_to_ix = { ch:i for i,ch in enumerate(unique_chars) }
    ix_to_char = { i:ch for i,ch in enumerate(unique_chars) }

    return char_data, data_size, vocab_size, char_to_ix, ix_to_char


def sample_data(model, seed, length, output_file, char_to_id, id_to_char):
    '''
    Samples a string of length 'length' from model and prints it to output file
    Seed 'seed' is used to prompt the first iteration of sampling 
    Char_to_id is mapping from char ('c') to index (5)
    id_to_char is mapping from char (5) to char ('c')
    '''

    input_char = seed
    input_char_index = char_to_id[input_char]
    previous_char_index = input_char_index

    for _ in range(length):

        next_char_probabilities = model(previous_char_index)
        next_char_index = torch.argmax(next_char_probabilities).item()
        next_char = id_to_char[next_char_index]
        output_file.write(next_char)

        previous_char_index = next_char_index

    

