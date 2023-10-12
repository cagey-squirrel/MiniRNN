
def load_data(path):
    
    txt_file = open(path, 'r', encoding='utf-8')
    char_data = txt_file.read()
    unique_chars = list(set(char_data))

    data_size, vocab_size = len(char_data), len(unique_chars)
    char_to_ix = { ch:i for i,ch in enumerate(unique_chars) }
    ix_to_char = { i:ch for i,ch in enumerate(unique_chars) }

    return char_data, data_size, vocab_size, char_to_ix, ix_to_char