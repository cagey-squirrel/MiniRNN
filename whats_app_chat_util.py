import emoji
from collections import defaultdict


def extract_text_data(path):
    
    file = open(path, 'r', encoding='utf8')
    out_file = open('output.txt', 'w', encoding='utf8')

    lines = [line for line in file]
    #lines = lines[:100]

    #dd = defaultdict(lambda: 0)

    for line in lines:
        line = line[:-1]  # ommiting the new line character at the end of the sentance

        # text is in format of 
        # [30.10.22., 12:45:32] Sender Name: Some text message: it could contain two dots
        # SO there 3 ':' signs before text msg, but text msg itself could contain :
        txt_split_on_two_dots = line.split(':')
        text_msg = ':'.join(txt_split_on_two_dots[3:])

        text_msg = text_msg[1:]  # Omitting the starting space after : 
        text_msg = emoji.replace_emoji(text_msg, '')  # Removing emojis
        
        if 'omitted' in text_msg:
            continue

        if len(text_msg) == 0:
            continue

        if '‎' in text_msg:
            continue
        
        out_file.write(text_msg + '\n')
        print(text_msg)
    
    #print(dict(dd))




def main():
    path = 'aca_chat/chat.txt'
    extract_text_data(path)



if __name__ == "__main__":

    main()