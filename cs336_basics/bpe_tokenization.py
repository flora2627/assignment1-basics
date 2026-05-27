
from heapq import merge
from hmac import new
import os
from pickletools import read_unicodestring1
import regex as re



def is_best_pair_matched (
    word : tuple[bytes],
    best_pair : tuple[bytes]
) -> bool :
    for i in range(len(word)-1):
        pair = (word[i],word[i+1])
        if pair == best_pair :
            return True        
    return False

def make_a_new_word (
    word : tuple[bytes],
    best_pair : tuple[bytes]
) -> tuple[bytes] :
    
    new_word = []
    i = 0 
    
    while i<len(word):
        if i < len(word)-1 :
            pair = (word[i],word[i+1])
            if pair == best_pair :
                new_word.append (word[i]+word[i+1])
                i=i+2
            else :
                new_word.append(word[i])
                i=i+1
        else :
            new_word.append(word[i])
            i=i+1

    
    return tuple(new_word) 
               
        

def train_bpe (
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # print("train_bpe 111111", input_path, vocab_size, special_tokens, **kwargs)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    

    vocab = {}
    for i in range(256) :
        vocab[i] = bytes([i])
    vocab[256]=b"<|endoftext|>"

    #print(vocab)

    merges = []
    with open(input_path,"rb") as f :
        text = f.read()
        flag = re.escape(special_tokens[0])
        chunks = re.split(flag.encode("utf-8"),text)
        
    #print (len(chunks))
    freq_dict = {}
    for chunk in chunks :
        str_chunk = chunk.decode("utf-8")
        tokens_of_one_chunk = re.findall(PAT,str_chunk)
    
        for token in tokens_of_one_chunk:
            bytes_token = token.encode('utf-8')
            words = tuple(bytes([x]) for x in bytes_token)
            freq_dict[words] = freq_dict.get(words,0) + 1
            
    #print ("freq",freq)

    # 基于freq表，计算每个字符组合出现的次数

    while len(vocab) < vocab_size:
        pair_count = {}
        for words, count in freq_dict.items() :
            for i in range(len(words)-1):
                pair = (words[i],words[i+1])
                pair_count[pair] = pair_count.get(pair,0) + count

        new_token = max(pair_count, key=lambda p : (pair_count[p],p))
        
        vocab[len(vocab)] = new_token[0]+new_token[1]
        merges.append(new_token)

        #重建freq_dict
        new_freq_dict = {}
        for words, count in freq_dict.items():
            #print(words,new_token)
            if is_best_pair_matched(words,new_token) :
                new_word = make_a_new_word(words,new_token)
                #print(new_word)
                new_freq_dict [new_word] = count
            else :
                new_freq_dict[words] = count
        freq_dict = new_freq_dict


    #print(vocab,merges)
    
    return vocab,merges