
import os
import regex as re
from .pretokenization_example import find_chunk_boundaries
from multiprocessing import Pool

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


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


def worker(task) :
    #print("task",task)
    input_path = task["input_path"]
    start = task["start"]
    end = task["end"]
    special_tokens = task["special_tokens"]


    freq_dict = {}
    with open(input_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start)
        flag = re.escape(special_tokens[0])
        chunks = re.split(flag.encode("utf-8"),text)

        for chunk in chunks :
            str_chunk = chunk.decode("utf-8")

            tokens_of_one_chunk = re.findall(PAT,str_chunk)
            
            for token in tokens_of_one_chunk:
                bytes_token = token.encode('utf-8')
                words = tuple(bytes([x]) for x in bytes_token)
                freq_dict[words] = freq_dict.get(words,0) + 1
    return freq_dict

def train_bpe (
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # print("train_bpe 111111", input_path, vocab_size, special_tokens, **kwargs)
    
    

    vocab = {}
    for i in range(256) :
        vocab[i] = bytes([i])
    vocab[256]=b"<|endoftext|>"

    #print(vocab)
    freq_dict_list = []
    freq_dict = {}
    merges = []
    # with open(input_path,"rb") as f :
    #     text = f.read()
    #     flag = re.escape(special_tokens[0])
    #     chunks = re.split(flag.encode("utf-8"),text)
    
    with open(input_path, "rb") as f:
        num_processes = 8
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        # for start, end in zip(boundaries[:-1], boundaries[1:]):
        #     f.seek(start)
        #     text = f.read(end - start)
        #     flag = re.escape(special_tokens[0])
        #     chunks = re.split(flag.encode("utf-8"),text)

        #     for chunk in chunks :
        #         str_chunk = chunk.decode("utf-8")

        #         tokens_of_one_chunk = re.findall(PAT,str_chunk)
            
        #         for token in tokens_of_one_chunk:
        #             bytes_token = token.encode('utf-8')
        #             words = tuple(bytes([x]) for x in bytes_token)
        #             freq_dict[words] = freq_dict.get(words,0) + 1

        task_list=[]
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            task_list.append({
                "input_path" : input_path,
                "start" : start,
                "end" : end,
                "special_tokens": special_tokens
            })

    with Pool(num_processes) as p:
        for d in p.imap(worker,task_list):
            for k,v in d.items() :
                freq_dict[k]=freq_dict.get(k,0)+v
    
    
           
        
    #print ("freq",freq)

    # 基于freq表，计算每个字符组合出现的次数
    
    pair_count = {}
    pair_2_word = {}
    for words, count in freq_dict.items() :
        for i in range(len(words)-1):
            pair = (words[i],words[i+1]) 
            pair_count[pair] = pair_count.get(pair,0) + count
            pair_2_word.setdefault(pair, set()).add(words) 

    while len(vocab) < vocab_size:
        new_vocab = max(pair_count, key=lambda p : (pair_count[p],p))
        vocab[len(vocab)] = new_vocab[0]+new_vocab[1]
        merges.append(new_vocab)

        for old_word in list(pair_2_word[new_vocab]) :
            # 首先这里有了一个new_word
            new_word = make_a_new_word(old_word,new_vocab)
            old_count = freq_dict[old_word]

            # 这里计算老的pair，把所有的数据都删了
            for i in range(len(old_word)-1):
                old_pair = (old_word[i],old_word[i+1]) 
                pair_count[old_pair] = pair_count.get(old_pair,0) - old_count
                pair_2_word[old_pair].discard(old_word)

            # 这里要计算新的pair有哪些
            for i in range(len(new_word)-1):
                new_pair = (new_word[i],new_word[i+1]) 
                # 这里只要新的pair，所以pair_count里面有值的不加
                pair_count[new_pair] = pair_count.get(new_pair,0) + old_count
                pair_2_word.setdefault(new_pair, set()).add(new_word) 

            freq_dict[new_word] = freq_dict[old_word]
            del freq_dict[old_word]
        






        #重建freq_dict
        # new_freq_dict = {}
        # for words, count in freq_dict.items():
        #     #print(words,new_token)
        #     if words in pair_2_word[new_token] :
        #         new_word = make_a_new_word(words,new_token)
        #         #print(new_word)
        #         new_freq_dict [new_word] = count
        #     else :
        #         new_freq_dict[words] = count
        # freq_dict = new_freq_dict


    #print(vocab,merges)
    
    return vocab,merges