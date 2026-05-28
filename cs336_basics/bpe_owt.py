import os
from cs336_basics.bpe_tokenization import train_bpe
import time
import pickle


if __name__ == "__main__":
    PATH = "./data/owt_train.txt"
    PATH_VALID = "./data/owt_valid.txt"
    t0  = time.time()
    test = True
    if test :
        vocab, merges = train_bpe(
            PATH,
            32_000,
            ["<|endoftext|>"],
        )
    else :
        vocab, merges = train_bpe(
            PATH_VALID,
            32_00,
            ["<|endoftext|>"],
        )

    
    #print (vocab)
    tokens = [token for id, token in vocab.items() if id > 256]
    long_vocab = max(tokens, key=len)    
    t1 = time.time()

    with open("./data/my_bpe_owt.pkl", "wb") as f:
      pickle.dump((vocab, merges), f)

    print ("longest_vocab",long_vocab)
    print("time",t1 - t0)
