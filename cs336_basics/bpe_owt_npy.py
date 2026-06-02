import os
from cs336_basics.bpe_tokenization import train_bpe
from cs336_basics.bpe_tokenization import bpe_tokenizer
import time
import pickle


if __name__ == "__main__":
    PATH = "./data/my_bpe_owt.pkl"
    t0  = time.time()
    tokenizer = bpe_tokenizer.from_file(PATH,["<|endoftext|>"])
    tokenizer.save_to_npy_file("./data/owt_valid.txt", "./data/my_bpe_owt_valid.npy")
    print("tokenizer",tokenizer)
    print("time",time.time() - t0)
