import tiktoken
import torch
import os
import numpy as np

# Taken from Karpathy's NanoGPT repo
class DataLoaderLite:
    def __init__(self, B, T, RANK, WORLD_SIZE, split):
        self.B = B
        self.T = T
        self.RANK = RANK
        self.WORLD_SIZE = WORLD_SIZE
        self.split = split
        self.DATASET_PATH = '/home/ubuntu/fineweb/sample_10b'
        
        # init shard
        self.sp_idx = 0
        self.shard_paths = sorted([os.path.join(self.DATASET_PATH, p) for p in os.listdir(self.DATASET_PATH) if self.split in p])
        self.tokens = self._read_shard(shard_path=self.shard_paths[self.sp_idx])

        # state
        self.current_position = RANK * B * T


    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1] # +1 to get the labels as well
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # labels

        self.current_position += B * T * self.WORLD_SIZE
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.sp_idx = (self.sp_idx + 1) % len(self.shard_paths)
            self.tokens = self._read_shard(shard_path=self.shard_paths[self.sp_idx])
            self.current_position = self.RANK * B * T
        return x, y
    

    def _read_shard(self, shard_path): 
        tokens = np.load(shard_path)
        tokens = np.astype(tokens, np.int32)
        tokens = torch.tensor(tokens, dtype=torch.long)
        if self.RANK == 0: 
            print(f"Reading shard: {self.sp_idx} | Loaded {len(tokens)} tokens | Num Batches/GPU: {len(tokens) // (self.B*self.T*self.WORLD_SIZE)}")
        return tokens