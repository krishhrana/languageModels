import tiktoken
import torch

# Taken from Karpathy's NanoGPT repo
class DataLoaderLite:
    def __init__(self, B, T, RANK, WORLD_SIZE):
        self.B = B
        self.T = T
        self.RANK = RANK
        self.WORLD_SIZE = WORLD_SIZE

        # at init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = RANK * B * T

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.WORLD_SIZE
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T * self.WORLD_SIZE + 1) > len(self.tokens):
            self.current_position = self.RANK * B * T
        return x, y