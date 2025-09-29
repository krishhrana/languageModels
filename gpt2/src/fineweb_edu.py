from datasets import load_dataset
import tiktoken
import numpy as np
import os
import multiprocessing as mp
import tqdm

tokenizer = tiktoken.encoding_for_model('gpt2')
fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train")

max_tokens_per_shard = int(1e8) # 100M
progress_bar = None



def tokenize(x): 
    tokens = [tokenizer.eot_token]
    tokens.extend(tokenizer.encode(x['text']))
    tokens = np.array(tokens, dtype=np.uint16)
    return tokens


def save_shard(shard_idx, tokens): 
    split = 'val' if shard_idx == 0 else 'train'
    print(f"Saving {len(tokens)} tokens to shard {shard_idx}")
    np.save(f'fineweb/shard_{split}_{shard_idx}.npy', tokens)

num_workers = int(os.cpu_count() // 2)

with mp.Pool(num_workers) as pool: 
    token_count = 0
    buffer = np.empty(max_tokens_per_shard, dtype=np.uint16)
    curr_shard_idx = 0


    for tokens in pool.imap(tokenize, fw, chunksize=16):
        if (token_count + len(tokens)) < max_tokens_per_shard: 
            buffer[token_count: token_count + len(tokens)] = tokens
            token_count = token_count + len(tokens)
            if progress_bar is None: 
                progress_bar = tqdm(total=max_tokens_per_shard, unit="tokens", desc=f"Shard {curr_shard_idx}")
            progress_bar.update(len(tokens))

        else: 
            rem = max_tokens_per_shard - token_count
            buffer[token_count: token_count + rem] = tokens[:rem]
            progress_bar.update(rem)
            save_shard(curr_shard_idx, tokens)
            
            curr_shard_idx = curr_shard_idx + 1
            buffer = np.empty(max_tokens_per_shard)
            buffer[0:len(tokens) - rem] = tokens[rem:]
            token_count = len(tokens) - rem
            progress_bar = None
            
        

    

