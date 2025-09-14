import torch
import os
import tiktoken
from dataclasses import dataclass
from model import GPT2
from model_config import GPT2Config
import torch.nn.functional as F
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_float32_matmul_precision('high')

print(f"Using Device: {device}")

@dataclass
class TrainerConfig(): 
    batch_size: int = 16


trainer_config = TrainerConfig
gpt_config = GPT2Config()

tokenizer = tiktoken.encoding_for_model('gpt2')

# Data loader
with open('src/input.txt', 'r') as f:
    text = f.read()

tokens = tokenizer.encode(text)
train_token_size = len(tokens)

if train_token_size % gpt_config.block_size != 0: 
    pad_size = gpt_config.block_size - (train_token_size % gpt_config.block_size)
    tokens = tokens + ([50256] * (pad_size + 1))

# (-1, T)
inputs = torch.tensor(tokens[:-1]).view(-1, gpt_config.block_size)
labels = torch.tensor(tokens[1:]).view(-1, gpt_config.block_size)

if len(inputs) % trainer_config.batch_size != 0: 
    pad_size = trainer_config.batch_size - (len(inputs) % trainer_config.batch_size)
    pad = torch.tensor([50256] * inputs.shape[1], dtype=torch.long).repeat([pad_size, 1]) # [pad_size, 1024]
    inputs = torch.cat([inputs, pad], dim = 0)
    labels = torch.cat([labels, pad], dim = 0)


inputs = inputs.view(-1, trainer_config.batch_size, gpt_config.block_size)
labels = labels.view(-1, trainer_config.batch_size, gpt_config.block_size)

assert inputs.shape == labels.shape
print(inputs.shape)

def dataset(idx):
    idx = idx % inputs.shape[0] 
    batch = inputs[idx, :, :] # [B, T]
    y = labels[idx, :, :]
    return batch, y


model = GPT2()
model = torch.compile(model)

model.to(device=device)
optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-4)
interations = 50


for idx in range(interations):
    start_time = time.perf_counter()

    batch, y = dataset(idx) 
    optimizer.zero_grad()
    B, T = batch.size()
    batch = batch.to(device=device)
    y = y.to(device)

    with torch.autocast(device_type=device, dtype=torch.bfloat16): 
        logits = model.forward(batch) # [B, T, vocab_size]
        logits = logits.view(B * T, -1)
        y = y.view(B*T)
        loss = F.cross_entropy(input=logits, target=y)

    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    end_time = time.perf_counter() - start_time

    print(f"Step: {idx} | loss: {loss.item()} | iter_time: {end_time} secs")

# print(f"{interations} iterations for {GPT2Config()} took {end_time / interations} secs/iter")
