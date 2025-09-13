import torch
import os
import tiktoken
from dataclasses import dataclass
from model import GPT2
from model_config import GPT2Config
import torch.nn.functional as F

@dataclass
class TrainerConfig(): 
    batch_size: int = 4


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

# (B, T)
inputs = torch.tensor(tokens[:-1]).view(-1, gpt_config.block_size)
labels = torch.tensor(tokens[1:]).view(-1, gpt_config.block_size)

if len(inputs) % trainer_config.batch_size != 0: 
    pad_size = trainer_config.batch_size - (len(inputs) % trainer_config.batch_size)
    pad = torch.tensor([50256] * inputs.shape[1], dtype=torch.long).repeat([pad_size, 1]) # [pad_size, 1024]
    inputs = torch.cat([inputs, pad], dim = 0).view(-1, trainer_config.batch_size, gpt_config.block_size)
    labels = torch.cat([labels, pad], dim = 0).view(-1, trainer_config.batch_size, gpt_config.block_size)

assert inputs.shape == labels.shape

def dataset(): 
    for idx in range(inputs.shape[0]):
        batch = inputs[0, :, :] # [B, T]
        y = labels[0, :, :]
        yield batch, y


model = GPT2()
optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-4)

for idx, (batch, y) in enumerate(dataset()): 
    optimizer.zero_grad()
    B, T = batch.size()
    logits = model.forward(batch) # [B, T, vocab_size]

    logits = logits.view(B * T, -1)
    y = y.view(B*T)
    loss = F.cross_entropy(input=logits, target=y)

    loss.backward()
    optimizer.step()
    print(f"Step: {idx} | loss: {loss.item()}")
    if idx == 5: 
        break
