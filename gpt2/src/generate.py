from model import GPT2
import torch
import torch.nn.functional as F
import time

device = 'cpu'

print(f"Using Device: {device}")

model = GPT2.from_pretrained('gpt2')
print("Loaded Model weights 🎉")
model.eval()
model.to(device)


import tiktoken
tokenizer = tiktoken.encoding_for_model('gpt2')
max_tokens = 128
num_samples = 2
temperature = 0.9
eps = 1e-6

prompt = "I am Donald Trump, I will "

tokens = tokenizer.encode(prompt)
tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(dim=0).repeat([num_samples, 1]) # [B, T]

torch.manual_seed(42)
start_time = time.perf_counter()
with torch.no_grad(): 
    for i in range(max_tokens): 
        logits = model.forward(tokens) # [B, T, vocab_size]
        curr_logit = logits[:, -1, :]
        scaled_logits = curr_logit / (temperature + eps)
        
        probs = F.softmax(scaled_logits, dim=-1) # [B, 1, vocab_size]
        topk_probs, topk_idx = torch.topk(probs, k=50, dim=-1) # (B, 50)
        sampled_idx = torch.multinomial(topk_probs, num_samples=1) # (B, 1)
        curr_tokens = torch.gather(topk_idx, dim = -1, index=sampled_idx) # (B, 1)
        
        tokens = torch.cat([tokens, curr_tokens], dim = -1)
        gen = tokenizer.decode_batch(curr_tokens.tolist())
        

print(f"{max_tokens} tokens per {num_samples} sequences took {time.perf_counter() - start_time} secs on {device}")

tokens = tokens.cpu().tolist()

tokens = tokenizer.decode_batch(tokens)
for i in tokens: 
    print('>', i)
