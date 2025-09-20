from model import GPT2
import torch
import torch.nn.functional as F
import time

device = 'cpu'

print(f"Using Device: {device}")


# model = GPT2.from_pretrained('gpt2')

model = GPT2()
state_dict = torch.load('model.checkpoint', map_location=device, weights_only=True)
new_state_dict = dict()

# for key, value in state_dict.items(): 
#     # print(key.split('.'))
#     key = ".".join(key.split('.')[2:])
#     # print(key)
#     new_state_dict[key] = value

model.load_state_dict(state_dict)

print("Loaded Model weights 🎉")
model.eval()
model.to(device)


import tiktoken
tokenizer = tiktoken.encoding_for_model('gpt2')
max_tokens = 128
num_samples = 5
temperature = 0.8
eps = 1e-8

prompt = """JULIET:
And stint thou too, I pray thee, nurse, say I."""


tokens = tokenizer.encode(prompt)
tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(dim=0).repeat([num_samples, 1]) # [B, T]

torch.manual_seed(42)
start_time = time.perf_counter()
log_prob_sums = torch.zeros(num_samples, dtype=torch.float, device=device)
generated_counts = torch.zeros(num_samples, dtype=torch.long, device=device)
with torch.no_grad(): 
    for i in range(max_tokens): 
        logits = model.forward(tokens) # [B, T, vocab_size]
        curr_logit = logits[:, -1, :]
        scaled_logits = curr_logit / (temperature + eps)

        log_probs = F.log_softmax(scaled_logits, dim=-1)
        probs = log_probs.exp()
        
        topk_probs, topk_idx = torch.topk(probs, k=50, dim=-1) # (B, 50)
        sampled_idx = torch.multinomial(topk_probs, num_samples=1) # (B, 1)
        curr_tokens = torch.gather(topk_idx, dim = -1, index=sampled_idx) # (B, 1)
        
        curr_log_probs = torch.gather(log_probs, dim=-1, index=curr_tokens)
        log_prob_sums += curr_log_probs.squeeze(-1) # sum of log probs of generated tokens
        generated_counts += 1

        tokens = torch.cat([tokens, curr_tokens], dim = -1)
                

print(f"{max_tokens} tokens per {num_samples} sequences took {time.perf_counter() - start_time} secs on {device}")

valid_counts = generated_counts.clamp_min(1).to(dtype=torch.float)
avg_neg_log_likelihood = -log_prob_sums / valid_counts
perplexities = avg_neg_log_likelihood.exp().cpu().tolist()

for idx, ppl in enumerate(perplexities, start=1):
    print(f"Sample {idx} perplexity: {ppl:.4f}")

tokens = tokens.cpu().tolist()

tokens = tokenizer.decode_batch(tokens)
for i in tokens: 
    print('>', i)

# Sample 1 perplexity: 2.4489
# Sample 2 perplexity: 2.6481
# Sample 3 perplexity: 2.5308
# Sample 4 perplexity: 3.0726
# Sample 5 perplexity: 3.5288
