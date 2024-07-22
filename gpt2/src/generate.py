from model import GPT2
import torch
import torch.nn.functional as F

# from transformers import AutoModelForCausalLM
# model = AutoModelForCausalLM.from_pretrained(f'openai-community/gpt2')

model = GPT2.from_pretrained('gpt2')
print("Loaded Model weights 🎉")
model.eval()


import tiktoken as tk

num_seqs = 5
max_num_tokens = 30

tokenizer = tk.get_encoding('gpt2')
prompt = "Hello, I'm a language model,"
tokens = tokenizer.encode(prompt)
tokens = torch.tensor(tokens, dtype=torch.long)

x = tokens.unsqueeze(dim = 0).repeat([num_seqs, 1])

torch.manual_seed(42)
with torch.no_grad():
    for i in range(max_num_tokens):
        # x ---> (B, T)
        logits = model.forward(x) # (B, T, vocab_size)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim = -1) # (B, vocab_size)
        # This gets the top-k tokens with max probailities
        topk_probs, topk_idx = torch.topk(probs, k = 50, dim = -1) # (B, 50)
        # torch.multinomial sample randomly according to a probability distribution, output is the index of the values that it sampled
        sampled_prob_idx = torch.multinomial(topk_probs, num_samples=1) # (B, 1)
        # Index into the top_k tokens array to get the token ids
        token_idx = torch.gather(topk_idx, dim = -1, index=sampled_prob_idx)
        x = torch.cat([x, token_idx], dim = -1) # (B, T + 1)

x = x.tolist()
print(len(x), len(x[0]))
generations = tokenizer.decode_batch(x)
for i in generations:
    print(i)