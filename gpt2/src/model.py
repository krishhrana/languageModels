import torch
import torch.nn.functional as F
from dataclasses import dataclass
import math


class SelfAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.query = torch.nn.Linear(config.n_embed, config.n_embed)
        self.key = torch.nn.Linear(config.n_embed, config.n_embed)
        self.value = torch.nn.Linear(config.n_embed, config.n_embed)
        self.mask = torch.tril(torch.ones(size=(config.n_block, config.n_block))).view(1, config.n_block, config.n_block)


    def forward(self, x):
        B, T, C = x.shape()
        q = self.query(x) # (B, T, C)
        kT = self.key(x).transpose(-1, -2) # (B, C, T)
        v = self.value(x) # (B, T, C)

        scaled_dot_prod = (q @ kT) * 1/math.sqrt(C) # (B, T, T)
        scaled_dot_prod = scaled_dot_prod.masked_fill(self.mask[:, :T, :T] == 0, float('-inf')) # (B, T, T)
        affinities = F.softmax(scaled_dot_prod, dim = -1)

        weighted_hs = affinities @ v
        return weighted_hs 


class MultiHeadAttention():
    def __init__(self, config): 
        super().__init__()
        self.config = config

        # Just use one linear layer to project q, k, v together and split them later
        self.query = torch.nn.Linear(config.n_embed, config.n_embed)
        self.key = torch.nn.Linear(config.n_embed, config.n_embed)
        self.value = torch.nn.Linear(config.n_embed, config.n_embed)

        self.proj = torch.nn.Linear(config.n_embed, config.n_embed)
        self.mask = torch.tril(torch.ones(size=(config.n_block, config.n_block))).view(1, 1, config.n_block, config.n_block)

    def forward(self, x): 
        B, T, C = x.shape
        q = self.query(x) # (B, T, C)
        k = self.key(x) # (B, T, C)
        v = self.value(x) # (B, T, C)

        mini_head_dim = C // self.config.n_heads

        # (B, T, n_heads, mini_dim) ---> (B, n_heads, T, mini_dims) ---> we want to parallelize across mini_heads
        q_mini = q.view(B, T, self.config.n_heads, mini_head_dim).permute(0, 2, 1, 3)
        # (B, T, n_heads, mini_dims) ---> (B, n_heads, mini_dims, T)
        kT = k.view(B, T, self.config.n_heads, mini_head_dim).permute(0, 2, 3, 1)
        # (B, n_heads, T, mini_head_dim)
        v = v.view(B, T, self.config.n_heads, mini_head_dim).permute(0, 2, 1, 3)

        print(f"q_mini: {q_mini.shape}")
        print(f"k: {kT.shape}")
        print(f"v: {v.shape}")

        print(mini_head_dim)
        print(kT.size(-1))

        scaled_dot_prod = (q_mini @ kT) * 1.0/math.sqrt(mini_head_dim) # (B, n_heads, T, T)
        scaled_dot_prod = scaled_dot_prod.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))

        affinities = F.softmax(scaled_dot_prod, dim = -1)
        print(affinities[0, 0, :, :])
        affinities = affinities @ v # (B, n_heads, T, mini_head_dims)

        affinities = affinities.permute(0, 2, 1, 3).contiguous().view(B, T, C)
        print(f"affinities: {affinities.shape}")
        affinities = self.proj(affinities)
        return affinities
    



    

class Attention(torch.nn.Module): 
    def __init__(self, config):
        super().__init__()
        self.c_attn = torch.nn.Linear(config.n_embed, 3 * config.n_embed)
        self.c_proj = torch.nn.Linear(config.n_embed, config.n_embed)
        self.n_embed = config.n_embed
        self.n_heads = config.n_heads
        self.register_buffer('bias', torch.tril(torch.ones(size=(config.n_block, config.n_block))).view(1, 1, config.n_block, config.n_block))

    def forward(self, x): 
        B, T, C = x.shape
        # (B, T, 3*C)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim = -1) # (B, T, C)

        mini_head_dim = C // self.n_heads

        # (B, n_heads, T, mini_dim)
        q = q.view(B, T, self.n_heads, mini_head_dim).permute(0, 2, 1, 3)
        k = k.view(B, T, self.n_heads, mini_head_dim).permute(0, 2, 1, 3)
        v = v.view(B, T, self.n_heads, mini_head_dim).permute(0, 2, 1, 3)

        # (B, n_heads, T, T)
        attn = (q @ k.transpose(-2, -1)) * (1.0/math.sqrt(mini_head_dim))
        attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim = -1)

        # (B, n_heads, T, mini_dims)
        weighted_sum = attn @ v

        # (B, n_heads, T, mini_dims) ---> (B, T, n_heads, mini_dims) ---> (B, T, n_heads * mini_dims)
        weighted_sum = weighted_sum.permute(0, 2, 1, 3).contiguous().view(B, T, self.n_heads * mini_head_dim)
        weighted_sum = self.c_proj(weighted_sum)
        return weighted_sum
        


class MLP(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = torch.nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = torch.nn.GELU(approximate='tanh')
        self.c_proj = torch.nn.Linear(4 * config.n_embed, config.n_embed)

    def forward(self, x): 
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x



class DecoderBlock(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(config.n_embed)
        self.attn = Attention(config=config)
        self.ln_2 = torch.nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # Pre-LayerNorm + residual connection (attn)
        x = x + self.mlp(self.ln_2(x)) # Pre-LayerNorm + residual connection (FFN)
        return x


@dataclass
class GPT2Config:
    n_block: int = 1024 # seq_len
    n_vocab: int = 50257
    n_embed: int = 768
    n_heads: int = 12
    n_layers: int = 12


class GPT2(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Creating hf style dict ---> matching hf var names for GPT2
        self.transformer = torch.nn.ModuleDict(
            dict(
                wte = torch.nn.Embedding(config.n_vocab, config.n_embed), # (seq_len, d_model)
                wpe = torch.nn.Embedding(config.n_block, config.n_embed), # (seq_len, d_model)
                h = torch.nn.ModuleList([DecoderBlock(config) for i in range(config.n_layers)]), # number of decoder blocks
                ln_f = torch.nn.LayerNorm(config.n_embed) # normalizing across channels for each token (B, T, C) ---> (B, T, C_norm)
            )
        )
        self.lm_head = torch.nn.Linear(config.n_embed, config.n_vocab, bias = False) # input: (B, T, n_embed) output: (B, T, n_vocab)


    def forward(self, idx, target=None): 
        B, T = idx.shape
        # (B, T, C)
        embs = self.transformer.wte(idx)
        pos = torch.arange(T, dtype=torch.long, device=idx.device)
        pos = self.transformer.wpe(pos) # (T, C)
        x = embs + pos # (B, T, C), pos will get auto broadcasted with batch dim using pytorch broadcasting rules

        for decoder_block in self.transformer.h:
            x = decoder_block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, n_vocab)
        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), target.view(-1)) # (B*T, vocab_size), (B*T)
        return logits, loss
    

    # Using Karpathy's code to load the weights from hf and chekcing hte implementation of the model
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel, AutoModelForCausalLM
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layers=12, n_heads=12, n_embed=768),  # 124M params
            'gpt2-medium':  dict(n_layers=24, n_heads=16, n_embed=1024), # 350M params
            'gpt2-large':   dict(n_layers=36, n_heads=20, n_embed=1280), # 774M params
            'gpt2-xl':      dict(n_layers=48, n_heads=25, n_embed=1600), # 1558M params
        }[model_type]
        config_args['n_vocab'] = 50257 # always 50257 for GPT model checkpoints
        config_args['n_block'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        print(config_args)
        config = GPT2Config(**config_args)
        model = GPT2(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = AutoModelForCausalLM.from_pretrained(f'openai-community/{model_type}')
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    



import tiktoken as tk


model = GPT2(GPT2Config())

with open('gpt2/src/input.txt') as f:
    text = f.read()

tokenizer = tk.get_encoding('gpt2')
tokens = tokenizer.encode(text[:2000])

B, T = 4, 32

subset = torch.tensor(tokens)[: B*T + 1]
x, y = subset[:-1].view(B, T), subset[1:].view(B, T)

logits, loss = model.forward(x, y)

print(logits.shape)
print(loss)

optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4)
for i in range(50):
    optimizer.zero_grad()
    logits, loss = model.forward(x, y)
    loss.backward()
    optimizer.step()
    print(f"Iteration {i + 1}: Loss {loss.item()}")


tokenizer = tk.get_encoding('gpt2')
prompt = "First Citizen:"
tokens = tokenizer.encode(prompt)
tokens = torch.tensor(tokens, dtype=torch.long)

x = tokens.unsqueeze(dim = 0) # (1, T)

torch.manual_seed(42)
with torch.no_grad():
    for i in range(30):
        # x ---> (B, T)
        logits = model.forward(x)[0] # (B, T, vocab_size)
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
