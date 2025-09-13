import torch
import torch.nn.functional as F
from model_config import GPT2Config
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'



class MLP(torch.nn.Module): 
    def __init__(self, config: GPT2Config): 
        super().__init__()
        
        self.c_fc = torch.nn.Linear(config.d_model, 4 * config.d_model)
        self.activation = torch.nn.GELU(approximate='tanh')
        self.c_proj = torch.nn.Linear(4 * config.d_model, config.d_model)
        self.c_proj.SCALE_FLAG = 1

    
    def forward(self, x): 
       x = self.c_fc(x)
       x = self.activation(x)
       x = self.c_proj(x)
       return x



class Attention(torch.nn.Module): 
    def __init__(self, config: GPT2Config): 
        super().__init__()

        self.c_attn = torch.nn.Linear(config.d_model, 3 * config.d_model)
        self.c_proj = torch.nn.Linear(config.d_model, config.d_model)
        self.c_proj.SCALE_FLAG = 1

        self.d_model = config.d_model
        self.n_heads = config.n_heads

        self.register_buffer("bias", tensor=torch.tril(torch.ones(size=(config.block_size, config.block_size))).view(1, 1, config.block_size, config.block_size))



    def forward(self, x): 
        # x ---> [B, T, C]
        B, T, C = x.size()

        # Q, K, V tensors
        qkv = self.c_attn.forward(x) # [B, T, 3 * C]
        q, k, v = qkv.split(split_size=self.d_model, dim=-1) # [B, T, C]

        # Multihead Attn ---> [B, nh, T, mini_dim] Split into n_heads
        # We want tokens to pass through heads, not heads to pass through tokens
        q = q.view(B, T, self.n_heads, C // self.n_heads).permute(0, 2, 1, 3)
        k = k.view(B, T, self.n_heads, C // self.n_heads).permute(0, 2, 3, 1) # [B, nh, mini_dim, T]
        v = v.view(B, T, self.n_heads, C // self.n_heads).permute(0, 2, 1, 3)

        # Calulate causal attn scores
        affinities = (q @ k) / math.sqrt(k.shape[2]) # [B, nh, T, T]
        affinities = affinities.masked_fill(mask=self.bias[:, :, :T, :T] == 0, value=float('-inf'))
        affinities = F.softmax(affinities, dim=-1)
        
        # Apply attn scores and concatenate mini attn heads (n_heads)
        y = affinities @ v # [B, nh, T, T] @ [B, nh, T, mini_dim] ---> [B, nh, T, mini_dim]
        y = y.permute(0, 2, 1, 3).reshape(B, T, C) # [B, T, C]

        # Pass though final projection layer
        y = self.c_proj.forward(y)
        return y



class Block(torch.nn.Module): 
    def __init__(self, config: GPT2Config):
        super().__init__()
        
        self.ln_1 = torch.nn.LayerNorm(config.d_model)
        self.attn = Attention(config=config)
        self.ln_2 = torch.nn.LayerNorm(config.d_model)
        self.mlp = MLP(config=config)

    
    def forward(self, x): 
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    



class GPT2(torch.nn.Module): 
    def __init__(self, config: GPT2Config = GPT2Config()): 
        super().__init__()

        self.config = config
        self.transformer = torch.nn.ModuleDict(dict(
            wte = torch.nn.Embedding(config.vocab_size, config.d_model), 
            wpe = torch.nn.Embedding(config.block_size, config.d_model), 
            h = torch.nn.ModuleList([Block(config) for _ in range(config.n_layers)]), 
            ln_f = torch.nn.LayerNorm(config.d_model), 
        ))
        self.lm_head = torch.nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # # This or the other way round, should yiled same result acc to me
        # self.lm_head.weight = self.transformer.wte.weight

        self.transformer.wte.weight = self.lm_head.weight


    
    def forward(self, x): 
        # x ---> [B, T]
        B, T = x.shape
        assert T <= self.config.block_size
        # pos + tok emb
        x = self.transformer.wte(x) + self.transformer.wpe(torch.arange(0, T, dtype=torch.long, device=x.device))

        # Pass thorugh transformer block
        for block in self.transformer.h: 
            x = block.forward(x)
        
        # Final layernorm
        x = self.transformer.ln_f.forward(x) # [B, T, d_model]
        logits = self.lm_head.forward(x) # [B, T, vocab_size]
        return logits
    

    def _init_weights(self, module: torch.nn.Module): 
        if isinstance(module, torch.nn.Linear): 
            # Initial init for linear layers
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # Sclaing for layers accumulating residual path ---> 2 residual streams per block
            if hasattr(module, 'SCALE_FLAG') and module.SCALE_FLAG == 1: 
                module.weight = module.weight / math.sqrt(2 * self.config.n_layers)
            if module.bias is not None: 
                torch.nn.init.constant_(module.bias, 0.0)
        if isinstance(module, torch.nn.Embedding): 
            torch.nn.init.normal_(module.weight, mean=0.0, std = 0.02)


    # Using Karpathy's code to load the weights from hf and chekcing hte implementation of the model
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel, AutoModelForCausalLM
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layers=12, n_heads=12, d_model=768),  # 124M params
            'gpt2-medium':  dict(n_layers=24, n_heads=16, d_model=1024), # 350M params
            'gpt2-large':   dict(n_layers=36, n_heads=20, d_model=1280), # 774M params
            'gpt2-xl':      dict(n_layers=48, n_heads=25, d_model=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
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
