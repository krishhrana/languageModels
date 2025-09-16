from dataclasses import dataclass


@dataclass
class GPT2Config: 
    n_layers: int = 12
    d_model: int = 768
    vocab_size: int = 50257
    n_heads: int = 12
    block_size: int = 1024


@dataclass
class TrainerConfig(): 
    batch_size: int = 2**19
    gpu_batch_size: int = 16
