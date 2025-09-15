from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR, LinearLR, ConstantLR
from dataclasses import dataclass
import torch 

@dataclass
class LRScheduleConfig():
    max_lr: float = 6e-4
    mult_factor: float = 0.1
    warmup_steps: int = 40
    cosine_steps: int = 40
    max_steps: int = 100
    linear_start_factor: float = 0.05


def create_lr_schedule(optimizer: torch.optim.Optimizer):
    schedule_config = LRScheduleConfig()
    linear_scheduler = LinearLR(optimizer, start_factor=schedule_config.linear_start_factor, end_factor=1.0, total_iters=schedule_config.warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=schedule_config.cosine_steps, eta_min=schedule_config.max_lr * schedule_config.mult_factor)
    constant_lr = ConstantLR(optimizer, factor=schedule_config.mult_factor, total_iters=(schedule_config.max_steps - schedule_config.cosine_steps - schedule_config.warmup_steps))
    sequential_lr = SequentialLR(optimizer, [linear_scheduler, cosine_scheduler, constant_lr], milestones=[schedule_config.warmup_steps, schedule_config.warmup_steps + schedule_config.cosine_steps])
    return sequential_lr





