import torch
import os
import tiktoken
from dataclasses import dataclass
from model import GPT2
from model_config import GPT2Config, TrainerConfig
import torch.nn.functional as F
import time
from lr_schedule import create_lr_schedule, LRScheduleConfig
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from src.dataloader import DataLoaderLite


is_ddp = (int(os.environ.get('RANK', -1))) != -1
if is_ddp: 
    dist.init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    master_process = ddp_rank == 0

    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)

else: 
    ddp_rank = 0
    world_size = 1
    local_rank = 0
    master_process = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)
if torch.cuda.is_available(): 
    torch.cuda.manual_seed(1337)

torch.set_float32_matmul_precision('high')


gpt_config = GPT2Config()
trainer_config = TrainerConfig()
lr_schedule_config = LRScheduleConfig()
dataloader = DataLoaderLite(B=trainer_config.gpu_batch_size, T=gpt_config.block_size)


model = GPT2().to(device)
model = torch.compile(model)
if is_ddp: 
    model = DDP(module=model, device_ids=[local_rank])


optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr_schedule_config.max_lr, weight_decay=0.1, betas=(0.9, 0.95), fused=True)
scheduler = create_lr_schedule(optimizer=optimizer)
grad_acc_steps = trainer_config.batch_size // (trainer_config.gpu_batch_size * gpt_config.block_size * world_size)


def process_batch():
    batch, y = dataloader.next_batch()
    B, T = batch.size()
    batch = batch.to(device=device)
    y = y.to(device)
    with torch.autocast(device_type=device, dtype=torch.bfloat16): 
        logits = model.forward(batch) # [B, T, vocab_size]
        logits = logits.view(B * T, -1)
        y = y.view(B*T)
        loss = F.cross_entropy(input=logits, target=y)
    return loss

# 1 step is 2**19 tokens ---> B*T per grad_acc_step per GPU ---> B*T*G*world_size ---> 16*1024*4*8
# B*T = 1 mini_batch per GPU
for step in range(lr_schedule_config.max_steps):
    start_time = time.perf_counter()
    optimizer.zero_grad()
    loss_acc = 0.0

    # No gradient sync while accumulating gradients
    with DDP.no_sync():
        for mini_step in range(grad_acc_steps - 1): 
            loss = process_batch()
            loss = loss / grad_acc_steps
            loss_acc = loss_acc + loss.detach() 
            # Accumulates the gradients in .grad param of each tensor (internally does a +=)
            loss.backward()

    # Exit context manager, perfom computation for last mini_batch, this time sync gradients
    loss = process_batch()
    loss = loss / grad_acc_steps
    loss_acc = loss_acc + loss.detach() 
    loss.backward()

    # get the average loss for the grad_accum steps across all gpus
    dist.all_reduce(tensor=loss_acc, op=dist.ReduceOp.AVG)
    
    # This should be same across each GPU as we do grad sync before and model params and grads are identical across GPUs
    norm = torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
    curr_lr = scheduler.get_last_lr()
    
    # Each GPU does its own optimizer.step() and scheduler.step()
    optimizer.step()
    scheduler.step()

    torch.cuda.synchronize()
    end_time = time.perf_counter() - start_time
    if master_process:
        print(f"Step: {step} | loss: {loss_acc.item()} | norm: {norm} | lr: {curr_lr} | iter_time: {end_time} secs")

# print(f"{interations} iterations for {GPT2Config()} took {end_time / interations} secs/iter")
