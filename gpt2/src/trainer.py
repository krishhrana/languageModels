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
from dataloader import DataLoaderLite
import wandb


is_ddp = (int(os.environ.get('RANK', -1))) != -1
if is_ddp: 
    dist.init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    master_process = ddp_rank == 0

    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)

    if master_process: 
        print("Starting DDP Process")

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
train_dataloader = DataLoaderLite(B=trainer_config.gpu_batch_size, T=gpt_config.block_size, RANK=local_rank, WORLD_SIZE=world_size, split='train')
val_dataloader = DataLoaderLite(B=trainer_config.gpu_batch_size, T=gpt_config.block_size, RANK=local_rank, WORLD_SIZE=world_size, split='val')

if master_process: 
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="krishrana-intel",
        # Set the wandb project where this run will be logged.
        project="gpt2",
        # Track hyperparameters and run metadata.
        config={
            "architecture": "GPT2",
            "dataset": 'fineweb-edu-10b',
            "train_config": trainer_config.__dict__, 
            "gpt_config": gpt_config.__dict__, 
            "lr_schedule": lr_schedule_config.__dict__
            }
        )


model = GPT2().to(device)
model = torch.compile(model)
if is_ddp: 
    model = DDP(module=model, device_ids=[local_rank])


optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr_schedule_config.max_lr, weight_decay=0.1, betas=(0.9, 0.95), fused=True)
scheduler = create_lr_schedule(optimizer=optimizer)
grad_acc_steps = trainer_config.batch_size // (trainer_config.gpu_batch_size * gpt_config.block_size * world_size)


class Trainer(): 
    def __init__(self, model, grad_acc_steps, train_loader, val_loader): 
        self.model = model
        self.grad_acc_steps = grad_acc_steps
        self.tain_loader = train_loader
        self.val_loader = val_loader

    # one step
    def train(self, step): 
        start_time = time.perf_counter()
        optimizer.zero_grad()
        loss_acc = 0.0

        with self.model.no_sync():
            for mini_step in range(self.grad_acc_steps - 1):
                if master_process: 
                    print(f"Grad acc step: {mini_step}")
                loss = self._process_batch(self.tain_loader)
                loss = loss / grad_acc_steps
                loss_acc = loss_acc + loss.detach() 
                # Accumulates the gradients in .grad param of each tensor (internally does a +=)
                loss.backward()
        # Exit context manager, perfom computation for last mini_batch, this time sync gradients
        loss = self._process_batch(self.tain_loader)
        loss = loss / grad_acc_steps
        loss_acc = loss_acc + loss.detach() 
        loss.backward()

        # get the average loss for the grad_accum steps across all gpus
        dist.all_reduce(tensor=loss_acc, op=dist.ReduceOp.AVG)
        ppl = torch.exp(loss_acc)

        # This should be same across each GPU as we do grad sync before and model params and grads are identical across GPUs
        norm = torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=1.0)
        curr_lr = scheduler.get_last_lr()[0]

        # Each GPU does its own optimizer.step() and scheduler.step()
        optimizer.step()
        scheduler.step()

        torch.cuda.synchronize()
        end_time = time.perf_counter() - start_time
        if master_process:
            log = {"train_loss": loss_acc.item(), "train_ppl": ppl.item(), "train_norm": norm, "lr": curr_lr, "train_iter_time": end_time}
            run.log(log)
            print(f"Step: {step} | loss: {loss_acc.item():04f} | ppl: {ppl.item():04f} | norm: {norm:04f} | lr: {curr_lr:04f} | iter_time: {end_time:04f} secs")
    

    @torch.no_grad()
    def validate(self, step): 
        loss_acc = 0
        for vs in range(lr_schedule_config.validation_steps):
            start_time = time.perf_counter()
            loss = self._process_batch(val_dataloader)
            loss_acc = loss_acc + loss.detach()
        
        loss_acc = loss_acc / lr_schedule_config.validation_steps # Avg loss across all steps
        dist.all_reduce(tensor=loss_acc, op=dist.ReduceOp.AVG) # Avg loss across all GPUS
        ppl = torch.exp(loss_acc)
       
        torch.cuda.synchronize()
        end_time = time.perf_counter() - start_time
        if master_process:
            log = {"val_loss": loss_acc.item(), "val_ppl": ppl.item(), "val_iter_time": end_time}
            run.log(log, step=step)
            print(f"Step: {step} | loss: {loss_acc.item():04f} | ppl: {ppl.item():04f} | iter_time: {end_time:04f} secs")


    def _process_batch(self, dataloader):
        batch, y = dataloader.next_batch()
        B, T = batch.size()
        batch = batch.to(device=device)
        y = y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16): 
            logits = self.model.forward(batch) # [B, T, vocab_size]
            logits = logits.view(B * T, -1)
            y = y.view(B*T)
            loss = F.cross_entropy(input=logits, target=y)
        return loss
    

trainer = Trainer(model=model, grad_acc_steps=grad_acc_steps, train_loader=train_dataloader, val_loader=val_dataloader)

last_step = lr_schedule_config.max_steps - 1

# one step ---> 2e19 tokens (0.5M)
# run validation every 50M tokens
for step in range(lr_schedule_config.max_steps):
    # Train
    trainer.train(step)
    
    # Validation
    if step % 100 == 0: 
        model.eval()
        val_loss = trainer.validate(step)
        model.train()
        
    # Checkpointing
    if master_process and (step % 5000 == 0 or step == last_step):
        checkpoint = {
            'model': model.module.state_dict(),
            'config': model.module.config,
            'step': step,
        }
        torch.save(checkpoint, f'/home/ubuntu/fineweb/model_checkpoints/"model_{step}.pt"')
    
if is_ddp: 
    dist.destroy_process_group()