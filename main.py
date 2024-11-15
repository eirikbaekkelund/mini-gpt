import os
import torch
import wandb
from config import TrainConfig, GPTConfig
from contextlib import nullcontext

from utils import (
    init_model_from_scratch,
    get_batch, 
    cosine_lr_scheduler,
    prepare_optimizer,
    estimate_loss,
)
DTYPES =  {
    'float32': torch.float32, 
    'bfloat16': torch.bfloat16, 
    'float16': torch.float16
}

def fit(
    model_config: GPTConfig,
    train_config: TrainConfig,
    model_path: str = None,
    data_dir: str = None,
) -> None:
    device = torch.device(train_config.device)
    model = init_model_from_scratch(model_config)
    # model.compile() NOTE: uncomment if you have triton support
    model.to(device)
    optimizer = prepare_optimizer(model, train_config)
    context = nullcontext() if train_config.device == 'cpu' else torch.amp.autocast(device_type=train_config.device, dtype=DTYPES[train_config.dtype])
    best_val_loss = float('inf')

    for w_group in optimizer.param_groups:
        w_group['initial_lr'] = cosine_lr_scheduler(0, train_config)
   
    
    if train_config.wandb:
        wandb.init(project="gpt2", config=train_config.model_dump())
        losses = estimate_loss(model, model_config, train_config, ctx=context)
        wandb.log(data={
            "train/loss": losses['train'],
            "val/loss": losses['val'],
            "lr0": optimizer.param_groups[0]['initial_lr'],
            "lr1": optimizer.param_groups[0]['lr'],
            },
            step=0)

    for i in range(train_config.max_iters):
        # for _ in range(train_config.grad_accum):
        x, y = get_batch(
            block_size=model_config.block_size, 
            batch_size=train_config.batch_size, 
            device=device, 
            device_type=train_config.dtype,
            split='train',)
        optimizer.zero_grad()
        with context:
            _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        i += 1
        print(f"Step: {i}/{train_config.max_iters}, Loss: {loss.item():.3f}", end='\r')
        
        for w_group in optimizer.param_groups:
            w_group['lr'] = cosine_lr_scheduler(i, train_config)
        
        if i % train_config.eval_steps == 0 and train_config.wandb:
            losses = estimate_loss(model, model_config, train_config, ctx=context)
            wandb.log(data={
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr0": optimizer.param_groups[0]['initial_lr'],
                "lr1": optimizer.param_groups[0]['lr'],
            },
                step=i,)
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                torch.save(model.state_dict(), os.path.join(model_path, 'gpt2.pt'))
    if train_config.wandb:
        wandb.finish()

if __name__ == "__main__":
    model_config = GPTConfig()
    train_config = TrainConfig()
    fit(model_config, train_config, "models", "data")

