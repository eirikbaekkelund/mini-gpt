import os
import math
import inspect
import tiktoken
import torch
import numpy as np

from typing import Tuple
from contextlib import nullcontext
from config import DDPConfig, GPTConfig, TrainConfig
from gpt2 import GPT

from torch.distributed import init_process_group


def tokenize(text: str, model: str = "gpt2"):
    encoder = tiktoken.get_encoding(model)
    ids = encoder.encode_ordinary(text)

    return ids

def init_model_from_scratch(config: GPTConfig) -> GPT:
    print("Initializing model..")
    model = GPT(config)
    print("Model initialized..")
    return model

def init_model_from_pretrained(outdir: str) -> GPT:
    path = os.path.join(outdir, 'ckpt.pt')
    checkpoint = torch.load(path)
    checkpoint_args = checkpoint['model_args']
    for k in GPTConfig().model_dump().keys():
        assert k in checkpoint_args, f"Model config does not have {k}"
        model = GPT(**checkpoint_args)
    
    state_dict = checkpoint['model']
    prefix = '_orig_mod.'
    for k in state_dict.keys():
        if k.startswith(prefix):
            state_dict[k[len(prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)

    return model

def cosine_lr_scheduler(iter: int, config: TrainConfig):
    if iter < config.warm_up_steps:
        return config.lr * iter / config.warm_up_steps
    
    elif iter > config.max_iters:
        return config.min_lr
    
    decay_ratio = (iter - config.warm_up_steps) / (config.max_iters - config.warm_up_steps)
    assert 0 <= decay_ratio <= 1, "Decay ratio should be between 0 and 1"
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))

    return config.min_lr + 0.5 * (config.lr - config.min_lr) * coeff


def prepare_optimizer(model: GPT, config: TrainConfig) -> torch.optim.Optimizer:
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for p in params.values() if p.ndim >= 2]
    no_decay_params = [p for p in params.values() if p.ndim < 2]

    optimizer_grouped_parameters = [
        {"params": no_decay_params, "weight_decay": 0.0},
        {"params": decay_params, "weight_decay": config.weight_decay},
    ]
    fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters and config.device == 'cuda'
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=config.lr,
        betas=config.betas,
        weight_decay=config.weight_decay,
        fused=fused
    )
    return optimizer

def prepare_ddp(config: DDPConfig, train_config: TrainConfig):
    ranks = [config.rank, config.local_rank, config.world_size]
    assert all([r != -1 for r in ranks]), "DDP configuration is not set properly"
    assert train_config % config.world_size == 0, "Batch size should be divisible by world size"
    init_process_group(backend=config.backend)
    device = f"cuda:{config.local_rank}"
    master_process = config.rank == 0
    seed_offset = config.rank
    return dict(
        device=device,
        master_process=master_process,
        seed_offset=seed_offset
    )

def prepare_single_gpu(config: TrainConfig):
    device = torch.device(config.device)
    return dict(
        device=device,
        master_process=True,
        seed_offset=0
    )


@torch.no_grad()
def estimate_loss(
    model: GPT, 
    model_config: GPTConfig,
    train_config: TrainConfig,
    ctx: torch.autocast | nullcontext
) -> dict:
    out = {}
    model.eval()
    kwargs = dict(
        block_size=model_config.block_size,
        batch_size=train_config.batch_size,
        device=torch.device(train_config.device),
        device_type=train_config.device
    )
    for split in ['train', 'val']:
        losses = torch.zeros(train_config.eval_steps)
        for k in range(train_config.eval_steps):
            x, y = get_batch(split=split, **kwargs)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    
    model.train()
    
    return out


def get_batch(
    block_size: int, 
    batch_size: int, 
    device: torch.device, 
    device_type: str, 
    split: str,
    data_dir: str = 'data',
) -> Tuple[torch.Tensor, torch.Tensor]:
    # recreate np.memmap every batch to avoid a memory leak:
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    assert split in ['train', 'val'], "Split should be either train or val"
    data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
    
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    
    return x, y