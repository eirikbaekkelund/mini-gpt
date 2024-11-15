import os
import torch
from pydantic import BaseModel

class GPTConfig(BaseModel):
    vocab_size: int = 50304 # GPT2 vocab size padded by 64 from 50257 
    block_size: int = 1024
    num_layers: int = 4
    num_heads: int = 4
    embedding_dim: int = 768
    dropout: float = 0.1
    bias: bool = False # used for LayerNorm and MLPs

class TrainConfig(BaseModel):
    split: float = 0.9
    eval_steps: int = 500
    save_checkpoint: bool = True
    batch_size: int = 12
    lr: float = 6e-4
    decay_lr: bool = True
    grad_clip: float = 1.0
    epochs: int = 10
    grad_accum: int = 5 * 8
    warm_up_steps: int = 2000
    weight_decay: float = 1e-1
    betas: tuple = (0.9, 0.95)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    compile_model: bool = True
    wandb: bool = True
    max_iters: int = 600000 
    min_lr: float = 6e-5

class DDPConfig(BaseModel):
    backend: str = "nccl"
    local_rank: int = os.environ.get("LOCAL_RANK", -1)
    world_size: int = os.environ.get("WORLD_SIZE", -1)
    rank: int = os.environ.get("RANK", -1)
