import torch
import math
from torch import nn
from torch.nn import functional as F
from config import GPTConfig
from typing import Tuple

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(config.embedding_dim, 4*config.embedding_dim, bias=config.bias)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(4*config.embedding_dim, config.embedding_dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x

class LayerNorm(nn.Module):
    """ 
    The layer normalization is a simple normalization technique that normalizes the activations of a layer.
        https://arxiv.org/pdf/1607.06450
    """
    def __init__(self, features: int, bias: bool = True):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features)) if bias else None
        self.eps = 1e-6
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.size(), self.weight, self.bias, self.eps)

class CasualSelfAttention(nn.Module):
    """ 
    Attention is all you need :)
        https://arxiv.org/pdf/1706.03762 
    """
    def __init__(self, config: GPTConfig):
        super(CasualSelfAttention, self).__init__()
        self.num_heads = config.num_heads
        self.embedding_dim = config.embedding_dim
        self.dropout = config.dropout

        self.qkv = nn.Linear(self.embedding_dim, self.embedding_dim*3, bias=config.bias)
        self.qkv_proj = nn.Linear(config.embedding_dim, config.embedding_dim, bias=config.bias)

        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)

        self._register_flash_attention(config.block_size)
    
    def _register_flash_attention(self, block_size: int):
        """ 
        The flash attention is a faster version of the scaled dot product attention
        Bias here is used to mask the right side of the input sequence to make the attention causal
        """
        # massive speedup on gpu for larger models
        self.supports_flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        # register buffer for masking s.t. attention is applied only to the left in the input sequence
        # this is what makes the self-attention casual, meaning we don't look into the future
        # by zeroing out the upper right triangle of the attention matrix -> the softmax will make those 0
        bias = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("bias", bias)
    
    def _attention_mechanism(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seq_len: int) -> torch.Tensor:
        if self.supports_flash:
            dropout = self.dropout if self.training else 0.0
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=dropout, is_causal=True
            )
        else:
            x = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # zero out the upper right triangle of the attention matrix
            x = x.masked_fill(self.bias[:, :, :seq_len, :seq_len] == 0, float('-inf'))
            x = F.softmax(x, dim=-1)
            x = self.attention_dropout(x)
            x = torch.matmul(x, v)
        
        return x
    
    def _transform_attn(self, x: torch.Tensor, batch_size: int, seq_len: int, n_embed: int) -> torch.Tensor:
        return x.view(batch_size, seq_len, self.num_heads, n_embed // self.num_heads).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, n_embed = x.size()
        q, k, v = self.qkv(x).split(self.embedding_dim, dim=2)
        q, k, v = map(lambda t: self._transform_attn(t, batch_size, seq_len, n_embed), (q, k, v))

        x = self._attention_mechanism(q, k, v, seq_len)
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embed)
        x = self.qkv_proj(x)
        x = self.residual_dropout(x)

        return x

class GPTBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super(GPTBlock, self).__init__()
        self.ln1 = LayerNorm(config.embedding_dim, bias=config.bias) 
        self.ln2 = LayerNorm(config.embedding_dim, bias=config.bias)
        self.attn = CasualSelfAttention(config)
        self.mlp = MLP(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x
        
class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super(GPT, self).__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                word_embd = nn.Embedding(config.vocab_size, config.embedding_dim),
                pos_embd = nn.Embedding(config.block_size, config.embedding_dim),
                drop = nn.Dropout(config.dropout),
                blocks = nn.ModuleList([GPTBlock(config) for _ in range(config.num_layers)]),
                layer_norm = LayerNorm(config.embedding_dim, bias=config.bias)
            )
        )
        # lm head is the language model head, mapping the output of the transformer to the vocab
        self.head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        #TODO: weight tying can cause issues during torch compile, check this
        self.transformer.word_embd.weight = self.head.weight

        # initialize weights for all modules
        self.apply(self._init_weights)
        # apply scaling to the qkv projection weights as in GPT-2 paper
        for name, param in self.named_parameters():
            if name.endswith("qkv_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02/math.sqrt(2 * config.num_layers))
    
        print(f"Numer of parameters: {self._count_parameters()}")        
    
    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            module.bias.data.zero_() if module.bias is not None else None
    
    def _count_parameters(self, without_embed: bool = True) -> int:
        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if without_embed:
            pos_size = self.transformer.word_embd.weight.size()
            param_count -= pos_size[0] * pos_size[1]
        
        return param_count
    
    @staticmethod
    def _calculate_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        
        return loss
    
    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        word_embd = self.transformer.word_embd(x)
        pos_embd = self.transformer.pos_embd(torch.arange(x.size(1), device=x.device))
        x = self.transformer.drop(word_embd + pos_embd)

        for block in self.transformer.blocks:
            x = block(x)
        
        x = self.transformer.layer_norm(x)

        if isinstance(targets, torch.Tensor):
            logits = self.head(x)
            loss = self._calculate_loss(logits, targets)
            return logits, loss
    
        logits = self.head(x[:, [-1], :])
        
        return logits, None
    
    @torch.no_grad()
    def _predict_next_token(self, x: torch.Tensor, temperature: float = 0.2, top_k: int | None = 50) -> torch.Tensor:
        
        x_conditioned = x if x.size(1) < self.config.block_size else x[:, -self.config.block_size:]
        
        logits, _ = self(x_conditioned)
        logits = logits[:, -1, :] / temperature
        # top k is used to sample from the top k most likely tokens instead of the full distribution
        # just a counter-measure to limit randomness
        if top_k is not None:
            top_k = min(top_k, logits.size(-1))
            values = torch.topk(logits, top_k).values
            logits[logits < values[:, [-1]]] = float('-inf')
        
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token

    @torch.no_grad()
    def generate(self, x: torch.Tensor, max_length: int = 100, temperature: float = 0.2, top_k: int | None = 50) -> torch.Tensor:
        """Generates a full sequence without streaming."""
        self.eval()
        
        for _ in range(max_length):
            next_token = self._predict_next_token(x, temperature, top_k)
            x = torch.cat((x, next_token), dim=-1) 
        
        return x

    @torch.no_grad()
    def generate_stream(self, x: torch.Tensor, max_length: int = 100, temperature: float = 0.2, top_k: int | None = 50):
        """Streams the generation, yielding one token at a time."""
        self.eval()
        
        for _ in range(max_length):
            next_token = self._predict_next_token(x, temperature, top_k)
            x = torch.cat((x, next_token), dim=-1) 
            # only yield the last token
            yield x[:, -1:]

