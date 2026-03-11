from dataclasses import dataclass
@dataclass
class GPTConfig:
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    vocab_size: int = 50257
    block_size: int = 1024

class TrainConfig:
    Batch_size: int = 8
    Block_size: int = 32
    max_steps: int = 15
    warmup_steps: int = 10
    max_lr: float = 3e-4
    betas: tuple =(0.9,0.95)
    weight_decay: float = 0.1
