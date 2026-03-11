from model import GPT
from config import GPTConfig,TrainConfig
import torch
import torch.nn as nn
import tiktoken
import math

class DataLoader:
    def __init__(self,B,T):
        self.B = B
        self.T = T
        with open('input.txt','r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens,dtype=torch.long)
        self.current_pos = 0
    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_pos:self.current_pos + B*T +1]
        x = buf[:-1].view(B,T)
        y = buf[1:].view(B,T)
        self.current_pos += B*T
        if self.current_pos + B*T + 1 > len(self.tokens):
            self.current_pos = 0
        return x,y


def get_optimizer(model,weight_decay=0.1,betas=(0.9,0.95),lr=3e-4):
    p = list(model.parameters())
    weight_decay_params = [v for v in p if v.ndim>=2]
    nondecay_params = [v for v in p if v.ndim<2]
    param_group = [
        {'params':weight_decay_params,'weight_decay':weight_decay},
        {'params':nondecay_params, 'weight_decay':0}
    ]
    optimizer = torch.optim.AdamW(param_group,betas=betas,lr=lr)
    return optimizer
def get_lr(step):
    if step > max_steps - 1:
        return min_lr
    elif step < warmup_steps:
        return max_lr * (step+1) / warmup_steps
    else:
        ratio = (step + 1 - warmup_steps) / (max_steps - warmup_steps)
        return min_lr + (max_lr - min_lr) * (math.cos(math.pi * ratio) + 1) / 2


if __name__ == "__main__":
    B,T = TrainConfig.Batch_size, TrainConfig.Block_size
    max_steps = TrainConfig.max_steps
    warmup_steps = TrainConfig.warmup_steps
    max_lr = TrainConfig.max_lr
    min_lr = max_lr * 0.1
    weight_decay = TrainConfig.weight_decay
    betas = TrainConfig.betas

    model = GPT(GPTConfig)
    dataloader = DataLoader(B,T)
    optimizer = get_optimizer(model,weight_decay=weight_decay,betas=betas)

    for i in range(max_steps):
        optimizer.zero_grad()
        loss_accum = 0
        for _ in range(10): # grad accumlation
            x,y = dataloader.next_batch()
            loss = model(x,y)
            loss = loss / 10
            loss_accum += loss.item()
            loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(),1) # grad
        lr = get_lr(i)
        for g in optimizer.param_groups:
            g['lr'] = lr
        optimizer.step()
        print(f'step {i:4d} | loss: {loss_accum:.6f} | norm: {norm:.4f} | lr: {lr:.4e}')


