import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CasualSelfAttention(nn.Module):
    def __init__(self,config,layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        self.c_proj = nn.Linear(config.n_embd,config.n_embd)
        self.register_buffer('mask',torch.tril(torch.ones(config.block_size,config.block_size)))
    def forward(self,x,past_key_values,use_cache):
        B,T,C = x.shape
        n_head = self.config.n_head
        d_head = C // n_head
        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.config.n_embd,dim=2)
        q = q.view(B,T,n_head,d_head).transpose(1,2) # (B,n_head,T,d_head)
        k = k.view(B,T,n_head,d_head).transpose(1,2)
        v = v.view(B,T,n_head,d_head).transpose(1,2)

        if use_cache:
            if past_key_values[self.layer_idx] is not None:
                past_k,past_v = past_key_values[self.layer_idx]
                k = torch.cat((past_k,k),dim=2)
                v = torch.cat((past_v,v),dim=2)
            past_key_values[self.layer_idx] = (k,v)
        _,_,T_total,_ = k.shape
        w = q @ k.transpose(-1,-2) / math.sqrt(d_head)
        w = w.masked_fill(self.mask[T_total-T:T_total,:T_total]==0,float('-inf')) # (B,n_head,T,T)
        w = F.softmax(w,dim=-1)
        y = w @ v
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd,4*config.n_embd)
        self.c_proj = nn.Linear(4*config.n_embd,config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self,config,layer_idx):
        super().__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config,layer_idx)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    def forward(self,x,past_key_values,use_cache):
        x = x + self.attn(self.ln_1(x),past_key_values,use_cache)
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size,config.n_embd),
            wpe = nn.Embedding(config.block_size,config.n_embd),
            h = nn.ModuleList([Block(config,i) for i in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd,config.vocab_size,bias=False)
        # weight sharing
        self.transformer.wte.weight = self.lm_head.weight
    def forward(self,x,target=None,past_key_values=None,pos=None,use_cache=False):
        B,T = x.shape
        tok_emb = self.transformer.wte(x)
        if T > 1: #prefill:
            pos_emb = self.transformer.wpe(torch.arange(T,device=x.device))
        else:
            pos_emb = self.transformer.wpe(torch.tensor(pos,dtype=torch.long,device=x.device))
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x,past_key_values,use_cache)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        if target is not None:
            loss = F.cross_entropy(logits.view(B*T,-1),target.view(B*T))
            return loss
        else:
            return logits
    
    def generate(self,x,max_new_tokens=30,use_cache=True):
        with torch.no_grad():
            if use_cache:
                past_key_values = [None] * self.config.n_layer
            else:
                past_key_values = None
            
            idx = x
            for _ in range(max_new_tokens):
                pos = x.shape[1]-1
                if use_cache:
                    inp = idx
                else:
                    inp = x
                logits = self(inp,past_key_values=past_key_values,pos=pos,use_cache=use_cache)[:,-1]
                probs = F.softmax(logits,dim=-1)
                topk_probs,topk_indices = torch.topk(probs,k = 50,dim=-1)
                idx = torch.multinomial(topk_probs,num_samples=1)
                idx = torch.gather(topk_indices,-1,idx)
                x = torch.cat((x,idx),dim=1)
        return x
    
    @classmethod
    def from_pretrained(cls):
        from config import GPTConfig
        from transformers import GPT2LMHeadModel
        model = cls(GPTConfig)
        sd = model.state_dict()
        model_hf = GPT2LMHeadModel.from_pretrained('/Users/dachen/Documents/AI4s_lab/zero_to_here/GPT2/gpt2_124m')
        sd_hf = model_hf.state_dict()
        for k in sd_hf:
            if any(k.endswith(s) for s in ['c_attn.weight','c_proj.weight','c_fc.weight']):
                with torch.no_grad():
                    assert sd_hf[k].T.shape == sd[k].shape
                    sd[k].copy_(sd_hf[k].T)
            else:
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model





