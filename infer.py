from model import GPT
import tiktoken
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

model = GPT.from_pretrained()
model.to(device)

torch.manual_seed(42)
if torch.cuda.is_available:
    torch.cuda.manual_seed(42)

prompt = "Hello, I'm a language model,"
tokenizer = tiktoken.get_encoding('gpt2')
tokens = tokenizer.encode(prompt)
x = torch.tensor(tokens,dtype=torch.long,device=device).unsqueeze(0).repeat(5,1)

out =  model.generate(x,max_new_tokens=30,use_cache=False)
for i in range(out.shape[0]):
    output = tokenizer.decode(out[i].detach().tolist())
    print(output)



