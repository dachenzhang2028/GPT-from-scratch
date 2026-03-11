from micrograd import Value
from neuron import MLP
import torch
import random

def test_grad():
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(gmg.data - gpt.data.item()) < tol
    # backward pass went well
    assert abs(amg.grad - apt.grad.item()) < tol, f'{amg.grad=},{apt.grad.item()=}'
    assert abs(bmg.grad - bpt.grad.item()) < tol
    print('test grad: all the output are correct!')

def f(x):
    w = [0.5,2] 
    eps = random.gauss(0,0.1)
    out = sum(wi * xi for wi,xi in zip(w,x)) + eps
    return out 
def test_mlp():
    random.seed(42)
    mlp = MLP(2,[4,1])
    region = torch.linspace(-5,5,100).tolist()
    x = [[x1,x2] for x1 in region for x2 in region]
    y = [f(xi) for xi in x ]
    for i in range(1000):
        batch = [random.randint(0,10000-1) for _ in range(500)]
        xb = [x[b] for b in batch]
        yb = [y[b] for b in batch]
        y_pred = [mlp(xi)[0] for xi in xb]
        loss = sum((yi-y_predi)**2 for yi,y_predi in zip(yb,y_pred)) / len(xb)
        for p in mlp.parameters():
            p.grad = 0
        loss.backward()
        lr = 0.05 if i<100 else 0.01
        for p in mlp.parameters():
            p.data += -lr * p.grad
        if i%100 == 0:
            print(f'step: {i:2d} | loss: {loss}')
    y_pred = [mlp(xi)[0] for xi in x]
    total_loss = sum((yi-y_predi)**2 for yi,y_predi in zip(y,y_pred)) / len(x)
    relative_error = sum((yi-y_predi).abs() / (abs(yi) +1e-8) for yi,y_predi in zip(y,y_pred)) / len(x)
    print(f'total loss: {total_loss}')
    print(f'relative error: {relative_error}')

test_grad()
test_mlp()


