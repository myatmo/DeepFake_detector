import torch
x = torch.rand(5, 3)
print(x)

assert torch.cuda.is_available()

device = torch.device("cuda")
x = torch.rand(2, 2, device=device)
y = torch.rand(2, 2)
y = y.to(device)
z = x.add(y)
