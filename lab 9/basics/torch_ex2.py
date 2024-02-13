import torch

torch.manual_seed(42)

a = torch.randn(3,3)
b = torch.randn(3,3)
print(a)
print(b)
print('\n')


# matrix addition
print(torch.add(a,b), '\n')

# matrix subtraction
print(torch.sub(a,b), '\n')

# matrix multiplication
print(torch.mm(a,b), '\n')

# matrix division
print(torch.div(a,b))

# original matrix
print(a, '\n')

# matrix transpose
print(torch.t(a))

