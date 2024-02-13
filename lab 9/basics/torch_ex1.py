import numpy as np
import torch

# initializing a numpy array
a = np.array(1)

# initializing a tensor
b = torch.tensor(1)

print(a)
print(b)

print(type(a))
print(type(b))

#numpy
# initializing two arrays
a = np.array(2)
b = np.array(1)
print(a,b)

print(a+b)

# subtraction
print(b-a)

# multiplication
print(a*b)

# division
print(a/b)

#pytorch on tensors
a = torch.tensor(2)
b = torch.tensor(1)
print(a,b)

# addition
print(a+b)

# subtraction
print(b-a)

# multiplication
print(a*b)

# division
print(a/b)

# matrix of zeros
a = np.zeros((3,3))
print(a)
print(a.shape)

# matrix of zeros
a = torch.zeros((3,3))
print(a)
print(a.shape)

# setting the random seed for numpy
np.random.seed(42)
# matrix of random numbers
a = np.random.randn(3,3)
print(a)

# setting the random seed for pytorch
torch.manual_seed(42)
# matrix of random numbers
b = torch.randn(3,3)
print(b)


