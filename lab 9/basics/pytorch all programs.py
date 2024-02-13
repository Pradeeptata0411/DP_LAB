import torch

# example1
r1 = torch.tensor(1)
print(r1)

# example2
r2 = torch.tensor(1, dtype=torch.int32)
print(r2)

# example3
q1_decimal = torch.tensor(1.12345, dtype=torch.float32)
print(q1_decimal)


# example5
vector = torch.tensor([1, 3, 6])
print(vector)

# example6
q2_boolean = torch.tensor([True, True, False], dtype=torch.bool)
print(q2_boolean)

# example7
q2_matrix = torch.tensor([[1, 2], [3, 4]])
print(q2_matrix)

# example8
print(torch.zeros(10))

# example9
print(torch.ones(10, 10))

# example10
type_float = torch.tensor(3.123456788, dtype=torch.float32)
type_int = type_float.int()
print(type_int.dtype)
print(type_float.dtype)

# example11
tensor_a = torch.tensor([3, 4], dtype=torch.int32)
tensor_b = torch.tensor([1, 2], dtype=torch.int32)
tensor_add = torch.add(tensor_a, tensor_b)
print(tensor_add)
