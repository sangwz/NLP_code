import numpy as np
import torch
file_path = '../data/test_100.csv'
data = open(file_path).readlines()
print(data)
print(eval(data[0]))
# result = np.array([','.join(i.strip().split(',')) for i in data], dtype=np.int)
result = torch.tensor([eval(i) for i in data], dtype=np.int)

print(data)
print(result)