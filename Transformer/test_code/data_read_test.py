import numpy as np

file_path = '../data/test_100.csv'
data = open(file_path).readlines()

result = np.array([i.strip().split(',') for i in data], dtype=np.int)

print(data)
print(result)