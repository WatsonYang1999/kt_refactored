import numpy as numpy
# 创建一个 m x n 的 NumPy 数组
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# 按行求和 (每行的和)
row_sums = np.sum(matrix, axis=1)

# 按列求和 (每列的和)
col_sums = np.sum(matrix, axis=0)

print("按行求和:")
print(row_sums)

print("按列求和:")
print(col_sums)
