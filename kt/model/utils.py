import torch

def combine_qa_or_sa(A: torch.Tensor, B: torch.Tensor, N: float) -> torch.Tensor:
    """
    根据指定逻辑计算矩阵 C。
    
    C[i, j] = A[i, j] + B[i, j] * N
    如果 A[i, j] 或 B[i, j] 小于 0，则 C[i, j] = -1
    
    参数:
    - A (torch.Tensor): 任意元素的矩阵
    - B (torch.Tensor): 元素为 0 或 1 的矩阵
    - N (float): 标量参数
    
    返回:
    - torch.Tensor: 计算得到的矩阵 C
    """
    # 矩阵计算
    print('----------------------------------------------------')
    print(A)
    print(B)
    print(N)
    C = A + B * N

    # 条件处理：A 或 B 小于 0
    condition = (A < 0) | (B < 0)
    C[condition] = -1
    
    return C
