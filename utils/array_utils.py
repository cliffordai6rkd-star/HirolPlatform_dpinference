"""数组格式转换工具函数。"""

import numpy as np


def ensure_column_vector(array: np.ndarray) -> np.ndarray:
    """确保数组是列向量格式。
    
    Args:
        array: 输入数组
        
    Returns:
        列向量格式的数组 (n, 1)
    """
    array = np.asarray(array)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    elif array.ndim == 2 and array.shape[1] == 1:
        return array
    elif array.ndim == 2 and array.shape[0] == 1:
        return array.T
    else:
        raise ValueError(f"无法将形状为 {array.shape} 的数组转换为列向量")


def ensure_flat_array(array: np.ndarray) -> np.ndarray:
    """确保数组是一维格式。
    
    Args:
        array: 输入数组
        
    Returns:
        一维数组
    """
    return np.asarray(array).flatten()