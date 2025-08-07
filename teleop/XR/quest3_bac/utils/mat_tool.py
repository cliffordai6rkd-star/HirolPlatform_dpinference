import numpy as np


def mat_update(prev_mat: np.ndarray, mat: np.ndarray) -> tuple:
    """Updates the previous matrix with the new matrix if the new matrix is non-singular.

    Args:
        prev_mat (np.ndarray): The previous matrix.
        mat (np.ndarray): The new matrix to update with.

    Returns:
        tuple: A tuple containing the updated matrix and a boolean flag indicating success.
    """
    if np.linalg.det(mat) == 0:
        return prev_mat, False  # Returns previous matrix and False flag if the new matrix is singular (determinant = 0).
    else:
        return mat, True  # Returns the new matrix and True flag if it is non-singular.


def fast_mat_inv(mat: np.ndarray) -> np.ndarray:
    """Computes the inverse of a 4x4 matrix using a fast method.

    Args:
        mat (np.ndarray): The 4x4 matrix to be inverted.

    Returns:
        np.ndarray: The inverse of the input matrix.
    """
    ret = np.eye(4)
    ret[:3, :3] = mat[:3, :3].T
    ret[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
    return ret
