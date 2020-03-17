import numpy as np


def get_range(packed):
    if isinstance(packed, (list, tuple)):
        if len(packed) == 1 and isinstance(packed[0], (int, float)):
            low, high = min(-packed[0], packed[0]), max(-packed[0], packed[0])
        elif (len(packed) >= 2 and isinstance(packed[0], (int, float)) and
              isinstance(packed[1], (int, float))):
            low, high = packed[0], packed[1]
        else:
            raise ValueError('Unsupported range type (must be int or float)')
    elif isinstance(packed, (int, float)):
        low, high = min(-packed, packed), max(-packed, packed)
    else:
        raise ValueError('Unsupported range type (must be int or float)')
    return low, high


def get_translation_matrix(low=-0.2, high=0.2):
    delta = np.random.uniform(low=low, high=high, size=(3,))
    matrix = np.eye(4)
    matrix[0:3, 3] = delta
    return matrix


def get_rotation_matrix(low=-np.pi, high=np.pi, theta=None):
    if theta is not None:
        theta_x, theta_y, theta_z = theta[0], theta[1], theta[2]
    else:
        theta_x = np.random.uniform(low=low, high=high)
        theta_y = np.random.uniform(low=low, high=high)
        theta_z = np.random.uniform(low=low, high=high)

    matrix_x = np.eye(4)
    matrix_x[1, 1] = np.cos(theta_x)
    matrix_x[1, 2] = -np.sin(theta_x)
    matrix_x[2, 1] = -matrix_x[1, 2]
    matrix_x[2, 2] = matrix_x[1, 1]

    matrix_y = np.eye(4)
    matrix_y[0, 0] = np.cos(theta_y)
    matrix_y[0, 2] = np.sin(theta_y)
    matrix_y[2, 0] = -matrix_y[0, 2]
    matrix_y[2, 2] = matrix_y[0, 0]

    matrix_z = np.eye(4)
    matrix_z[0, 0] = np.cos(theta_z)
    matrix_z[0, 1] = -np.sin(theta_z)
    matrix_z[1, 0] = -matrix_z[0, 1]
    matrix_z[1, 1] = matrix_z[0, 0]
    return np.matmul(matrix_z, np.matmul(matrix_y, matrix_x))


def get_shearing_matrix(low=-0.2, high=0.2):
    matrix = np.random.uniform(low=low, high=high, size=(4, 4))
    matrix[0:3, 3] = 0
    matrix[3, 0:3] = 0
    matrix[range(4), range(4)] = 1
    return matrix
