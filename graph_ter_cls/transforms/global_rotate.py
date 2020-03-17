import json
import random

import numpy as np

from graph_ter_cls.transforms import utils
from graph_ter_cls.transforms.transformer import Transformer


class GlobalRotate(Transformer):
    def __init__(self,
                 num_samples=256,
                 mode='isotropic',  # isotropic or anisotropic
                 transform_range=(-np.pi / 36.0, np.pi / 36.0)):
        super().__init__(out_features=3)
        self.num_samples = num_samples
        self.mode = mode
        self.low, self.high = utils.get_range(transform_range)

    @staticmethod
    def _build_rotation_matrix(parameters):
        theta_x = parameters[0]
        theta_y = parameters[1]
        theta_z = parameters[2]

        matrix_x = np.eye(3)
        matrix_x[1, 1] = np.cos(theta_x)
        matrix_x[1, 2] = -np.sin(theta_x)
        matrix_x[2, 1] = -matrix_x[1, 2]
        matrix_x[2, 2] = matrix_x[1, 1]

        matrix_y = np.eye(3)
        matrix_y[0, 0] = np.cos(theta_y)
        matrix_y[0, 2] = np.sin(theta_y)
        matrix_y[2, 0] = -matrix_y[0, 2]
        matrix_y[2, 2] = matrix_y[0, 0]

        matrix_z = np.eye(3)
        matrix_z[0, 0] = np.cos(theta_z)
        matrix_z[0, 1] = -np.sin(theta_z)
        matrix_z[1, 0] = -matrix_z[0, 1]
        matrix_z[1, 1] = matrix_z[0, 0]

        matrix = np.matmul(matrix_z, np.matmul(matrix_y, matrix_x))
        return matrix

    def __call__(self, x):
        num_points = x.shape[-1]
        if self.mode.startswith('aniso'):
            matrix = np.random.uniform(
                low=self.low, high=self.high, size=(3, self.num_samples)
            )
        else:
            matrix = np.random.uniform(
                low=self.low, high=self.high, size=(3, 1)
            )
            matrix = np.repeat(matrix, self.num_samples, axis=1)

        mask = np.sort(random.sample(range(num_points), self.num_samples))
        y = x.copy()
        if self.mode.startswith('aniso'):
            for index, transform_id in enumerate(mask):
                rotation_mat = self._build_rotation_matrix(matrix[:, index])
                y[:, transform_id] = np.dot(rotation_mat, y[:, transform_id])
        else:
            rotation_mat = self._build_rotation_matrix(matrix[:, 0])
            y[:, mask] = np.dot(rotation_mat, y[:, mask])
        mask = np.repeat(np.expand_dims(mask, axis=0), 3, axis=0)
        return y, matrix, mask

    def __repr__(self):
        info = self.get_config()
        info_json = json.dumps(info, sort_keys=False, indent=2)
        return info_json

    def get_config(self):
        result = {
            'name': self.__class__.__name__,
            'sampled points': self.num_samples,
            'mode': self.mode,
            'range': (self.low, self.high)
        }
        return result


def main():
    x = np.array([[1, 2, 3, 4, 5, 6, 7],
                  [1, 2, 3, 4, 5, 6, 7],
                  [1, 2, 3, 4, 5, 6, 7]], dtype=float)
    transform = GlobalRotate(num_samples=2, mode='isotropic')
    y, m, mask = transform(x)
    print(x)
    print(y)
    print(m)
    print(mask)


if __name__ == '__main__':
    main()
