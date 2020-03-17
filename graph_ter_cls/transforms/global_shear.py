import json
import random

import numpy as np

from graph_ter_cls.transforms import utils
from graph_ter_cls.transforms.transformer import Transformer


class GlobalShear(Transformer):
    def __init__(self,
                 num_samples=256,
                 mode='isotropic',  # isotropic or anisotropic
                 transform_range=(-0.2, 0.2)):
        super().__init__(out_features=6)
        self.num_samples = num_samples
        self.mode = mode
        self.low, self.high = utils.get_range(transform_range)

    @staticmethod
    def _build_shearing_matrix(parameters):
        matrix = np.eye(3)
        matrix[0, 1] = parameters[0]
        matrix[1, 0] = parameters[1]
        matrix[0, 2] = parameters[2]
        matrix[2, 0] = parameters[3]
        matrix[1, 2] = parameters[4]
        matrix[2, 1] = parameters[5]
        return matrix

    def __call__(self, x):
        num_points = x.shape[-1]
        if self.mode.startswith('aniso'):
            matrix = np.random.uniform(
                low=self.low, high=self.high,
                size=(self.out_features, self.num_samples)
            )
        else:
            matrix = np.random.uniform(
                low=self.low, high=self.high, size=(self.out_features, 1)
            )
            matrix = np.repeat(matrix, self.num_samples, axis=1)

        mask = np.sort(random.sample(range(num_points), self.num_samples))
        y = x.copy()
        if self.mode.startswith('aniso'):
            for index, transform_id in enumerate(mask):
                shearing_mat = self._build_shearing_matrix(matrix[:, index])
                y[:, transform_id] = np.dot(shearing_mat, y[:, transform_id])
        else:
            shearing_mat = self._build_shearing_matrix(matrix[:, 0])
            y[:, mask] = np.dot(shearing_mat, y[:, mask])
        mask = np.repeat(
            np.expand_dims(mask, axis=0), self.out_features, axis=0
        )
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
    transform = GlobalShear(num_samples=2, mode='isotropic')
    y, m, mask = transform(x)
    print(x)
    print(y)
    print(m)
    print(mask)


if __name__ == '__main__':
    main()
