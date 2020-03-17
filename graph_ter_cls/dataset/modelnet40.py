import os

import numpy as np

from torch.utils.data import Dataset

from graph_ter_cls.tools.utils import load_h5


class ModelNet40(Dataset):
    def __init__(self, data_path, num_points=1024, transform=None,
                 phase='train'):
        self.data_path = os.path.join(data_path, 'modelnet40_ply_hdf5_2048')
        self.num_points = num_points
        self.num_classes = 40
        self.transform = transform

        # store data
        shape_name_file = os.path.join(self.data_path, 'shape_names.txt')
        self.shape_names = [line.rstrip() for line in open(shape_name_file)]
        self.coordinates = []
        self.labels = []
        try:
            files = os.path.join(self.data_path, '{}_files.txt'.format(phase))
            files = [line.rstrip() for line in open(files)]
            for index, file in enumerate(files):
                file_name = file.split('/')[-1]
                files[index] = os.path.join(self.data_path, file_name)
        except FileNotFoundError:
            raise ValueError('Unknown phase or invalid data path.')
        for file in files:
            current_data, current_label = load_h5(file)
            current_data = current_data[:, 0:self.num_points, :]
            self.coordinates.append(current_data)
            self.labels.append(current_label)
        self.coordinates = np.vstack(self.coordinates).astype(np.float32)
        self.labels = np.vstack(self.labels).squeeze().astype(np.int64)

    def __len__(self):
        return self.coordinates.shape[0]

    def __getitem__(self, index):
        coord = np.transpose(self.coordinates[index])  # 3 * N
        label = self.labels[index]
        data = (coord,)
        # transform coordinates
        if self.transform is not None:
            transformed, matrix, mask = self.transform(coord)
            data += (transformed, matrix, mask)
        data += (label,)
        return data


def main():
    loader = ModelNet40('../../data', phase='test')
    print(len(loader))
    print(len(loader.shape_names))
    print(loader.shape_names)
    for i in range(10):
        data, label = loader.__getitem__(i)
        print(data.shape, label.shape, label)


if __name__ == '__main__':
    main()
