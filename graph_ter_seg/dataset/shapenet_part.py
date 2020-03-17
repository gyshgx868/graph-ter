import json
import os

import numpy as np

from torch.utils.data import Dataset

from graph_ter_seg.tools.utils import load_h5_data_label_seg


class ShapeNetPart(Dataset):
    def __init__(self, data_path, num_points=2048, transform=None,
                 phase='train'):
        self.data_path = os.path.join(data_path, 'shapenet_part')
        self.num_points = num_points
        self.num_classes = 16
        self.num_parts = 50
        self.classes_dict = {
            'airplane': [0, 1, 2, 3],
            'bag': [4, 5],
            'cap': [6, 7],
            'car': [8, 9, 10, 11],
            'chair': [12, 13, 14, 15],
            'earphone': [16, 17, 18],
            'guitar': [19, 20, 21],
            'knife': [22, 23],
            'lamp': [24, 25, 26, 27],
            'laptop': [28, 29],
            'motorbike': [30, 31, 32, 33, 34, 35],
            'mug': [36, 37],
            'pistol': [38, 39, 40],
            'rocket': [41, 42, 43],
            'skateboard': [44, 45, 46],
            'table': [47, 48, 49]
        }
        self.shape_names = {}
        for cat in self.classes_dict.keys():
            for label in self.classes_dict[cat]:
                self.shape_names[label] = cat

        self.transform = transform
        if phase == 'train':
            files = os.path.join(self.data_path, 'train_hdf5_file_list.txt')
            # files = os.path.join(self.data_path, 'val_hdf5_file_list.txt')
        else:
            files = os.path.join(self.data_path, 'test_hdf5_file_list.txt')
        file_list = [line.rstrip() for line in open(files)]
        num_files = len(file_list)
        self.coordinates = list()
        self.labels = list()
        self.segmentation = list()
        for i in range(num_files):
            cur_file = os.path.join(self.data_path, file_list[i])
            cur_data, cur_label, cur_seg = load_h5_data_label_seg(cur_file)
            cur_data = cur_data[:, 0:self.num_points, :]
            cur_seg = cur_seg[:, 0:self.num_points]
            self.coordinates.append(cur_data)
            self.labels.append(cur_label)
            self.segmentation.append(cur_seg)
        self.coordinates = np.vstack(self.coordinates).astype(np.float32)
        self.labels = np.vstack(self.labels).squeeze().astype(np.int64)
        self.segmentation = np.vstack(self.segmentation).astype(np.int64)

        color_map_file = os.path.join(self.data_path, 'part_color_mapping.json')
        self.color_map = json.load(open(color_map_file, 'r'))

    def __getitem__(self, index):
        coord = np.transpose(self.coordinates[index])  # 3 * N
        label = self.labels[index]
        seg = self.segmentation[index]  # N
        data = (coord,)
        # transform coordinates
        if self.transform is not None:
            transformed, matrix, mask = self.transform(coord)
            data += (transformed, matrix, mask)
        one_hot = np.zeros(shape=(16, 1), dtype=np.int64)
        one_hot[label, 0] = 1
        data += (one_hot, seg)
        return data

    def __len__(self):
        return self.coordinates.shape[0]

    def output_colored_point_cloud(self, points, seg, output_file):
        with open(output_file, 'w') as handle:
            for i in range(self.num_points):
                color = self.color_map[seg[i]]
                handle.write(
                    'v %f %f %f %f %f %f\n' % (
                        points[0][i], points[1][i], points[2][i],
                        color[0], color[1], color[2]
                    )
                )


def main():
    from graph_ter_cls.transforms.global_translate import GlobalTranslate
    train = ShapeNetPart(
        '../../data', phase='train', transform=GlobalTranslate(num_points=2048)
    )
    test = ShapeNetPart('../../data', phase='test')
    print(len(train))
    print(len(test))
    # print(train.coordinates.shape)
    # print(train.labels.shape)
    # print(train.segmentation.shape)
    #
    # coord, transformed, matrix, mask, label, seg = train.__getitem__(0)
    # print(coord.shape, transformed.shape)
    # print(matrix.shape, mask.shape, label.shape, seg.shape)
    # print(label)
    # print(seg)

    # train.output_colored_point_cloud(coord, seg, './test.obj')


if __name__ == '__main__':
    main()
