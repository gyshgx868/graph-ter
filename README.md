# GraphTER: Unsupervised Learning of Graph Transformation Equivariant Representations via Auto-Encoding Node-wise Transformations

This repository is the official PyTorch implementation of the following paper:

Xiang Gao, Wei Hu, Guo-Jun Qi, "GraphTER: Unsupervised Learning of Graph Transformation Equivariant Representations via Auto-Encoding Node-wise Transformations," In *Proceedings of the IEEE/CVF Conferences on Computer Vision and Pattern Recognition (CVPR)*, Seattle, Washington, June 2020.

## Requirements

 - Python3==3.7.4
 - pytorch==1.2.0
 - torchvision==0.4.2
 - tensorboardX==1.9
 - hdf5==1.10.4

**Note:** A cuDNN error `CUDNN_STATUS_NOT_SUPPORTED` occurs when you use PyTorch with a version greater than `1.2.0`, but we have not figured out this issue in this code.

## Datasets and Pre-trained Models

To evaluate the model, `ModelNet40` and `ShapeNet Part` dataset in HDF5 format are required to be downloaded and unzipped to the `data` folder.

Download `ModelNet40` dataset for classification task by running the following commands:

```shell script
cd ./data
wget https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip
unzip modelnet40_ply_hdf5_2048.zip
rm modelnet40_ply_hdf5_2048.zip
```

Download `ShapeNet Part` dataset for segmentation task by running the following commands:

```shell script
cd ./data
wget https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip
unzip shapenet_part_seg_hdf5_data.zip
rm shapenet_part_seg_hdf5_data.zip
mv hdf5_data shapenet_part
```

The pre-trained models can be downloaded from [OneDrive](https://pkueducn-my.sharepoint.com/:f:/g/personal/gyshgx868_pku_edu_cn/EhT2xo0syHtBgXqnHr9re28B3yFmeyTjXad8RcMa9bovwQ) or [GoogleDrive](https://drive.google.com/drive/folders/18pQrOx9GwiC2WXe5s-9OI_zozAmSdDsr?usp=sharing), and manually place the `classification` and `segmentation` folders into the `pretrained` folder.

## Usage

We take classification task as an example to introduce how to use our code, and segmentation task is similar.

### Testing Pre-trained Models

You can run the following command to reproduce the results in our paper:

```shell script
python main_classification.py --phase test --device 0 1 2 3 --test-batch-size 32 --data-path ./data --transform [full class name] --backbone [backbone checkpoints] --classifier [classifier checkpoints]
```

You should specify the `[full class name]` (e.g., `graph_ter_cls.transforms.GlobalRotate`), the `[backbone checkpoints]`, and the `[classifier checkpoints]`. For instance:

```shell script
python main_classification.py --phase test --device 0 1 2 3 --test-batch-size 32 --data-path ./data --transform graph_ter_cls.transforms.GlobalRotate --backbone ./pretrained/classification/backbone_global_rotate_aniso.pt --classifier ./pretrained/classification/classifier_global_rotate_aniso.pt
```

You can also run the following command to make it easier to evaluate the pre-trained models:

```shell script
python main_classfication.py --config ./config/classification/global_rotate_aniso.yaml
```

Note that we test our model on 4 NVIDIA RTX 2080Ti GPUs. You can use `--device` to specify the indices of GPUs and use `--test-batch-size` to fit the memory, or appoint `--use-cuda false` to use CPUs for evaluation.

### Training

To train a feature extractor in an unsupervised fashion, run

```shell script
python main_classification.py --phase backbone --device 0 1 2 3 --train-batch-size 32 --data-path ./data --transform [full class name]
``` 

Again, you should specify the `[full class name]` (e.g., `graph_ter_cls.transforms.GlobalRotate`).

After training the feature extractor, you need to train the classifier by running the following command:

```shell script
python main_classification.py --phase classifier --device 0 1 2 3 --train-batch-size 32 --data-path ./data --transform [full class name] --backbone [backbone checkpoints]
``` 

The `[backbone checkpoints]` refers to checkpoints of the trained feature extractor.

The log files, network parameters, and TensorBoard logs will be saved to `results` folder by default. We can use TensorBoard to view the training progress:

```shell script
tensorboard --logdir ./results/tensorboard
```

For more hyper-parameters, please refer to `graph_ter_cls/tools/configuration.py` and `graph_ter_seg/tools/configuration.py`.

## Reference

Please cite our paper if you use any part of the code from this repository:

```
@inproceedings{gao2020graphter,
  title={Graph{TER}: Unsupervised Learning of Graph Transformation Equivariant Representations via Auto-Encoding Node-wise Transformations},
  author={Gao, Xiang and Hu, Wei and Qi, Guo-Jun},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month={June},
  year={2020}
}
```

## Acknowledgement

Our code is released under MIT License (see `LICENSE` for details). Some of the code in this repository was borrowed from the following repositories:

 - [AET](https://github.com/maple-research-lab/AET)
 - [DGCNN](https://github.com/WangYueFt/dgcnn)
 - [ST-GCN](https://github.com/yysijie/st-gcn)
