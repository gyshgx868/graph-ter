import argparse
import h5py
import torch


def import_class(name):
    try:
        components = name.split('.')
        module = __import__(components[0])
        for c in components[1:]:
            module = getattr(module, c)
    except AttributeError:
        module = None
    return module


def load_h5(file_name):
    f = h5py.File(file_name)
    data = f['data'][:]
    label = f['label'][:]
    return data, label


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_total_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total, 'Trainable': trainable}


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    x2 = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -x2 - inner - x2.transpose(2, 1)
    indices = pairwise_distance.topk(k=k, dim=-1)[1]
    return indices  # (batch_size, num_points, k)


def get_edge_feature(x, k=20):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    num_feats = x.size(1)

    indices = knn(x, k=k)  # (batch_size, num_points, k)
    indices_base = torch.arange(
        0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    indices = indices + indices_base
    indices = indices.view(-1)

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[indices, :]
    feature = feature.view(batch_size, num_points, k, num_feats)
    x = x.view(batch_size, num_points, 1, num_feats).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)
    return feature


def main():
    name = 'graph_avt.transforms.affine.Affine'
    mod = import_class(name)
    s = {
        'translation': False
    }
    mod = mod(**s)
    print(mod)


if __name__ == '__main__':
    main()
