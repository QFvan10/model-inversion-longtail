from __future__ import print_function, division
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class FashionMNIST(Dataset):

    DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'Fashion-MNIST')
    SPLITS = (
        'victim_train', 'victim_val', 'victim_test',
        'attacker_train', 'attacker_val', 'attacker_test',
    )

    def __init__(self, root=None, split='victim_train', split_seed=1,
                 transform=None, target_transform=None):
        if split not in self.SPLITS:
            raise ValueError('unknown Fashion-MNIST split: {}'.format(split))
        if split_seed < 0:
            raise ValueError('Fashion-MNIST split_seed must be non-negative')
        self.root = os.path.abspath(os.path.expanduser(root or self.DEFAULT_ROOT))
        self.split = split
        self.split_seed = split_seed
        self.transform = transform
        self.target_transform = target_transform
        self.path = os.path.join(
            self.root, 'splits', 'seed_{}'.format(split_seed), split + '.npz')
        if not os.path.isfile(self.path):
            raise FileNotFoundError(
                'Fashion-MNIST partition not found: {}. Check --data-root and '
                '--split-seed; existing partitions are never regenerated automatically.'.format(
                    self.path))

        with np.load(self.path, allow_pickle=False) as partition:
            missing = {'images', 'labels', 'indices'} - set(partition.files)
            if missing:
                raise ValueError('Fashion-MNIST partition {} is missing {}'.format(
                    self.path, ', '.join(sorted(missing))))
            self.data = partition['images']
            self.labels = partition['labels']
            self.indices = partition['indices']

        if (self.data.ndim != 3 or self.data.shape[1:] != (28, 28)
                or self.data.dtype != np.uint8 or len(self.data) == 0):
            raise ValueError('Fashion-MNIST images must be nonempty uint8 [N, 28, 28]: {}'.format(
                self.path))
        if (self.labels.shape != (len(self.data),)
                or not np.issubdtype(self.labels.dtype, np.integer)
                or np.any((self.labels < 0) | (self.labels > 9))):
            raise ValueError('Fashion-MNIST labels must be integers [N] in 0..9: {}'.format(
                self.path))
        if (self.indices.shape != (len(self.data),)
                or not np.issubdtype(self.indices.dtype, np.integer)
                or np.any(self.indices < 0)):
            raise ValueError('Fashion-MNIST indices must be non-negative integers [N]: {}'.format(
                self.path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = Image.fromarray(self.data[index])
        target = int(self.labels[index])
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class FaceScrub(Dataset):
    def __init__(self, root, transform=None, target_transform=None, train=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        input = np.load(os.path.join(self.root, 'facescrub_color1.npz'))
        actor_images = input['actor_images']
        actor_labels = input['actor_labels']
        actress_images = input['actress_images']
        actress_labels = input['actress_labels']

        data = np.concatenate([actor_images, actress_images], axis=0)
        labels = np.concatenate([actor_labels, actress_labels], axis=0)

        v_min = data.min(axis=0)
        v_max = data.max(axis=0)
        data = (data - v_min) / np.where(v_max == v_min, 1, v_max - v_min)

        perm = np.random.RandomState(666).permutation(len(data))
        data = data[perm]
        labels = labels[perm]

        if train:
            self.data = data[0:int(0.8 * len(data))]
            self.labels = labels[0:int(0.8 * len(data))]
        else:
            self.data = data[int(0.8 * len(data)):]
            self.labels = labels[int(0.8 * len(data)):]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]

        if img.dtype == np.float64 or img.dtype == np.float32:
            img = (img * 255).astype(np.uint8)

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CelebA(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        data = np.load(os.path.join(self.root, 'celebA_crop_color.npz'))['celeba']


        v_min = data.min(axis=0)
        v_max = data.max(axis=0)
        data = (data - v_min) / np.where(v_max == v_min, 1, v_max - v_min)
        labels = np.array([0] * len(data))

        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]

        if img.dtype == np.float64 or img.dtype == np.float32:
            img = (img * 255).astype(np.uint8)

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
