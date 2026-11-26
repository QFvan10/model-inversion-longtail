from __future__ import print_function

import argparse
from datetime import datetime
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
from torchvision import transforms

from curated_chestxray_data import (
    CHESTXRAY_CLASSES, DEFAULT_ROOT as CURATED_DEFAULT_ROOT,
    build_chestxray_datasets, resolve_dataset_root,
)
from data import CelebA, FaceScrub, FashionMNIST
from experiment_io import latest_classifier_checkpoint, write_report
from metrics import psnr, ssim
from model import (
    MODEL_CHOICES,
    build_classifier,
    build_inversion,
    classifier_architecture_from_checkpoint,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

parser = argparse.ArgumentParser(description='Train a model inversion attack')
parser.add_argument('--batch-size', type=int, default=64, metavar='')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='')
parser.add_argument('--epochs', type=int, default=200, metavar='')
parser.add_argument('--lr', type=float, default=2e-4,
                    help='Adam learning rate of the inversion model')
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1, metavar='')
parser.add_argument('--log-interval', type=int, default=10, metavar='')
parser.add_argument('--nc', type=int, default=1)
parser.add_argument('--ndf', type=int, default=128)
parser.add_argument('--ngf', type=int, default=128)
parser.add_argument('--truncation', type=int, default=None,
                    help=('Top-k, default: all classes. Pass --truncation 4 to use '
                          'the k=4 setting of the paper; curated-chestxray range: '
                          '1..4, Fashion-MNIST: 1..10'))
parser.add_argument('--c', type=float, default=50.0,
                    help='additive offset of the exp-log and log transforms')
parser.add_argument('--gamma', type=float, default=0.3,
                    help='exp/log mixture weight of the proposed exp-log transform')
parser.add_argument('--transform', choices=['hybrid', 'log', 'power'],
                    default='hybrid',
                    help='hybrid (exp-log) for --method inversion; log/power for --method baseline')
parser.add_argument('--activation', choices=['tanh', 'prelu', 'gelu'],
                    default='tanh',
                    help='decoder activation; PReLU keeps the transposed-conv '
                         'topology but replaces Tanh')
parser.add_argument('--num-workers', '--num_workers', dest='num_workers',
                    type=int, default=8, metavar='')
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--min-epochs', type=int, default=0)
parser.add_argument('--lr-step-size', type=int, default=10,
                    help='StepLR period in epochs; raise it when attacker_train '
                         'is small so each LR phase keeps a similar number of '
                         'optimizer steps')
parser.add_argument('--model', type=str, default='cnn',
                    choices=MODEL_CHOICES)
parser.add_argument('--dataset', type=str, default='facescrub',
                    choices=['facescrub', 'fashion-mnist', 'curated-chestxray'])
parser.add_argument('--data-root', type=str, default=None,
                    help='dataset root; defaults to the selected dataset project directory')
parser.add_argument('--split-seed', type=int, default=1,
                    help='dataset partition seed; independent of training --seed')
parser.add_argument('--classifier-checkpoint', type=str, default=None,
                    help='explicit victim classifier checkpoint path')
parser.add_argument('--classifier-output-root', default='out/classifier_runs',
                    help='root containing completed classifier runs')
parser.add_argument('--output-root', type=str, default=None)
parser.add_argument('--method', type=str, default='inversion',
                    choices=['inversion', 'noise', 'dropping', 'baseline'],
                    help=(
                        'inversion=exp-log + tail_mixer/tail_gate residual '
                        'compensation; noise/dropping=raw top-k perturbation + '
                        'the same residual compensation; baseline=log/power + '
                        'the plain five-layer decoder'))
parser.add_argument('--class-balanced', action='store_true', default=False,
                    help='class-balanced sampling of attacker_train')
parser.add_argument('--noise-ratio', '--noise-l2-ratio',
                    dest='noise_l2_ratio', type=float, default=0.1,
                    help='relative L2 noise applied to each top-k vector')
parser.add_argument('--drop-rate', type=float, default=0.3,
                    help='probability of dropping each top-k entry')
parser.add_argument('--ssim-weight', type=float, default=0.0,
                    help='weight of 1-SSIM in the training objective')
parser.add_argument('--l1-weight', type=float, default=0.0,
                    help='weight of L1 reconstruction loss')
parser.add_argument('--class-variance-weight', type=float, default=0.0,
                    help='power p of within-class-variance loss weights; '
                         '0 disables the weighting')
parser.add_argument('--class-weight-eta', type=float, default=1e-3,
                    help='stabiliser of the class-variance weights')
parser.add_argument('--downscale-loss-size', type=int, default=0,
                    help='also optimise MSE at this resolution; 0 disables')
parser.add_argument('--downscale-loss-weight', type=float, default=0.0,
                    help='weight of the downscaled MSE term')
parser.add_argument('--selection-size', type=int, default=0,
                    help='validate at this resolution; 0 uses the native size')
parser.add_argument('--selection-metric', choices=['mse', 'mse_ssim'],
                    default='mse',
                    help='attacker validation criterion for checkpoint selection')
parser.add_argument('--selection-ssim-weight', type=float, default=0.1,
                    help='SSIM penalty weight when --selection-metric mse_ssim')
AUGMENTATION_CHOICES = (
    'none', 'standard', 'translate', 'affine-mild', 'intensity-mild',
    'combined-mild',
)
parser.add_argument('--inversion-augmentation', choices=AUGMENTATION_CHOICES,
                    default='standard',
                    help=('attacker_train augmentation profile; standard preserves '
                          'the original pipeline, none trains on clean images'))
parser.add_argument('--smoke-samples', type=int, default=0,
                    help='explicit smoke-only subset size; zero means full data')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_state(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def train(classifier, inversion, log_interval, device, data_loader,
          optimizer, epoch, ssim_weight=0.0, l1_weight=0.0,
          class_weights=None, downscale_size=0,
          downscale_weight=0.0):
    classifier.eval()
    inversion.train()
    core = inversion.module if isinstance(inversion, nn.DataParallel) else inversion
    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            prediction = classifier(data, release=True)

        if epoch == 1 and batch_idx == 0:
            topk_values, topk_indices = torch.topk(
                prediction[0], core.truncation)
            print('\nFirst batch, first sample Top-{} values:'.format(
                core.truncation))
            print(topk_values.detach().cpu().tolist())
            print('Corresponding class indices:')
            print(topk_indices.detach().cpu().tolist())

        reconstruction = inversion(prediction)
        if class_weights is not None:
            per_sample = (
                (reconstruction - data) ** 2).flatten(1).mean(dim=1)
            weights = class_weights.to(device)[target.to(device)]
            loss = (weights * per_sample).mean()
        else:
            loss = F.mse_loss(reconstruction, data)
        if downscale_size > 0 and downscale_weight > 0:
            small_prediction = F.interpolate(
                reconstruction, size=(downscale_size, downscale_size),
                mode='bilinear', align_corners=False, antialias=True)
            small_target = F.interpolate(
                data, size=(downscale_size, downscale_size),
                mode='bilinear', align_corners=False, antialias=True)
            loss = loss + float(downscale_weight) * F.mse_loss(
                small_prediction, small_target)
        if ssim_weight > 0.0:
            loss = loss + float(ssim_weight) * (
                1.0 - ssim(reconstruction, data, data_range=1.0))
        if l1_weight > 0.0:
            loss = loss + float(l1_weight) * F.l1_loss(
                reconstruction, data)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                loss.item()))


def save_reconstruction(data, reconstruction, path):
    count = min(32, data.size(0))
    comparison = torch.stack(
        (reconstruction[:count], data[:count]), dim=1)
    comparison = comparison.view(-1, data.size(1), 64, 64)
    vutils.save_image(comparison, path, nrow=8, normalize=False)


def evaluate(classifier, inversion, device, data_loader, image_path=None,
             calculate_metrics=False, calculate_ssim=False, resize=None):
    classifier.eval()
    inversion.eval()
    squared_error = 0.0
    psnr_total = 0.0
    ssim_total = 0.0
    batch_ssim_total = 0.0
    batch_ssim_count = 0
    sample_count = 0
    pixel_count = 0
    image_saved = False

    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            prediction = classifier(data, release=True)
            reconstruction = inversion(prediction)
            if resize:
                reconstruction_eval = F.interpolate(
                    reconstruction, size=(resize, resize), mode='bilinear',
                    align_corners=False, antialias=True)
                data_eval = F.interpolate(
                    data, size=(resize, resize), mode='bilinear',
                    align_corners=False, antialias=True)
            else:
                reconstruction_eval = reconstruction
                data_eval = data
            pixel_count += data_eval.numel()
            squared_error += F.mse_loss(
                reconstruction_eval, data_eval, reduction='sum').item()

            if calculate_ssim:
                batch_ssim_total += ssim(
                    reconstruction_eval, data_eval,
                    data_range=1.0).item() * data.size(0)
                batch_ssim_count += data.size(0)

            if image_path is not None and not image_saved:
                save_reconstruction(data, reconstruction, image_path)
                image_saved = True

            if calculate_metrics:
                for index in range(data.size(0)):
                    truth = data_eval[index:index + 1]
                    inverse = reconstruction_eval[index:index + 1]
                    psnr_total += psnr(truth, inverse, data_range=1.0).item()
                    ssim_total += ssim(truth, inverse, data_range=1.0).item()
                    sample_count += 1

    if pixel_count == 0:
        raise RuntimeError('cannot evaluate an empty dataset')
    mse = squared_error / pixel_count
    if calculate_metrics:
        return mse, psnr_total / sample_count, ssim_total / sample_count
    if calculate_ssim:
        return mse, batch_ssim_total / max(batch_ssim_count, 1)
    return mse


def _split_celeba(dataset, seed):
    size = len(dataset)
    train_size = int(0.8 * size)
    validation_size = int(0.1 * size)
    test_size = size - train_size - validation_size
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.random_split(
        dataset, [train_size, validation_size, test_size], generator=generator)


class ImageTransformDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, target = self.dataset[index]
        return self.transform(image), target


def inversion_transforms(augmentation='standard'):
    if augmentation not in AUGMENTATION_CHOICES:
        raise ValueError('unknown inversion augmentation: {}'.format(augmentation))
    base_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    train_operations = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
    ]
    if augmentation == 'none':
        train_operations.append(transforms.ToTensor())
        return base_transform, transforms.Compose(train_operations)
    if augmentation in ('standard', 'translate'):
        train_operations.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
        ])
        if augmentation == 'translate':
            train_operations.append(transforms.RandomAffine(
                degrees=0, translate=(0.03, 0.03)))
        train_operations.append(transforms.ColorJitter(contrast=0.2))
    elif augmentation in ('affine-mild', 'combined-mild'):
        train_operations.append(transforms.RandomAffine(
            degrees=5, translate=(0.03, 0.03), scale=(0.97, 1.03)))
        if augmentation == 'combined-mild':
            train_operations.append(transforms.ColorJitter(
                brightness=0.05, contrast=0.10))
    elif augmentation == 'intensity-mild':
        train_operations.append(transforms.ColorJitter(
            brightness=0.05, contrast=0.10))
    train_operations.append(transforms.ToTensor())
    return base_transform, transforms.Compose(train_operations)


def default_data_root(*candidates):
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return os.path.abspath(candidate)
    return os.path.abspath(candidates[0])


def resolve_base_data_root(root):
    if root:
        return os.path.abspath(os.path.expanduser(root))
    return default_data_root(
        os.path.join(SCRIPT_DIR, 'data'),
        os.path.join(PROJECT_DIR, 'data'),
        './data')


def build_datasets(args, base_transform, train_transform=None):
    train_transform = train_transform or base_transform
    if args.dataset == 'curated-chestxray':
        args.data_root = resolve_dataset_root(
            args.data_root or CURATED_DEFAULT_ROOT)
        datasets = build_chestxray_datasets(
            args.data_root, base_transform, base_transform, seed=args.split_seed)
        datasets['attacker_train'].transform = train_transform
        args.class_names = list(CHESTXRAY_CLASSES)
        args.split_fingerprint = datasets['attacker_train'].split_fingerprint
        return {
            split: datasets[split] for split in (
                'attacker_train', 'attacker_val', 'attacker_test',
                'victim_train', 'victim_test',
            )
        }, len(CHESTXRAY_CLASSES)
    if args.dataset == 'fashion-mnist':
        args.data_root = os.path.abspath(os.path.expanduser(
            args.data_root or FashionMNIST.DEFAULT_ROOT))
        if not os.path.isdir(args.data_root):
            args.data_root = default_data_root(
                os.path.join(SCRIPT_DIR, 'Fashion-MNIST'),
                os.path.join(PROJECT_DIR, 'Fashion-MNIST'),
                './Fashion-MNIST')
        return {
            split: FashionMNIST(
                args.data_root, split=split, split_seed=args.split_seed,
                transform=(train_transform if split == 'attacker_train'
                           else base_transform))
            for split in (
                'attacker_train', 'attacker_val', 'attacker_test',
                'victim_train', 'victim_test',
            )
        }, 10
    args.data_root = resolve_base_data_root(args.data_root)
    if os.path.isfile(os.path.join(args.data_root, 'facescrub_color1.npz')):
        args.data_root = os.path.dirname(args.data_root)
    celeba = CelebA(os.path.join(args.data_root, 'celebA'))
    attacker_train, attacker_val, attacker_test = _split_celeba(
        celeba, args.split_seed)
    datasets = {
        'attacker_train': ImageTransformDataset(attacker_train, train_transform),
        'attacker_val': ImageTransformDataset(attacker_val, base_transform),
        'attacker_test': ImageTransformDataset(attacker_test, base_transform),
        'victim_train': FaceScrub(
            os.path.join(args.data_root, 'facescrub'),
            transform=base_transform, train=True),
        'victim_test': FaceScrub(
            os.path.join(args.data_root, 'facescrub'),
            transform=base_transform, train=False),
    }
    return datasets, 530


def resolve_classifier_checkpoint(args):
    if args.classifier_checkpoint:
        path = os.path.abspath(os.path.expanduser(args.classifier_checkpoint))
        if not os.path.isfile(path):
            raise FileNotFoundError(
                'classifier checkpoint does not exist: {}'.format(path))
        return path

    latest = latest_classifier_checkpoint(
        args.classifier_output_root, args.dataset, args.model)
    if latest is not None:
        return latest
    if args.dataset in ('fashion-mnist', 'curated-chestxray'):
        raise FileNotFoundError(
            '{} classifier checkpoint not found for {} under {}. '
            'Train the matching classifier first or provide --classifier-checkpoint.'.format(
                args.dataset, args.model, os.path.abspath(args.classifier_output_root)))
    candidates = ['out/classifier_{}.pth'.format(args.model)]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(
        'classifier checkpoint not found; tried: {}'.format(
            ', '.join(os.path.abspath(path) for path in candidates)))


def validate_classifier_checkpoint(checkpoint, args, num_classes, path):
    if not isinstance(checkpoint, dict) or 'model' not in checkpoint:
        raise ValueError(
            'invalid classifier checkpoint (missing model state): {}'.format(path))

    config = checkpoint.get('config', {})
    if not isinstance(config, dict):
        config = {}
    stored_dataset = config.get('dataset', checkpoint.get('dataset'))
    if args.dataset == 'curated-chestxray':
        required = ('dataset', 'split_seed', 'nz', 'class_names', 'split_fingerprint')
        missing = [name for name in required if name not in config]
        if missing:
            raise ValueError('curated-chestxray checkpoint missing metadata: {}'.format(missing))
        if config['class_names'] != list(CHESTXRAY_CLASSES):
            raise ValueError(
                'classifier checkpoint class mapping does not match curated-chestxray')
        if config['split_fingerprint'] != getattr(args, 'split_fingerprint', None):
            raise ValueError('classifier checkpoint split fingerprint does not match current data')
        if checkpoint.get('task', 'multiclass') != 'multiclass':
            raise ValueError('curated-chestxray requires a multiclass classifier checkpoint')
    if stored_dataset is not None and stored_dataset != args.dataset:
        raise ValueError(
            'classifier checkpoint dataset {!r} does not match requested '
            'dataset {!r}: {}'.format(stored_dataset, args.dataset, path))

    if args.dataset in ('fashion-mnist', 'curated-chestxray') and 'split_seed' in config:
        if int(config['split_seed']) != args.split_seed:
            raise ValueError(
                'classifier checkpoint split_seed={} does not match requested {}: {}'.format(
                    config['split_seed'], args.split_seed, path))

    stored_model = config.get('model')
    if stored_model is not None and stored_model != args.model:
        raise ValueError('classifier checkpoint model {!r} does not match {!r}'.format(
            stored_model, args.model))
    for name in ('nc', 'ndf'):
        if name == 'ndf' and args.model != 'cnn':
            continue
        if name in config and int(config[name]) != getattr(args, name):
            raise ValueError('classifier checkpoint {}={} does not match requested {}'.format(
                name, config[name], getattr(args, name)))

    stored_classes = config.get(
        'nz', checkpoint.get('num_classes', checkpoint.get('nz')))
    if stored_classes is not None and int(stored_classes) != num_classes:
        raise ValueError(
            'classifier checkpoint has {} outputs, expected {}: {}'.format(
                stored_classes, num_classes, path))


def float_tag(value):
    return '{:.12g}'.format(float(value)).replace('-', 'm').replace('.', 'p')


def method_variant(args):
    if args.method == 'noise':
        return 'noise_residual_l2_{}'.format(float_tag(args.noise_l2_ratio))
    if args.method == 'dropping':
        return 'dropping_residual_p_{}'.format(float_tag(args.drop_rate))
    if args.method == 'baseline':
        if args.transform == 'log':
            return 'baseline_log_c_{}'.format(float_tag(args.c))
        return 'baseline_power'
    return 'inversion_hybrid_gamma_{}_c_{}_act_{}'.format(
        float_tag(args.gamma), float_tag(args.c), args.activation)


def create_run_directories(args, truncation):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_root = os.path.abspath(os.path.join(
        args.output_root,
        args.dataset,
        args.model,
        method_variant(args),
        'top{}'.format(truncation),
        'seed{}'.format(args.seed),
        timestamp,
    ))
    directories = {
        'run': run_root,
        'checkpoints': os.path.join(run_root, 'checkpoints'),
        'images': os.path.join(run_root, 'images'),
        'metrics': os.path.join(run_root, 'metrics'),
    }
    for path in directories.values():
        os.makedirs(path, exist_ok=False)
    return directories


def inversion_method_name(method):
    return 'entropy' if method == 'inversion' else method


def split_descriptions(dataset):
    if dataset == 'curated-chestxray':
        return {
            'attacker_test': 'Curated Chest X-ray attacker test partition',
            'victim_train': 'Curated Chest X-ray victim training partition (private reconstruction)',
            'victim_test': 'Curated Chest X-ray victim test partition',
        }
    if dataset == 'fashion-mnist':
        return {
            'attacker_test': 'Fashion-MNIST saved attacker test partition',
            'victim_train': 'Fashion-MNIST saved victim training partition (private reconstruction)',
            'victim_test': 'Fashion-MNIST saved victim test partition',
        }
    return {
        'attacker_test': 'CelebA test split (10% of CelebA)',
        'victim_train': 'FaceScrub training split (fixed 80%)',
        'victim_test': 'FaceScrub test split (fixed 20%)',
    }


def dataset_targets(dataset):
    if hasattr(dataset, 'labels'):
        return np.asarray(dataset.labels, dtype=np.int64)
    if hasattr(dataset, 'records'):
        return np.asarray(
            [int(label) for _, label in dataset.records], dtype=np.int64)
    if isinstance(dataset, torch.utils.data.Subset):
        base = dataset_targets(dataset.dataset)
        if base is not None:
            return base[np.asarray(dataset.indices, dtype=np.int64)]
    return None


def class_balanced_sampler(dataset):
    targets = dataset_targets(dataset)
    if targets is None:
        print('Class-balanced sampling unavailable; falling back to shuffle')
        return None
    counts = np.bincount(targets)
    if len(counts) == 0:
        return None
    weights = 1.0 / np.maximum(counts[targets], 1)
    print('Class-balanced sampling: {} classes, counts={}'.format(
        len(counts), counts.tolist()))
    return torch.utils.data.WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(targets), replacement=True)


def compute_class_variance_weights(dataset, num_classes, power, eta):
    if power <= 0:
        return None
    if not (hasattr(dataset, 'data') and hasattr(dataset, 'labels')):
        print('Class-variance weighting unavailable for this dataset; '
              'using uniform weights')
        return None
    images = torch.from_numpy(np.asarray(dataset.data)).float()
    labels = torch.from_numpy(np.asarray(dataset.labels)).long()
    if images.ndim == 3:
        images = images.unsqueeze(1)
    if images.max() > 1.5:
        images = images / 255.0
    if images.size(1) != 1:
        images = images.mean(dim=1, keepdim=True)
    variances = torch.zeros(num_classes)
    for class_id in range(num_classes):
        mask = labels == class_id
        if not mask.any():
            continue
        values = images[mask]
        variances[class_id] = (
            (values - values.mean(dim=0, keepdim=True)) ** 2).mean()
    weights = ((variances + eta) / (variances.mean() + eta)) ** power
    weights = weights / weights.mean()
    print('Class-variance weights (p={:g}): {}'.format(
        power, ['{:.3f}'.format(value) for value in weights.tolist()]))
    return weights


def main():
    args = parser.parse_args()
    started = time.monotonic()
    if args.smoke_samples < 0:
        parser.error('smoke-samples must be nonnegative')
    if args.method == 'inversion' and args.transform != 'hybrid':
        parser.error('selected inversion method requires --transform hybrid; '
                     'use --method baseline for log/power')
    if args.method == 'baseline' and args.transform not in ('log', 'power'):
        parser.error('--method baseline requires --transform log or power')
    if args.nc != 1:
        parser.error('--nc must be 1: data preprocessing uses grayscale images')
    if args.epochs < 1:
        raise ValueError('epochs must be at least 1')
    if args.lr_step_size < 1:
        raise ValueError('lr step size must be at least 1')
    if args.noise_l2_ratio < 0:
        raise ValueError('noise ratio must be non-negative')
    if not 0.0 <= args.drop_rate < 1.0:
        raise ValueError('drop rate must be in [0, 1)')
    if args.method == 'inversion' and not 0.0 <= args.gamma <= 1.0:
        raise ValueError('gamma must be in [0, 1]')
    if args.ssim_weight < 0 or args.l1_weight < 0:
        raise ValueError('SSIM and L1 weights must be non-negative')
    if args.class_variance_weight < 0 or args.class_weight_eta <= 0:
        raise ValueError('class variance weight must be non-negative and '
                         'eta must be positive')
    if args.selection_ssim_weight < 0:
        raise ValueError('selection SSIM weight must be non-negative')
    if args.output_root is None:
        args.output_root = 'out/inversion_runs'
    seed_everything(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    loader_kwargs = {
        'num_workers': args.num_workers,
        'pin_memory': use_cuda,
    }
    base_transform, train_transform = inversion_transforms(args.inversion_augmentation)
    datasets, num_classes = build_datasets(
        args, base_transform, train_transform)
    truncation = args.truncation if args.truncation is not None else num_classes
    if not 1 <= truncation <= num_classes:
        raise ValueError('truncation must be between 1 and {}'.format(num_classes))
    args.truncation = truncation

    class_weights = compute_class_variance_weights(
        datasets['attacker_train'], num_classes,
        args.class_variance_weight, args.class_weight_eta)

    if args.smoke_samples:
        datasets = {name: torch.utils.data.Subset(
            ds, range(min(len(ds), args.smoke_samples)))
            for name, ds in datasets.items()}

    train_sampler = (
        class_balanced_sampler(datasets['attacker_train'])
        if args.class_balanced else None)
    loaders = {}
    for name, dataset in datasets.items():
        sampler = train_sampler if name == 'attacker_train' else None
        loaders[name] = torch.utils.data.DataLoader(
            dataset,
            batch_size=(args.batch_size if name == 'attacker_train'
                        else args.test_batch_size),
            shuffle=(name == 'attacker_train' and sampler is None),
            sampler=sampler,
            **loader_kwargs)

    classifier_path = resolve_classifier_checkpoint(args)
    classifier_checkpoint = load_state(classifier_path, device)
    validate_classifier_checkpoint(
        classifier_checkpoint, args, num_classes, classifier_path)
    architecture_config = classifier_architecture_from_checkpoint(
        classifier_checkpoint)
    classifier_task = classifier_checkpoint.get('task', 'multiclass')
    classifier = nn.DataParallel(build_classifier(
        args.model,
        nc=args.nc,
        ndf=args.ndf,
        nz=num_classes,
        task=classifier_task,
        architecture_config=architecture_config,
    )).to(device)
    core = build_inversion(
        inversion_method_name(args.method),
        nc=args.nc, ngf=args.ngf, nz=num_classes,
        truncation=truncation, c=args.c,
        gamma=args.gamma, transform=args.transform,
        noise_l2_ratio=args.noise_l2_ratio,
        drop_rate=args.drop_rate,
        activation=args.activation).to(device)
    inversion = nn.DataParallel(core).to(device)

    classifier.load_state_dict(classifier_checkpoint['model'])
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)

    inversion_architecture = core.experiment_config()

    trainable_parameters = [
        parameter for parameter in inversion.parameters()
        if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError('inversion model has no trainable parameters')
    optimizer = optim.Adam(
        trainable_parameters, lr=args.lr, betas=(0.5, 0.999),
        amsgrad=True, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step_size, gamma=0.5)

    directories = create_run_directories(args, truncation)
    write_report(os.path.join(directories['run'], 'config.txt'),
                 dict(vars(args), resolved_classifier_checkpoint=classifier_path,
                      inversion_architecture=inversion_architecture))
    with open(os.path.join(directories['run'], 'cost.json'),
              'w', encoding='utf-8') as output:
        json.dump(dict(
            trainable_parameters=sum(p.numel() for p in inversion.parameters()
                                     if p.requires_grad),
            frozen_buffer_elements=sum(b.numel() for b in inversion.buffers()),
            preparation_seconds=time.monotonic() - started,
        ), output, indent=2)

    inversion_path = os.path.join(
        directories['checkpoints'], 'best_inversion.pth')
    best_image_path = os.path.join(
        directories['images'],
        'best_{}_victim_train.png'.format(args.dataset))
    best_val_image_path = os.path.join(
        directories['images'],
        'best_{}_attacker_val.png'.format(args.dataset))
    best_score = float('inf')
    best_validation_mse = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0

    print('Dataset: {}, victim: {}, method: {}, transform: {}, top-k: {}, device: {}'.format(
        args.dataset, args.model, args.method, args.transform, truncation, device))
    print('Classifier checkpoint: {}'.format(classifier_path))
    print('Inversion architecture: {}'.format(inversion_architecture))
    print('Run output: {}'.format(directories['run']))
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.monotonic()
        epoch_lr = optimizer.param_groups[0]['lr']
        train(classifier, inversion, args.log_interval, device,
              loaders['attacker_train'], optimizer, epoch,
              ssim_weight=args.ssim_weight, l1_weight=args.l1_weight,
              class_weights=class_weights,
              downscale_size=args.downscale_loss_size,
              downscale_weight=args.downscale_loss_weight)
        val_image_path = os.path.join(
            directories['images'],
            '{}_attacker_val_epoch_{:03d}.png'.format(args.dataset, epoch))
        if args.selection_metric == 'mse_ssim':
            validation_mse, validation_ssim = evaluate(
                classifier, inversion, device, loaders['attacker_val'],
                image_path=val_image_path, calculate_ssim=True,
                resize=args.selection_size or None)
            selection_score = validation_mse + args.selection_ssim_weight * (
                1.0 - validation_ssim)
        else:
            validation_mse = evaluate(
                classifier, inversion, device, loaders['attacker_val'],
                image_path=val_image_path,
                resize=args.selection_size or None)
            validation_ssim = None
            selection_score = validation_mse
        if not math.isfinite(selection_score):
            raise FloatingPointError('non-finite validation score')
        scheduler.step()
        history = {
            'epoch': epoch,
            'validation_mse': validation_mse,
            'selection_score': selection_score,
            'lr': epoch_lr,
            'seconds': time.monotonic() - epoch_start,
        }
        if validation_ssim is not None:
            history['validation_ssim'] = validation_ssim
        write_report(os.path.join(directories['metrics'], 'history.txt'),
                     history, append=True)

        if selection_score < best_score:
            best_score = selection_score
            best_validation_mse = validation_mse
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'model': inversion.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_score,
                'best_val_mse': validation_mse,
                'best_val_ssim': validation_ssim,
                'selection_metric': args.selection_metric,
                'method': args.method,
                'transform': args.transform,
                'dataset': args.dataset,
                'victim_model': args.model,
                'truncation': truncation,
                'classifier_checkpoint': classifier_path,
                'classifier_architecture': architecture_config,
                'inversion_architecture': inversion_architecture,
                'config': vars(args),
            }, inversion_path)
            print('[Epoch {}] best validation score: {:.6f} (MSE {:.6f})'.format(
                epoch, best_score, validation_mse))
        else:
            epochs_without_improvement += 1
            print('[Epoch {}] validation MSE: {:.6f} score: {:.6f} ({}/{})'.format(
                epoch, validation_mse, selection_score,
                epochs_without_improvement, args.patience))
            if (epoch >= args.min_epochs and args.patience > 0
                    and epochs_without_improvement >= args.patience):
                print('Early stopping')
                break

    best_checkpoint = load_state(inversion_path, device)
    inversion.load_state_dict(best_checkpoint['model'])
    evaluate(
        classifier, inversion, device, loaders['attacker_val'],
        image_path=best_val_image_path)
    evaluation_splits = ('attacker_test', 'victim_train', 'victim_test')
    descriptions = split_descriptions(args.dataset)
    results = {}
    for split in evaluation_splits:
        image_path = os.path.join(
            directories['images'],
            'best_{}_{}.png'.format(args.dataset, split))
        results[split] = evaluate(
            classifier, inversion, device, loaders[split],
            image_path=image_path, calculate_metrics=True)

    result_path = os.path.join(directories['metrics'], 'results.txt')
    with open(result_path, 'w', encoding='utf-8') as output:
        output.write('run directory: {}\n'.format(directories['run']))
        output.write('dataset: {}\n'.format(args.dataset))
        output.write('victim model: {}\n'.format(args.model))
        output.write('method: {}\n'.format(args.method))
        output.write('top-k: {}\n'.format(truncation))
        output.write('seed: {}\n'.format(args.seed))
        output.write('classifier checkpoint: {}\n'.format(classifier_path))
        output.write('best epoch: {}\n'.format(best_epoch))
        output.write('best attacker validation MSE: {:.8f}\n'.format(
            best_validation_mse))
        output.write('best attacker validation selection score: {:.8f}\n'.format(
            best_score))
        output.write('saved reconstruction: {}\n'.format(best_image_path))
        output.write('\nactive inversion architecture:\n')
        for name, value in inversion_architecture.items():
            output.write('  {}: {}\n'.format(name, value))
        output.write('\nrun parameters:\n')
        for name, value in sorted(vars(args).items()):
            output.write('  {}: {}\n'.format(name, value))
        output.write('\nfinal metrics:\n')
        for split in evaluation_splits:
            mse_value, psnr_value, ssim_value = results[split]
            line = '{} [{}]: MSE={:.8f}, PSNR={:.4f}, SSIM={:.6f}'.format(
                split, descriptions[split], mse_value, psnr_value, ssim_value)
            print(line)
            output.write(line + '\n')
    with open(os.path.join(directories['metrics'], 'results.json'),
              'w', encoding='utf-8') as output:
        json.dump(dict(
            best_epoch=best_epoch, best_val_mse=best_validation_mse,
            best_selection_score=best_score,
            selection_metric=args.selection_metric,
            metrics={key: dict(zip(('mse', 'psnr', 'ssim'), value))
                     for key, value in results.items()},
            sample_counts={key: len(loaders[key].dataset) for key in loaders},
            inversion_architecture=inversion_architecture,
            checkpoint=inversion_path, config=vars(args),
            classifier_checkpoint=classifier_path,
            elapsed_seconds=time.monotonic() - started,
        ), output, indent=2)
    print('Checkpoint saved to {}'.format(inversion_path))
    print('Results saved to {}'.format(result_path))


if __name__ == '__main__':
    main()
