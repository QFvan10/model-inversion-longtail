import csv
import hashlib
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

CHESTXRAY_CLASSES = (
    'COVID-19', 'Normal', 'Pneumonia-Bacterial', 'Pneumonia-Viral',
)
SPLITS = tuple(role + '_' + part
               for role in ('victim', 'attacker')
               for part in ('train', 'val', 'test'))
SPLIT_RATIOS = tuple(zip(SPLITS, (0.40, 0.05, 0.05, 0.40, 0.05, 0.05)))
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')
DEFAULT_ROOT = str(Path(__file__).resolve().parent / 'data' / 'Curated X-Ray Dataset')


def resolve_dataset_root(root):
    root = Path(root).expanduser().resolve()
    for candidate in (root, root / 'Curated X-Ray Dataset',
                      root / 'extracted' / 'Curated X-Ray Dataset'):
        if all((candidate / name).is_dir() for name in CHESTXRAY_CLASSES):
            return str(candidate)
    raise FileNotFoundError(
        'Cannot find the four-class Chest X-ray folders below {}'.format(root))


def _split_counts(size):
    counts = [size * weight // 10 for weight in (8, 1, 1)]
    remainders = [size * weight % 10 for weight in (8, 1, 1)]
    order = sorted(range(3), key=lambda index: -remainders[index])
    for index in order[:size - sum(counts)]:
        counts[index] += 1
    return counts


def make_split_records(root, seed=1):
    if not isinstance(seed, int) or not 0 <= seed < 2 ** 32:
        raise ValueError('split seed must be an integer in [0, 2**32)')
    dataset_root = Path(resolve_dataset_root(root))
    samples = []
    for label, class_name in enumerate(CHESTXRAY_CLASSES):
        class_root = dataset_root / class_name
        if not class_root.resolve().is_relative_to(dataset_root):
            raise ValueError('class directory escapes dataset root: {}'.format(class_root))
        paths = [path for path in sorted(class_root.iterdir())
                 if path.suffix.lower() in IMAGE_EXTENSIONS]
        if not paths:
            raise RuntimeError('Empty class directory: {}'.format(class_root))
        for path in paths:
            if not path.is_file() or not path.resolve().is_relative_to(dataset_root):
                raise ValueError('invalid image path: {}'.format(path))
            samples.append((str(path), label))
    if len(samples) % 2:
        raise ValueError('equal victim/attacker pools require an even image count')
    labels = [label for _, label in samples]
    indices = list(range(len(samples)))
    victim, attacker = train_test_split(
        indices, train_size=len(samples) // 2, stratify=labels, random_state=seed)
    records = {}
    for role, pool in (('victim', victim), ('attacker', attacker)):
        n_train, n_val, n_test = _split_counts(len(pool))
        if min(n_train, n_val, n_test) < len(CHESTXRAY_CLASSES):
            raise ValueError('not enough images for all classes in every split')
        train, held_out = train_test_split(
            pool, train_size=n_train, stratify=[labels[i] for i in pool],
            random_state=seed)
        val, test = train_test_split(
            held_out, train_size=n_val,
            stratify=[labels[i] for i in held_out], random_state=seed)
        for part, selected in (('train', train), ('val', val), ('test', test)):
            records[role + '_' + part] = [samples[i] for i in selected]
    validate_split_records(records)
    return records


def validate_split_records(records):
    if set(records) != set(SPLITS):
        raise ValueError('expected all six privacy splits')
    seen = set()
    for split in SPLITS:
        if not records[split]:
            raise ValueError('empty split: {}'.format(split))
        for path, label in records[split]:
            key = str(Path(path).resolve())
            if key in seen:
                raise ValueError('image appears more than once: {}'.format(path))
            if not 0 <= label < len(CHESTXRAY_CLASSES):
                raise ValueError('invalid class label: {}'.format(label))
            seen.add(key)
    sizes = [sum(len(records[role + '_' + p]) for p in ('train', 'val', 'test'))
             for role in ('victim', 'attacker')]
    if sizes[0] != sizes[1]:
        raise ValueError('victim and attacker must have equal image counts')
    for role, size in zip(('victim', 'attacker'), sizes):
        actual = [len(records[role + '_' + p]) for p in ('train', 'val', 'test')]
        if actual != _split_counts(size):
            raise ValueError('incorrect split counts for {}'.format(role))


def split_fingerprint(root, records):
    root = Path(resolve_dataset_root(root))
    digest = hashlib.sha256()
    for split in SPLITS:
        for path, label in sorted(records[split]):
            relative = Path(path).relative_to(root).as_posix()
            digest.update(('{}\t{}\t{}\n'.format(relative, label, split)).encode('utf-8'))
    return digest.hexdigest()


def export_split_audit(root, output_dir, seed=1):
    root = Path(resolve_dataset_root(root))
    output_dir = Path(output_dir).expanduser().resolve()
    records = make_split_records(root, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'curated_chestxray_seed_{}.csv'.format(seed)
    report_path = output_dir / 'curated_chestxray_seed_{}_stats.txt'.format(seed)
    with csv_path.open('w', encoding='utf-8', newline='') as output:
        writer = csv.writer(output)
        writer.writerow(('relative_path', 'label', 'class_name', 'split'))
        for split in SPLITS:
            for path, label in records[split]:
                writer.writerow((Path(path).relative_to(root).as_posix(),
                                 label, CHESTXRAY_CLASSES[label], split))
    lines = ['dataset: curated-chestxray', 'split_seed: {}'.format(seed),
             'unit: image (no patient identity metadata)',
             'fingerprint: ' + split_fingerprint(root, records),
             'total_images: {}'.format(sum(map(len, records.values()))),
             'pairwise_path_overlap: 0']
    for split in SPLITS:
        counts = Counter(label for _, label in records[split])
        lines.append('{}: total={}, classes={}'.format(
            split, len(records[split]),
            ', '.join('{}={}'.format(name, counts[i])
                      for i, name in enumerate(CHESTXRAY_CLASSES))))
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return csv_path, report_path


class CuratedChestXrayDataset(Dataset):
    def __init__(self, records, transform=None, split=None, fingerprint=None):
        self.records = list(records)
        self.transform = transform
        self.split = split
        self.split_fingerprint = fingerprint

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, label = self.records[index]
        with Image.open(path) as source:
            image = source.convert('L')
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


def build_chestxray_datasets(root, train_transform, test_transform, seed=1):
    records = make_split_records(root, seed)
    fingerprint = split_fingerprint(root, records)
    return {
        split: CuratedChestXrayDataset(
            rows, train_transform if split.endswith('_train') else test_transform,
            split=split, fingerprint=fingerprint)
        for split, rows in records.items()
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='export the curated Chest X-ray split audit files')
    parser.add_argument('--data-root', default=DEFAULT_ROOT)
    parser.add_argument('--split-seed', type=int, default=1)
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()
    root = resolve_dataset_root(args.data_root)
    paths = export_split_audit(
        root, args.output_dir or str(Path(root).parent / 'curated_splits'),
        args.split_seed)
    print('\n'.join(map(str, paths)))
