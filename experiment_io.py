from datetime import datetime
from pathlib import Path
import os
import tempfile


def format_metrics(values):
    return ' | '.join('{}={}'.format(key, value) for key, value in values.items())


def write_report(path, values, append=False):
    def lines(mapping, indent=0):
        for key, value in mapping.items():
            prefix = ' ' * indent + str(key) + ':'
            if isinstance(value, dict):
                yield prefix
                yield from lines(value, indent + 2)
            else:
                yield '{} {}'.format(prefix, value)

    with open(path, 'a' if append else 'w', encoding='utf-8') as output:
        output.write('\n'.join(lines(values)) + '\n')
        if append:
            output.write('\n')


def classifier_directory(output_root, dataset, model):
    return Path(output_root).expanduser().resolve() / dataset / model


def create_classifier_directories(args):
    root = classifier_directory(args.output_root, args.dataset, args.model)
    profile = getattr(args, 'effective_training_profile', None) or args.training_profile
    run = (root / profile / 'seed{}'.format(args.seed)
           / datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    checkpoints = run / 'checkpoints'
    metrics = run / 'metrics'
    checkpoints.mkdir(parents=True, exist_ok=False)
    metrics.mkdir()
    if args.checkpoint_output:
        checkpoint = Path(args.checkpoint_output).expanduser().resolve()
    else:
        suffix = '' if profile == 'legacy' else '_{}'.format(profile)
        checkpoint = (Path(args.output_root).expanduser().resolve().parent
                      / 'classifier_{}{}.pth'.format(args.model, suffix))
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    return {'run': str(run), 'metrics': str(metrics),
            'checkpoint': str(checkpoint),
            'archive_checkpoint': str(checkpoints / 'best_classifier.pth')}


def publish_classifier_checkpoint(args, checkpoint):
    root = classifier_directory(args.output_root, args.dataset, args.model)
    root.mkdir(parents=True, exist_ok=True)
    relative = os.path.relpath(checkpoint, root)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=root,
                                     delete=False) as output:
        output.write(relative + '\n')
        temporary = output.name
    os.replace(temporary, root / 'latest.txt')


def latest_classifier_checkpoint(output_root, dataset, model):
    root = classifier_directory(output_root, dataset, model)
    pointer = root / 'latest.txt'
    if not pointer.is_file():
        return None
    relative = pointer.read_text(encoding='utf-8').strip()
    path = (root / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError('checkpoint indexed by {} does not exist: {}'.format(
            pointer, path))
    return str(path)
