from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


MODEL_CHOICES = ('cnn', 'resnet18', 'resnet34',
                 'vgg16', 'mobilenetv2', 'mobilenetv3')

INVERSION_METHODS = ('inversion', 'noise', 'dropping', 'baseline')
BASELINE_TRANSFORMS = ('log', 'power')

INPUT_SIZE = 64


def _classifier_output(logits, release, task):
    if task == 'multilabel':
        return torch.sigmoid(logits) if release else logits
    if task == 'multiclass':
        return F.softmax(logits, dim=1) if release else F.log_softmax(logits, dim=1)
    raise ValueError('unknown classifier task: {}'.format(task))


def _init_conv(module):
    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def _init_linear(module):
    nn.init.normal_(module.weight, 0, 0.01)
    nn.init.zeros_(module.bias)


def _init_small(module, std=0.01):
    nn.init.normal_(module.weight, 0.0, float(std))
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class Classifier(nn.Module):

    def __init__(self, nc, ndf, nz, task='multiclass'):
        super(Classifier, self).__init__()

        self.nc = nc
        self.ndf = ndf
        self.nz = nz
        self.task = task

        self.encoder = nn.Sequential(
            nn.Conv2d(nc, ndf, 3, 1, 1),
            nn.BatchNorm2d(ndf),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            nn.Conv2d(ndf, ndf * 2, 3, 1, 1),
            nn.BatchNorm2d(ndf * 2),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1),
            nn.BatchNorm2d(ndf * 4),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1),
            nn.BatchNorm2d(ndf * 8),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
        )

        self.fc = nn.Sequential(
            nn.Linear(ndf * 8 * 4 * 4, nz * 5),
            nn.Dropout(0.5),
            nn.Linear(nz * 5, nz),
        )

    def forward(self, x, release=False):
        x = x.view(-1, self.nc, INPUT_SIZE, INPUT_SIZE)
        x = self.encoder(x)
        x = x.view(-1, self.ndf * 8 * 4 * 4)
        x = self.fc(x)
        return _classifier_output(x, release, self.task)


class _BackboneClassifier(nn.Module):

    def __init__(self, nc, nz, task='multiclass', normalize_input=False,
                 input_mean=0.5, input_std=0.5):
        super(_BackboneClassifier, self).__init__()
        if input_std <= 0:
            raise ValueError('input_std must be positive')
        self.nc = nc
        self.nz = nz
        self.task = task
        self.normalize_input = bool(normalize_input)
        self.input_mean = float(input_mean)
        self.input_std = float(input_std)
        self.model = None

    def forward(self, x, release=False):
        x = x.view(-1, self.nc, INPUT_SIZE, INPUT_SIZE)
        if self.normalize_input:
            x = (x - self.input_mean) / self.input_std
        logits = self.model(x)
        return _classifier_output(logits, release, self.task)


class _ResNetClassifier(_BackboneClassifier):

    architecture = None

    def __init__(self, nc, nz, task='multiclass', small_input_stem=False,
                 normalize_input=False, input_mean=0.5, input_std=0.5):
        super(_ResNetClassifier, self).__init__(
            nc, nz, task=task, normalize_input=normalize_input,
            input_mean=input_mean, input_std=input_std)
        self.small_input_stem = bool(small_input_stem)
        self.model = self.architecture(weights=None)
        if self.small_input_stem:
            self.model.conv1 = nn.Conv2d(
                nc, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.model.maxpool = nn.Identity()
        else:
            self.model.conv1 = nn.Conv2d(
                nc, 64, kernel_size=7, stride=2, padding=3, bias=False)
        _init_conv(self.model.conv1)
        self.model.fc = nn.Linear(self.model.fc.in_features, nz)
        _init_linear(self.model.fc)


class ResNet18Classifier(_ResNetClassifier):
    architecture = staticmethod(models.resnet18)


class ResNet34Classifier(_ResNetClassifier):
    architecture = staticmethod(models.resnet34)


class VGG16Classifier(_BackboneClassifier):

    def __init__(self, nc, nz, task='multiclass', small_input_stem=False,
                 normalize_input=False, input_mean=0.5, input_std=0.5):
        super(VGG16Classifier, self).__init__(
            nc, nz, task=task, normalize_input=normalize_input,
            input_mean=input_mean, input_std=input_std)
        self.small_input_stem = bool(small_input_stem)
        layers = []
        channels = nc
        for value in (64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
                      512, 512, 512, 'M', 512, 512, 512, 'M'):
            if value == 'M':
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers.extend([nn.Conv2d(channels, value, 3, padding=1),
                               nn.ReLU(inplace=True)])
                channels = value
        self.model = nn.Sequential(
            nn.Sequential(*layers),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, nz),
        )
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                _init_conv(module)
            elif isinstance(module, nn.Linear):
                _init_linear(module)


def _set_head_dropout(module, probability):
    probability = float(probability)
    if not 0.0 <= probability < 1.0:
        raise ValueError('head_dropout must be in [0, 1)')
    replaced = 0
    for child in module.modules():
        if isinstance(child, nn.Dropout):
            child.p = probability
            replaced += 1
    if replaced == 0:
        raise ValueError('classifier head has no Dropout layer to configure')
    return replaced


class _MobileNetClassifier(_BackboneClassifier):

    architecture = None

    def __init__(self, nc, nz, task='multiclass', small_input_stem=False,
                 normalize_input=False, input_mean=0.5, input_std=0.5,
                 head_dropout=None):
        super(_MobileNetClassifier, self).__init__(
            nc, nz, task=task, normalize_input=normalize_input,
            input_mean=input_mean, input_std=input_std)
        self.small_input_stem = bool(small_input_stem)
        self.head_dropout = None if head_dropout is None else float(head_dropout)
        self.model = self.architecture(weights=None, num_classes=nz)
        original = self.model.features[0][0]
        self.model.features[0][0] = nn.Conv2d(
            nc, original.out_channels, original.kernel_size,
            stride=1 if self.small_input_stem else original.stride,
            padding=original.padding, bias=False)
        _init_conv(self.model.features[0][0])
        if self.head_dropout is not None:
            _set_head_dropout(self.model.classifier, self.head_dropout)


class MobileNetV2Classifier(_MobileNetClassifier):
    architecture = staticmethod(models.mobilenet_v2)


class MobileNetV3Classifier(_MobileNetClassifier):
    architecture = staticmethod(models.mobilenet_v3_large)


def _decoder_activation(name, channels):
    if name in (None, 'tanh'):
        return nn.Tanh()
    if name == 'prelu':
        return nn.PReLU(channels)
    if name == 'gelu':
        return nn.GELU()
    raise ValueError('unknown decoder activation: {}'.format(name))


def build_plain_decoder(nc, ngf, nz, activation='tanh'):
    return nn.Sequential(
        nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0),
        nn.BatchNorm2d(ngf * 8),
        _decoder_activation(activation, ngf * 8),
        nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
        nn.BatchNorm2d(ngf * 4),
        _decoder_activation(activation, ngf * 4),
        nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
        nn.BatchNorm2d(ngf * 2),
        _decoder_activation(activation, ngf * 2),
        nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
        nn.BatchNorm2d(ngf),
        _decoder_activation(activation, ngf),
        nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
        nn.Sigmoid(),
    )


class ProposedInversion(nn.Module):

    method_name = 'proposed'
    uses_residual_compensation = True

    def __init__(self, nc, ngf, nz, truncation, activation='tanh'):
        super(ProposedInversion, self).__init__()
        if not 1 <= truncation <= nz:
            raise ValueError('truncation must satisfy 1 <= truncation <= nz')

        self.nc = nc
        self.ngf = ngf
        self.nz = nz
        self.truncation = truncation
        self.c = 0.0
        self.activation = activation

        self.tail_mixer = nn.Sequential(
            nn.LayerNorm(self.nz),
            nn.Linear(self.nz, self.nz),
            nn.GELU(),
            nn.Linear(self.nz, self.nz),
        )
        self.tail_gate = nn.Sequential(
            nn.LayerNorm(self.nz),
            nn.Linear(self.nz, self.nz),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.tail_mixer[-1].weight)
        nn.init.zeros_(self.tail_mixer[-1].bias)
        nn.init.zeros_(self.tail_gate[1].weight)
        nn.init.zeros_(self.tail_gate[1].bias)

        self.decoder = build_plain_decoder(
            nc, ngf, nz, activation=activation)

    def _validate_prediction(self, prediction):
        if prediction.ndim != 2 or prediction.size(1) != self.nz:
            raise ValueError(
                'prediction must have shape [batch, {}], got {}'.format(
                    self.nz, tuple(prediction.shape)))

    @staticmethod
    def _scatter_like(reference, indices, values):
        return torch.zeros_like(reference).scatter(1, indices, values)

    @staticmethod
    def _shift_nonnegative(values):
        row_min = values.min(dim=1, keepdim=True)[0]
        return values + F.relu(-row_min)

    def _transform_topk(self, topk):
        raise NotImplementedError

    def get_representation_stages(self, prediction):
        self._validate_prediction(prediction)
        topk, indices = torch.topk(prediction, self.truncation, dim=1)
        raw = self._scatter_like(prediction, indices, topk)

        processed_topk = self._transform_topk(topk)
        processed = self._scatter_like(prediction, indices, processed_topk)

        residual = self.tail_mixer(processed)
        gate = self.tail_gate(processed)
        post_mixer = processed + residual * gate
        return {
            'topk_values': topk,
            'topk_indices': indices,
            'raw': raw,
            'processed_topk_values': processed_topk,
            'processed': processed,
            'residual': residual,
            'gate': gate,
            'post_mixer': post_mixer,
            'decoder_input': post_mixer,
        }

    def get_representations(self, prediction):
        stages = self.get_representation_stages(prediction)
        return stages['raw'], stages['post_mixer']

    def decode_representation(self, representation):
        reconstruction = self.decoder(representation.view(-1, self.nz, 1, 1))
        return reconstruction.view(-1, self.nc, INPUT_SIZE, INPUT_SIZE)

    def forward(self, prediction):
        _, post_mixer = self.get_representations(prediction)
        return self.decode_representation(post_mixer)

    def experiment_config(self):
        return {
            'inversion_method': self.method_name,
            'truncation': self.truncation,
            'c': self.c,
            'decoder': 'plain',
            'uses_residual_compensation': True,
            'uses_tail_gate': True,
            'uses_channel_attention': False,
            'architecture_version': 4,
            'activation': self.activation,
        }


class ExpLogInversion(ProposedInversion):

    method_name = 'inversion'

    def __init__(self, nc, ngf, nz, truncation, c=50.0, gamma=0.3,
                 epsilon=1e-38, activation='tanh'):
        if not 0.0 <= gamma <= 1.0:
            raise ValueError('gamma must be in [0, 1]')
        super(ExpLogInversion, self).__init__(
            nc, ngf, nz, truncation, activation=activation)
        self.c = float(c)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)

    def _transform_topk(self, topk):
        transformed = self.gamma * torch.exp(topk)
        transformed = transformed + (1.0 - self.gamma) * torch.log(
            topk + self.epsilon)
        transformed = transformed + self.c
        return self._shift_nonnegative(transformed)

    def experiment_config(self):
        config = super(ExpLogInversion, self).experiment_config()
        config['transform'] = 'exp_log'
        config['gamma'] = self.gamma
        config['epsilon'] = self.epsilon
        return config


class AddingNoiseInversion(ProposedInversion):

    method_name = 'noise'

    def __init__(self, nc, ngf, nz, truncation, noise_l2_ratio=0.1):
        if noise_l2_ratio is None or noise_l2_ratio < 0.0:
            raise ValueError('noise_l2_ratio must be a non-negative number')
        super(AddingNoiseInversion, self).__init__(nc, ngf, nz, truncation)
        self.noise_l2_ratio = float(noise_l2_ratio)

    def sample_scaled_noise(self, topk):
        noise = torch.randn_like(topk)
        noise_norm = noise.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        signal_norm = topk.norm(p=2, dim=1, keepdim=True)
        return noise * (self.noise_l2_ratio * signal_norm / noise_norm)

    def _transform_topk(self, topk):
        perturbed = torch.clamp(topk + self.sample_scaled_noise(topk), min=0.0)

        empty = perturbed.sum(dim=1) == 0
        if empty.any():
            fallback_index = torch.randint(
                0, topk.size(1), (topk.size(0), 1), device=topk.device)
            fallback_value = torch.full(
                fallback_index.shape, 1e-12,
                device=topk.device, dtype=topk.dtype)
            fallback = torch.zeros_like(perturbed).scatter(
                1, fallback_index, fallback_value)
            perturbed = torch.where(empty.unsqueeze(1), fallback, perturbed)
        return perturbed

    def experiment_config(self):
        config = super(AddingNoiseInversion, self).experiment_config()
        config.pop('c')
        config['transform'] = 'noise'
        config['noise_l2_ratio'] = self.noise_l2_ratio
        return config


class DroppingInversion(ProposedInversion):

    method_name = 'dropping'

    def __init__(self, nc, ngf, nz, truncation, drop_rate=0.3):
        if drop_rate is None or not 0.0 <= drop_rate < 1.0:
            raise ValueError('drop_rate must be in [0, 1)')
        super(DroppingInversion, self).__init__(nc, ngf, nz, truncation)
        self.drop_rate = float(drop_rate)

    def sample_nonempty_mask(self, topk):
        if self.drop_rate == 0.0:
            return torch.ones_like(topk)

        mask = torch.rand_like(topk) >= self.drop_rate
        empty = ~mask.any(dim=1)
        attempts = 0
        while empty.any() and attempts < 32:
            mask[empty] = torch.rand_like(topk[empty]) >= self.drop_rate
            empty = ~mask.any(dim=1)
            attempts += 1

        if empty.any():
            rows = empty.nonzero(as_tuple=False).squeeze(1)
            columns = torch.randint(
                0, topk.size(1), (rows.numel(),), device=topk.device)
            mask[rows, columns] = True
        return mask.to(dtype=topk.dtype)

    def _transform_topk(self, topk):
        return topk * self.sample_nonempty_mask(topk)

    def experiment_config(self):
        config = super(DroppingInversion, self).experiment_config()
        config.pop('c')
        config['transform'] = 'dropping'
        config['drop_rate'] = self.drop_rate
        config['k1_noop'] = self.truncation == 1
        return config


class BaselineInversion(nn.Module):

    method_name = 'baseline'
    uses_residual_compensation = False
    uses_channel_attention = False

    def __init__(self, nc, ngf, nz, truncation, c=0.0):
        super(BaselineInversion, self).__init__()
        if not 1 <= truncation <= nz:
            raise ValueError('truncation must satisfy 1 <= truncation <= nz')
        self.nc = nc
        self.ngf = ngf
        self.nz = nz
        self.truncation = truncation
        self.c = float(c)
        self.decoder = build_plain_decoder(nc, ngf, nz)

    def _transform_topk(self, topk):
        raise NotImplementedError

    @staticmethod
    def _shift_nonnegative(values):
        row_min = values.min(dim=1, keepdim=True)[0]
        return values + F.relu(-row_min)

    def get_representation_stages(self, prediction):
        if prediction.ndim != 2 or prediction.size(1) != self.nz:
            raise ValueError(
                'prediction must have shape [batch, {}]'.format(self.nz))
        topk, indices = torch.topk(prediction, self.truncation, dim=1)
        raw = torch.zeros_like(prediction).scatter(1, indices, topk)
        processed_topk = self._transform_topk(topk)
        processed = torch.zeros_like(prediction).scatter(
            1, indices, processed_topk)
        return {
            'topk_values': topk,
            'topk_indices': indices,
            'raw': raw,
            'processed_topk_values': processed_topk,
            'processed': processed,
            'decoder_input': processed,
        }

    def get_representations(self, prediction):
        stages = self.get_representation_stages(prediction)
        return stages['raw'], stages['decoder_input']

    def decode_representation(self, representation):
        reconstruction = self.decoder(representation.view(-1, self.nz, 1, 1))
        return reconstruction.view(-1, self.nc, INPUT_SIZE, INPUT_SIZE)

    def forward(self, prediction):
        _, decoder_input = self.get_representations(prediction)
        return self.decode_representation(decoder_input)

    def experiment_config(self):
        return {
            'inversion_method': self.method_name,
            'truncation': self.truncation,
            'c': self.c,
            'decoder': 'original',
            'uses_residual_compensation': False,
            'uses_tail_gate': False,
            'uses_channel_attention': False,
            'architecture_version': 4,
        }


class LogBaselineInversion(BaselineInversion):

    def __init__(self, nc, ngf, nz, truncation, c=50.0):
        super(LogBaselineInversion, self).__init__(
            nc, ngf, nz, truncation, c=c)

    def _transform_topk(self, topk):
        transformed = torch.clamp(torch.log(topk), min=-1000) + self.c
        return self._shift_nonnegative(transformed)

    def experiment_config(self):
        config = super(LogBaselineInversion, self).experiment_config()
        config.update(transform='log')
        return config


class PowerBaselineInversion(BaselineInversion):

    method_name = 'baseline'

    def __init__(self, nc, ngf, nz, truncation, c=0.0, power=1.0 / 11.0):
        super(PowerBaselineInversion, self).__init__(
            nc, ngf, nz, truncation, c=c)
        self.power = float(power)

    def _transform_topk(self, topk):
        transformed = torch.pow(topk, self.power)
        if self.c:
            transformed = transformed + self.c
        return self._shift_nonnegative(transformed)

    def experiment_config(self):
        config = super(PowerBaselineInversion, self).experiment_config()
        config.update(transform='power', power=self.power)
        return config


def classifier_architecture_from_checkpoint(checkpoint):
    config = checkpoint.get('config', {}) if isinstance(checkpoint, dict) else {}
    if not isinstance(config, dict):
        config = {}
    architecture = config.get('classifier_architecture', {})
    return dict(architecture) if isinstance(architecture, dict) else {}


def _default_classifier_architecture(model_name):
    if model_name != 'cnn':
        return {
            'small_input_stem': False,
            'normalize_input': False,
            'input_mean': 0.5,
            'input_std': 0.5,
            'head_dropout': None,
        }
    return {}


def build_classifier(
    model_name, nc, nz, ndf=128, task='multiclass',
    architecture_config=None,
):
    architecture_config = dict(
        _default_classifier_architecture(model_name)
        if architecture_config is None else architecture_config
    )
    normalize_input = bool(architecture_config.get('normalize_input', False))
    input_mean = float(architecture_config.get('input_mean', 0.5))
    input_std = float(architecture_config.get('input_std', 0.5))
    small_input_stem = bool(architecture_config.get('small_input_stem', False))
    head_dropout = architecture_config.get('head_dropout')
    head_dropout = None if head_dropout is None else float(head_dropout)
    common = dict(
        nc=nc, nz=nz, task=task,
        normalize_input=normalize_input,
        input_mean=input_mean, input_std=input_std,
    )

    if model_name == 'cnn':
        return Classifier(nc=nc, ndf=ndf, nz=nz, task=task)

    if model_name == 'resnet18':
        return ResNet18Classifier(small_input_stem=small_input_stem, **common)

    if model_name == 'resnet34':
        return ResNet34Classifier(small_input_stem=small_input_stem, **common)

    if model_name == 'vgg16':
        return VGG16Classifier(small_input_stem=small_input_stem, **common)

    if model_name == 'mobilenetv2':
        return MobileNetV2Classifier(
            small_input_stem=small_input_stem, head_dropout=head_dropout,
            **common)

    if model_name == 'mobilenetv3':
        return MobileNetV3Classifier(
            small_input_stem=small_input_stem, head_dropout=head_dropout,
            **common)

    raise ValueError('unknown classifier model: {}'.format(model_name))


def build_inversion(
    method,
    nc,
    ngf,
    nz,
    truncation,
    c=50.0,
    gamma=0.3,
    noise_l2_ratio=None,
    drop_rate=None,
    transform='hybrid',
    activation='tanh',
):
    if method in ('inversion', 'entropy'):
        if transform not in (None, 'hybrid', 'exp_log'):
            raise ValueError(
                'method=inversion requires transform=hybrid (exp-log); '
                'use method=baseline for log/power')
        return ExpLogInversion(
            nc, ngf, nz, truncation, c=c, gamma=gamma,
            activation=activation)

    if method == 'noise':
        return AddingNoiseInversion(
            nc, ngf, nz, truncation, noise_l2_ratio=noise_l2_ratio)

    if method == 'dropping':
        return DroppingInversion(
            nc, ngf, nz, truncation, drop_rate=drop_rate)

    if method == 'baseline':
        if transform == 'log':
            return LogBaselineInversion(nc, ngf, nz, truncation, c=c)
        if transform == 'power':
            return PowerBaselineInversion(nc, ngf, nz, truncation, c=0.0)
        raise ValueError(
            'method=baseline requires transform=log or transform=power')

    raise ValueError('unknown inversion method: {}'.format(method))
