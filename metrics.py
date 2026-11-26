from __future__ import print_function

import torch
import torch.nn.functional as F

try:
    from piq import psnr as _piq_psnr
    from piq import ssim as _piq_ssim
    HAS_PIQ = True
except Exception:
    _piq_psnr = None
    _piq_ssim = None
    HAS_PIQ = False


def _gaussian_kernel(size=11, sigma=1.5, device=None, dtype=torch.float32):
    coordinates = torch.arange(
        size, dtype=dtype, device=device) - (size - 1) / 2.0
    weights = torch.exp(-(coordinates ** 2) / (2.0 * sigma ** 2))
    weights = weights / weights.sum()
    return (weights[:, None] * weights[None, :])[None, None]


def _filter(x, kernel):
    pad = kernel.size(-1) // 2
    return F.conv2d(F.pad(x, (pad,) * 4, mode='reflect'), kernel)


def _ssim_fallback(x, y, data_range=1.0, size=11, sigma=1.5):
    if x.size(1) > 1:
        x = x.mean(dim=1, keepdim=True)
        y = y.mean(dim=1, keepdim=True)
    kernel = _gaussian_kernel(size, sigma, x.device, x.dtype)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x = _filter(x, kernel)
    mu_y = _filter(y, kernel)
    sigma_x = _filter(x * x, kernel) - mu_x ** 2
    sigma_y = _filter(y * y, kernel) - mu_y ** 2
    sigma_xy = _filter(x * y, kernel) - mu_x * mu_y
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    return (numerator / denominator).mean()


def psnr(x, y, data_range=1.0):
    if HAS_PIQ:
        return _piq_psnr(x, y, data_range=data_range).mean()
    mse = F.mse_loss(x, y)
    return 10.0 * torch.log10(
        torch.as_tensor(data_range ** 2, dtype=mse.dtype, device=mse.device)
        / mse.clamp_min(1e-12))


def ssim(x, y, data_range=1.0):
    if HAS_PIQ:
        return _piq_ssim(x, y, data_range=data_range).mean()
    return _ssim_fallback(x, y, data_range=data_range)
