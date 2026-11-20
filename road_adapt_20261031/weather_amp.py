"""Offline AMP inference check for the already initialized local Weather student."""
import torch

from ultralytics.utils import LOGGER


def check_weather_amp(model):
    """Check finite, equally shaped FP32/AMP predictions without image assets.

    This is a startup capability check, not a loss/gradient stability guarantee.
    RT-DETR query order may differ under AMP, so elementwise equality is not a
    suitable check. The caller keeps recording the effective AMP setting.
    """
    device = next(model.parameters()).device
    if device.type != 'cuda':
        return False
    training = [(module, module.training) for module in model.modules()]
    try:
        # A model forward can consume either CPU or CUDA RNG. Preserve both,
        # including any other visible CUDA devices, on success and failure.
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            model.eval()
            images = torch.zeros((1, 3, 640, 640), device=device, dtype=torch.float32)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    fp32 = model(images)
                with torch.cuda.amp.autocast(enabled=True):
                    mixed = model(images)
            fp32 = fp32[0] if isinstance(fp32, tuple) and fp32 else fp32
            mixed = mixed[0] if isinstance(mixed, tuple) and mixed else mixed
            if not (torch.is_tensor(fp32) and torch.is_tensor(mixed)):
                LOGGER.warning('Weather AMP: local student did not return prediction tensors; using FP32.')
                return False
            if fp32.shape != mixed.shape or not (torch.isfinite(fp32).all() and torch.isfinite(mixed).all()):
                LOGGER.warning('Weather AMP: local predictions differ in shape or are nonfinite; using FP32.')
                return False
        LOGGER.info('Weather AMP: local 640-input FP32/AMP prediction checks passed.')
        return True
    except RuntimeError as exc:
        LOGGER.warning(f'Weather AMP: local CUDA inference check failed; using FP32. {exc}')
        return False
    finally:
        # Restore flags individually: model.train(old) would overwrite an
        # intentionally different child-module mode, including frozen BN.
        for module, was_training in training:
            module.training = was_training
