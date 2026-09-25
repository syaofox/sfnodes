"""AuK 本地模型发现（复刻/移植自 ComfyUI-AuK_Doc model_paths.py，MIT）。

上游：DocWorkBox/ComfyUI-AuK_Doc（基于 Tencent-Hunyuan/AuK，MIT，见 auk/LICENSE）。
逐字移植，仅两处适配：日志走 sfnodes 的 get_logger；`_config_model_name` 改名
`config_model_name` 供 loader 复用（不再二次读取 OmegaConf）。

不下载模型、不改动文件系统：只扫描 ComfyUI models 目录并解析 config.yaml/vae/Qwen。
"""

import json
import struct
from pathlib import Path

from omegaconf import OmegaConf

# 顶层包导入时 `...` 正常；测试以 `nodes.audio.auk_paths` 顶层导入时 `...` 越界，
# 回退绝对导入（image_interrogator_api.py 同款可移植性兜底）。
try:
    from ...sf_utils.logger import get_logger
except Exception:  # pragma: no cover - 测试/移植性兜底
    from sf_utils.logger import get_logger  # type: ignore

VARIANT_TO_MODEL_NAME = {'base': 'AuK', 'flash': 'AuK-Flash'}
NODE_CONFIG_DIR = Path(__file__).resolve().parent / 'configs'

logger = get_logger(__name__)


def safetensors_metadata(path):
    """Read the safetensors header metadata without touching tensor data."""
    try:
        with open(path, 'rb') as handle:
            header_length = struct.unpack('<Q', handle.read(8))[0]
            header = json.loads(handle.read(header_length))
    except (OSError, ValueError, struct.error):
        return {}
    metadata = header.get('__metadata__')
    return metadata if isinstance(metadata, dict) else {}


def checkpoint_variant(checkpoint):
    """Infer the AuK variant from repack metadata, falling back to the file name."""
    variant = str(safetensors_metadata(checkpoint).get('auk_variant') or '').lower()
    if variant not in VARIANT_TO_MODEL_NAME:
        stem = checkpoint.stem.lower()
        variant = 'flash' if 'flash' in stem else ('base' if 'base' in stem else '')
    return variant


def config_model_name(config):
    try:
        return OmegaConf.load(config).model.get('name')
    except Exception:
        return None


def _validated_config(config, checkpoint):
    variant = checkpoint_variant(checkpoint)
    expected = VARIANT_TO_MODEL_NAME.get(variant)
    actual = config_model_name(config)
    if expected and actual != expected:
        raise ValueError(
            f'{checkpoint.name} looks like the {variant} variant, but {config} describes {actual}. '
            f'Put the checkpoint next to the matching config.yaml (base -> AuK, flash -> AuK-Flash).'
        )
    return config


def checkpoint_choices(roots):
    names = set()
    for value in roots:
        root = Path(value)
        if not root.is_dir():
            continue
        for path in root.rglob('*.safetensors'):
            if path.name == 'vae.safetensors':
                continue
            if safetensors_metadata(path).get('auk_component') == 'diffusion':
                # ComfyUI-format repack (fp32/bf16/int8/w4a8); config.yaml is resolved separately.
                names.add(path.relative_to(root).as_posix())
            elif (path.parent / 'config.yaml').is_file():
                names.add(path.relative_to(root).as_posix())
    return sorted(names)


def resolve_config(checkpoint, roots):
    """Resolve config.yaml: beside the checkpoint, then a matching external one, then embedded."""
    sibling = checkpoint.parent / 'config.yaml'
    if sibling.is_file():
        return _validated_config(sibling, checkpoint)
    variant = checkpoint_variant(checkpoint)
    target = VARIANT_TO_MODEL_NAME.get(variant)
    if target:
        for value in roots:
            root = Path(value)
            if not root.is_dir():
                continue
            for config in sorted(root.rglob('config.yaml')):
                if config_model_name(config) == target:
                    return _validated_config(config, checkpoint)
        embedded = NODE_CONFIG_DIR / f'{target}.yaml'
        if embedded.is_file():
            logger.info(f'Using embedded config {embedded} for {checkpoint.name}')
            return embedded
    raise FileNotFoundError(
        f'No config.yaml found for {checkpoint.name}. Keep an original AuK or AuK-Flash folder '
        f'under models/auk (config.yaml alone is enough), or place config.yaml beside the checkpoint.'
    )


def resolve_vae(checkpoint, config_path, roots):
    """Resolve the VAE: beside the checkpoint, then models/auk/vae (also models/vae)."""
    sibling = checkpoint.parent / 'vae.safetensors'
    if sibling.is_file():
        return sibling
    for value in roots:
        root = Path(value)
        if not root.is_dir():
            continue
        candidates = sorted(set(root.glob('*.safetensors')) | set(root.glob('*/*.safetensors')))
        for path in candidates:
            if safetensors_metadata(path).get('auk_component') == 'vae':
                return path
        for path in candidates:
            stem = path.stem.lower()
            if 'vae' in stem and 'auk' in stem:
                return path
    if config_path is not None:
        try:
            relative = OmegaConf.load(config_path).model.get('vae', {}).get('vae_model_path')
        except Exception:
            relative = None
        if relative:
            candidate = Path(config_path).parent / str(relative)
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        f'No VAE found for {checkpoint.name}. Put auk_vae.safetensors under models/auk/vae '
        f'(models/vae/auk also works), or place vae.safetensors beside the checkpoint.'
    )


def qwen_choices(roots):
    names = set()
    for value in roots:
        root = Path(value)
        if not root.is_dir():
            continue
        for config in root.rglob('config.json'):
            try:
                data = json.loads(config.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                continue
            if str(data.get('model_type', '')).startswith('qwen2_5_omni'):
                names.add(config.parent.relative_to(root).as_posix())
        for path in root.rglob('*.safetensors'):
            stem = path.stem.lower()
            if safetensors_metadata(path).get('auk_component') == 'encoder' or ('omni' in stem and 'qwen' in stem):
                names.add(path.relative_to(root).as_posix())
    return sorted(names)


def resolve_qwen_config(qwen, roots):
    """Config/tokenizer directory for a Qwen selection (directory or single quantized file)."""
    if qwen.is_dir():
        return qwen
    if (qwen.parent / 'config.json').is_file():
        return qwen.parent
    for value in roots:
        root = Path(value)
        if not root.is_dir():
            continue
        for config in sorted(root.rglob('config.json')):
            try:
                data = json.loads(config.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                continue
            if str(data.get('model_type', '')).startswith('qwen2_5_omni'):
                return config.parent
    raise FileNotFoundError(
        f'No Qwen2.5-Omni config.json found for {qwen.name}. Keep a Qwen2.5-Omni-3B folder under '
        f'models/text_encoders (config and tokenizer files are enough), or place config.json plus the '
        f'tokenizer files beside the .safetensors file.'
    )


def resolve_choice(roots, name):
    for value in roots:
        root = Path(value).resolve()
        target = (root / name).resolve()
        if not target.is_relative_to(root):
            raise ValueError('Model selection must be inside the configured models directory.')
        if target.exists():
            return target
    raise FileNotFoundError(f'Model not found: {name}. Put models in the configured directories and refresh ComfyUI.')
