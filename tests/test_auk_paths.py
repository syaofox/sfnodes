# AuK 模型发现纯逻辑测试（python tests/test_auk_paths.py）
# 覆盖：safetensors metadata 读取、变体推断（文件名/metadata）、checkpoint 列表（含
# ComfyUI repack 与 vae.safetensors 排除）、config.yaml 解析（相邻/外部/内置兜底/变体不匹配报错）、
# VAE 解析（相邻/metadata/config 相对路径）、Qwen 列表与 config 目录解析、路径包含校验。
# 无 torch/ComfyUI/模型依赖：safetensors 头与 config.yaml 均为临时目录内的最小构造。

import json
import os
import pathlib
import struct
import sys
import tempfile
import types

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

# ── 最小 mock：omegaconf（host 无该包；用 yaml + 属性访问对象等价 model.name 读取）──
import yaml  # noqa: E402


class _Cfg:
    def __init__(self, data):
        self._data = data

    def __getattr__(self, name):
        value = self._data[name]
        return _Cfg(value) if isinstance(value, dict) else value

    def get(self, name, default=None):
        value = self._data.get(name, default)
        return _Cfg(value) if isinstance(value, dict) else value


omegaconf = types.ModuleType("omegaconf")
omegaconf.OmegaConf = types.SimpleNamespace(
    load=lambda path: _Cfg(yaml.safe_load(open(path, encoding="utf-8")))
)
sys.modules["omegaconf"] = omegaconf

from nodes.audio import auk_paths as P  # noqa: E402

failures = []


def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")


def raises(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
        return False
    except Exception:
        return True


def make_config(path, name, vae_path=None, extra=None):
    data = {"model": {"name": name}}
    if vae_path:
        data["model"]["vae"] = {"vae_model_path": vae_path}
    if extra:
        data["model"].update(extra)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True)


def make_safetensors(path, metadata=None, tensors=None):
    header = dict(tensors or {})
    if metadata is not None:
        header["__metadata__"] = metadata
    payload = json.dumps(header).encode("utf-8")
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(payload)))
        handle.write(payload)


work = tempfile.mkdtemp(prefix="sf_auk_paths_")
models = os.path.join(work, "auk")
os.makedirs(os.path.join(models, "AuK"))
os.makedirs(os.path.join(models, "AuK-Flash"))
os.makedirs(os.path.join(models, "vae"))
os.makedirs(os.path.join(models, "repack"))
os.makedirs(os.path.join(work, "text_encoders", "Qwen2.5-Omni-3B"))

make_config(os.path.join(models, "AuK", "config.yaml"), "AuK")
make_config(os.path.join(models, "AuK-Flash", "config.yaml"), "AuK-Flash")
make_safetensors(os.path.join(models, "AuK", "auk_base.safetensors"))
make_safetensors(os.path.join(models, "AuK-Flash", "auk_flash.safetensors"))
make_safetensors(os.path.join(models, "AuK", "vae.safetensors"))
make_safetensors(os.path.join(models, "vae", "auk_vae.safetensors"), {"auk_component": "vae"})
make_safetensors(os.path.join(models, "repack", "auk_base_int8.safetensors"),
                 {"auk_component": "diffusion", "auk_variant": "base"})
make_safetensors(os.path.join(models, "repack", "mystery_w4a8.safetensors"), {"auk_component": "diffusion"})
make_safetensors(os.path.join(work, "text_encoders", "qwen_omni_int8.safetensors"), {"auk_component": "encoder"})
with open(os.path.join(work, "text_encoders", "Qwen2.5-Omni-3B", "config.json"), "w", encoding="utf-8") as handle:
    json.dump({"model_type": "qwen2_5_omni"}, handle)

# ── metadata / 变体 ──
check("metadata 读取", P.safetensors_metadata(os.path.join(models, "repack", "auk_base_int8.safetensors")).get("auk_variant") == "base")
check("无 metadata 返回空", P.safetensors_metadata(os.path.join(models, "AuK", "auk_base.safetensors")) == {})
check("坏文件返回空", P.safetensors_metadata(os.path.join(work, "missing.safetensors")) == {})
check("文件名推断 base", P.checkpoint_variant(pathlib.Path(os.path.join(models, "AuK", "auk_base.safetensors"))) == "base")
check("文件名推断 flash", P.checkpoint_variant(pathlib.Path(os.path.join(models, "AuK-Flash", "auk_flash.safetensors"))) == "flash")
check("metadata 优先于文件名", P.checkpoint_variant(pathlib.Path(os.path.join(models, "repack", "auk_base_int8.safetensors"))) == "base")
check("未知变体为空", P.checkpoint_variant(pathlib.Path(os.path.join(models, "repack", "mystery_w4a8.safetensors"))) == "")

# ── checkpoint 列表 ──
choices = P.checkpoint_choices([models])
check("列出相邻 config 的权重", "AuK/auk_base.safetensors" in choices and "AuK-Flash/auk_flash.safetensors" in choices)
check("列出 repack 权重", "repack/auk_base_int8.safetensors" in choices and "repack/mystery_w4a8.safetensors" in choices)
check("排除 vae.safetensors", "AuK/vae.safetensors" not in choices)
check("排除非 AuK 目录", not any(c.startswith("vae/") for c in choices))

# ── config 解析 ──
check("相邻 config 优先", P.resolve_config(
    pathlib.Path(os.path.join(models, "AuK", "auk_base.safetensors")), [models]
) == pathlib.Path(os.path.join(models, "AuK", "config.yaml")))
check("外部同类 config 优先于内置", P.resolve_config(
    pathlib.Path(os.path.join(models, "repack", "auk_base_int8.safetensors")), [models]
) == pathlib.Path(os.path.join(models, "AuK", "config.yaml")))
check("无外部 config 走内置兜底", P.resolve_config(
    pathlib.Path(os.path.join(models, "repack", "auk_base_int8.safetensors")), []
) == pathlib.Path(root) / "nodes" / "audio" / "configs" / "AuK.yaml")

bad_dir = os.path.join(models, "bad")
os.makedirs(bad_dir)
make_config(os.path.join(bad_dir, "config.yaml"), "AuK")
make_safetensors(os.path.join(bad_dir, "auk_flash.safetensors"))
check("变体不匹配报错", raises(P.resolve_config, pathlib.Path(os.path.join(bad_dir, "auk_flash.safetensors")), [models]))

unknown = os.path.join(models, "unknown")
os.makedirs(unknown)
make_safetensors(os.path.join(unknown, "mystery.safetensors"))
check("未知变体无 config 报错", raises(P.resolve_config, pathlib.Path(os.path.join(unknown, "mystery.safetensors")), [models]))

# ── VAE 解析 ──
check("相邻 vae 优先", P.resolve_vae(
    pathlib.Path(os.path.join(models, "AuK", "auk_base.safetensors")), None, [os.path.join(models, "vae")]
) == pathlib.Path(os.path.join(models, "AuK", "vae.safetensors")))

config_only = os.path.join(models, "cfgonly")
os.makedirs(config_only)
make_safetensors(os.path.join(config_only, "auk_base.safetensors"))
make_config(os.path.join(config_only, "config.yaml"), "AuK", vae_path=os.path.join("..", "vae", "auk_vae.safetensors"))
check("config 相对路径兜底", os.path.realpath(P.resolve_vae(
    pathlib.Path(os.path.join(config_only, "auk_base.safetensors")),
    os.path.join(config_only, "config.yaml"),
    [],
)) == os.path.realpath(os.path.join(models, "vae", "auk_vae.safetensors")))

check("metadata auk_component=vae", P.resolve_vae(
    pathlib.Path(os.path.join(config_only, "auk_base.safetensors")),
    None,
    [os.path.join(models, "vae")],
) == pathlib.Path(os.path.join(models, "vae", "auk_vae.safetensors")))

check("无 VAE 报错", raises(P.resolve_vae, pathlib.Path(os.path.join(config_only, "auk_base.safetensors")), None, []))

# ── Qwen 解析 ──
qwen = P.qwen_choices([os.path.join(work, "text_encoders")])
check("Qwen 目录识别", "Qwen2.5-Omni-3B" in qwen)
check("Qwen 单文件识别", "qwen_omni_int8.safetensors" in qwen)
check("Qwen config 目录（文件夹）", P.resolve_qwen_config(
    pathlib.Path(work) / "text_encoders" / "Qwen2.5-Omni-3B", [os.path.join(work, "text_encoders")]
) == pathlib.Path(work) / "text_encoders" / "Qwen2.5-Omni-3B")

single = os.path.join(work, "text_encoders", "Qwen2.5-Omni-3B")
os.makedirs(os.path.join(work, "single"))
make_safetensors(os.path.join(work, "single", "qwen_omni_int8.safetensors"))
with open(os.path.join(work, "single", "config.json"), "w", encoding="utf-8") as handle:
    json.dump({"model_type": "qwen2_5_omni"}, handle)
check("Qwen config 目录（单文件同级）", P.resolve_qwen_config(
    pathlib.Path(work) / "single" / "qwen_omni_int8.safetensors", []
) == pathlib.Path(work) / "single")
check("Qwen config 缺失报错", raises(P.resolve_qwen_config, pathlib.Path(work) / "nope.safetensors", []))

# ── 选择与包含校验 ──
check("resolve_choice 命中", P.resolve_choice([models], "AuK/auk_base.safetensors") == (pathlib.Path(models) / "AuK" / "auk_base.safetensors").resolve())
check("resolve_choice 越界拒绝", raises(P.resolve_choice, [models], "../outside.safetensors"))
check("resolve_choice 缺失报错", raises(P.resolve_choice, [models], "AuK/missing.safetensors"))

if failures:
    print(f"\n{len(failures)} 项失败：")
    for name in failures:
        print("  -", name)
    sys.exit(1)
print("\nOK")
