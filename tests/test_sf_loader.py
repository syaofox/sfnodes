"""sfnodes 包骨架加载器：在无 ComfyUI 环境加载 nodes/ 下带相对导入的模块。

用法：
    import test_sf_loader as L
    simple_math = L.load_node("nodes/utils/simple_math.py")
"""
import importlib.util
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 子包 -> 相对 ROOT 的路径
_SUBPACKAGES = {
    "sfnodes": ".",
    "sfnodes.sf_utils": "sf_utils",
    "sfnodes.nodes": "nodes",
    "sfnodes.nodes.face": "nodes/face",
    "sfnodes.nodes.image": "nodes/image",
    "sfnodes.nodes.mask": "nodes/mask",
    "sfnodes.nodes.model": "nodes/model",
    "sfnodes.nodes.text": "nodes/text",
    "sfnodes.nodes.utils": "nodes/utils",
    "sfnodes.nodes.inpaint": "nodes/inpaint",
}


def _ensure_pkgs():
    for name, path in _SUBPACKAGES.items():
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = [os.path.join(ROOT, path)]
            sys.modules[name] = m


def load_node(rel_path):
    """加载 nodes/ 或 sf_utils/ 下的模块（含相对导入），返回模块对象。

    nodes/utils/simple_math.py -> sfnodes.nodes.utils.simple_math
    sf_utils/disk_state.py      -> sfnodes.sf_utils.disk_state
    """
    _ensure_pkgs()
    rel = rel_path.replace(os.sep, "/")
    if rel.startswith("nodes/"):
        assert rel.endswith(".py"), rel
        stem = rel[len("nodes/"):-3]
        full_name = "sfnodes.nodes." + stem.replace("/", ".")
    elif rel.startswith("sf_utils/"):
        assert rel.endswith(".py"), rel
        stem = rel[len("sf_utils/"):-3]
        full_name = "sfnodes.sf_utils." + stem.replace("/", ".")
    else:
        raise AssertionError(f"unsupported path: {rel}")
    spec = importlib.util.spec_from_file_location(
        full_name, os.path.join(ROOT, rel_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod
