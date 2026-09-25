"""ComfyUI-llama-cpp_vlm 插件桥接（软依赖）：查找已加载插件的 LLAMA_CPP_STORAGE。

插件目录名含连字符、由 ComfyUI 以路径式模块名加载，不能按包名 import；按属性指纹在
sys.modules 里找已加载实例，避免二次实例化导致 LLAMA_CPP_STORAGE 分裂（用户 Loader
加载的模型必须与调用方是同一份）。

指纹必须验到 storage 具备 load_model/clean（仅查名字会撞上插件注册的
`torch.ops.LLAMA_CPP_STORAGE` 命名空间——实测踩坑），并用插件节点类名
llama_cpp_instruct_adv 排除同名 storage 的其他插件。

使用方：SFQwenImage21PromptEnhancer 本地LLaMA 模式、SFAuKLlamaCppSettings（PE 适配）。
"""

import sys


def find_llama_plugin():
    """返回已加载的插件模块（未安装/未加载返回 None）。"""
    for module in list(sys.modules.values()):
        try:
            storage = getattr(module, "LLAMA_CPP_STORAGE", None)
            if storage is None or not hasattr(storage, "load_model") or not hasattr(storage, "clean"):
                continue
            if not hasattr(module, "llama_cpp_instruct_adv"):
                continue
            return module
        except Exception:  # 惰性加载模块的 __getattr__ 可能抛非 AttributeError
            continue
    return None
