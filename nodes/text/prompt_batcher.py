import os
import re

import folder_paths
from aiohttp import web

_CATEGORY = "sfnodes/text"


def _get_prompt_base_dir() -> str:
    base = os.path.join(folder_paths.get_user_directory(), "sfnodes", "prompt")
    os.makedirs(base, exist_ok=True)
    return base


def _list_subdirs() -> list:
    base = _get_prompt_base_dir()
    try:
        return sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
    except OSError:
        return []


def _resolve_folder(folder: str, new_folder: str = "") -> str:
    base = os.path.normpath(_get_prompt_base_dir())
    name = (new_folder or folder or "default").strip()
    if not name or os.path.isabs(name):
        name = "default"
    else:
        name = name.lstrip("/\\")
    target = os.path.normpath(os.path.join(base, name))
    if not (target == base or target.startswith(base + os.sep)):
        target = os.path.join(base, "default")
    os.makedirs(target, exist_ok=True)
    return target


class SFLoadPromptsFromFolder:
    @classmethod
    def INPUT_TYPES(cls):
        folders = _list_subdirs() or ["default"]
        return {
            "required": {
                "folder": (folders, {"tooltip": "从 user/sfnodes/prompt/ 下选择提示词子目录，加载其中全部 txt 文件"}),
            },
            "optional": {
                "file_prefix": ("STRING", {"default": "", "tooltip": "仅加载以此前缀开头的 txt 文件"}),
                "file_load_cap": ("INT", {"default": 0, "min": 0, "step": 1, "tooltip": "限制加载的文件数量（0 = 无限制）"}),
                "start_index": ("INT", {"default": 0, "min": 0, "step": 1, "tooltip": "从文件列表中的此索引开始加载"}),
                "load_always": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled", "tooltip": "每次执行时强制重新加载文件"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "file_path")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "load_prompts"
    CATEGORY = _CATEGORY
    DESCRIPTION = "从 user/sfnodes/prompt/ 下的子目录中批量加载 txt 提示词文件，输出提示词列表与文件路径列表，支持前缀筛选、数量限制与起始索引"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if kwargs.get("load_always", False):
            return float("NaN")
        return hash(frozenset(kwargs.items()))

    def load_prompts(self, folder, file_prefix="", file_load_cap=0, start_index=0, load_always=False):
        directory = _resolve_folder(folder)

        dir_files = os.listdir(directory)
        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{directory}'.")

        if file_prefix:
            txt_files = [f for f in dir_files if f.lower().startswith(file_prefix.lower()) and f.lower().endswith('.txt')]
        else:
            txt_files = [f for f in dir_files if f.lower().endswith('.txt')]

        if not txt_files:
            raise FileNotFoundError(f"No matching .txt files found in directory '{directory}' with prefix '{file_prefix}'.")

        def sort_key(filename):
            match = re.search(r'\d+', filename)
            if match:
                return int(match.group())
            return filename

        txt_files = sorted(txt_files, key=sort_key)
        txt_files = [os.path.join(directory, x) for x in txt_files][start_index:]

        prompts = []
        file_paths = []
        file_count = 0
        for txt_path in txt_files:
            if file_load_cap > 0 and file_count >= file_load_cap:
                break
            try:
                with open(txt_path, 'r', encoding='utf-8') as file:
                    prompts.append(file.read().strip())
                    file_paths.append(os.path.abspath(txt_path))
                    file_count += 1
            except OSError:
                pass

        return (prompts, file_paths)


class SFSaveTextToFiles:
    @classmethod
    def INPUT_TYPES(cls):
        folders = _list_subdirs() or ["default"]
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "tooltip": "多行文本，每行将保存为一个单独的 txt 文件"}),
                "folder": (folders, {"tooltip": "保存到 user/sfnodes/prompt/ 下的该子目录"}),
            },
            "optional": {
                "new_folder": ("STRING", {"default": "", "tooltip": "非空时优先：自动创建该子目录并保存到其中"}),
                "file_prefix": ("STRING", {"default": "Scene", "tooltip": "生成文件名的前缀"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output_path", "file_prefix")
    FUNCTION = "save_text_to_files"
    CATEGORY = _CATEGORY
    DESCRIPTION = "将多行文本的每一行保存为 user/sfnodes/prompt/ 子目录下的独立 txt 文件，文件名自动递增不覆盖已有文件"
    OUTPUT_NODE = True

    def save_text_to_files(self, text, folder, new_folder="", file_prefix="Scene"):
        directory = _resolve_folder(folder, new_folder)

        lines = text.split('\n')
        file_count = 1
        for line in lines:
            if line.strip():
                while True:
                    filename = f"{file_prefix}_{file_count:05d}.txt"
                    filepath = os.path.join(directory, filename)
                    if not os.path.exists(filepath):
                        break
                    file_count += 1
                with open(filepath, 'w', encoding='utf-8') as file:
                    file.write(line.strip())
                file_count += 1

        return (os.path.abspath(directory), file_prefix)


def _register_prompt_routes():
    try:
        from server import PromptServer
        ins = getattr(PromptServer, "instance", None)
        if ins is None or not hasattr(ins, "routes"):
            return
        routes = ins.routes

        @routes.get("/api/sfnodes/prompt/folders")
        async def _list_folders(request: web.Request) -> web.Response:
            try:
                return web.json_response({"folders": _list_subdirs()})
            except Exception:
                return web.Response(status=500)
    except Exception:
        pass


_register_prompt_routes()
