import os
import re
import time

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
    # realpath 二次校验：normpath+startswith 挡不住 base 下 symlink 子目录指向
    # 外部（os.listdir/open 会跟随链接到 base 之外）。无条件执行——target 自身
    # 是 symlink 指向外部时同样必须拒绝（islink 守卫会让逃逸面漏过）。
    real_target = os.path.realpath(target)
    real_base = os.path.realpath(base)
    if not (real_target == real_base or real_target.startswith(real_base + os.sep)):
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
            # 每次强制重载：返回变化的时间戳而非 NaN（NaN 恒不等于自身会让缓存
            # 键折叠所有祖先、下游每次 Run 全量重跑）。
            return str(time.time_ns())
        # 输入值 + 目录内 txt 文件 (name, mtime) 聚合：目录新增/修改文件也能
        # 触发重跑（只哈希输入值时新增文件不会被感知，输出陈旧）。
        try:
            directory = _resolve_folder(kwargs.get("folder", "default"))
            files = []
            if os.path.isdir(directory):
                for fn in sorted(os.listdir(directory)):
                    if fn.lower().endswith(".txt"):
                        p = os.path.join(directory, fn)
                        try:
                            files.append((fn, os.path.getmtime(p)))
                        except OSError:
                            pass
        except Exception:
            files = []
        return hash((frozenset(kwargs.items()), tuple(files)))

    def load_prompts(self, folder, file_prefix="", file_load_cap=0, start_index=0, load_always=False):
        directory = _resolve_folder(folder)

        try:
            dir_files = os.listdir(directory)
        except OSError:
            dir_files = []
        if not dir_files:
            # 空目录：返回空列表而非抛错（工作流继续跑，与 load_images_path 的
            # 空目录降级一致）。
            print(f"[SFLoadPromptsFromFolder] 目录为空: {directory}")
            return ([], [])

        if file_prefix:
            txt_files = [f for f in dir_files if f.lower().startswith(file_prefix.lower()) and f.lower().endswith('.txt')]
        else:
            txt_files = [f for f in dir_files if f.lower().endswith('.txt')]

        if not txt_files:
            print(f"[SFLoadPromptsFromFolder] 目录中无匹配 .txt: {directory} prefix={file_prefix!r}")
            return ([], [])

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


def _sanitize_stem(name: str) -> str:
    base = str(name).replace("\\", "/").rsplit("/", 1)[-1]
    stem = os.path.splitext(base)[0]
    stem = re.sub(r'[\\/:*?"<>|\x00]', "_", stem)
    return stem or "untitled"


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
                "file_prefix": ("STRING", {"default": "", "tooltip": "生成文件名的前缀（连 filenames 时组合为 {前缀}_{文件名}，留空则不加前缀）"}),
                "filenames": ("STRING", {"tooltip": "保存文件名（不含扩展名，可接 SF Parse Path 的 stem 输出）；按文本行顺序一一配对命名（存在时直接覆盖），不足的行回退 file_prefix 自动递增"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output_path", "file_prefix")
    FUNCTION = "save_text_to_files"
    CATEGORY = _CATEGORY
    DESCRIPTION = "将多行文本的每一行保存为 user/sfnodes/prompt/ 子目录下的独立 txt 文件；连接 filenames 列表时按行顺序用指定文件名保存（直接覆盖），无指定文件名的行自动递增不覆盖已有文件"
    OUTPUT_NODE = True
    INPUT_IS_LIST = True

    def save_text_to_files(self, text, folder, new_folder="", file_prefix="", filenames=None):
        directory = _resolve_folder(folder[0] if isinstance(folder, list) else folder,
                                    new_folder[0] if isinstance(new_folder, list) else new_folder)
        prefix = file_prefix[0] if isinstance(file_prefix, list) else file_prefix

        texts = text if isinstance(text, list) else [text]
        names = list(filenames) if isinstance(filenames, list) else []

        lines = []
        for block in texts:
            for line in str(block).split('\n'):
                if line.strip():
                    lines.append(line.strip())

        def make_name(stem):
            return (prefix + "_" if prefix else "") + stem + ".txt"

        file_count = 1
        for i, line in enumerate(lines):
            if i < len(names) and names[i]:
                filename = make_name(_sanitize_stem(names[i]))
            else:
                while True:
                    filename = make_name(f"{file_count:05d}")
                    if not os.path.exists(os.path.join(directory, filename)):
                        break
                    file_count += 1
                file_count += 1
            with open(os.path.join(directory, filename), 'w', encoding='utf-8') as file:
                file.write(line)

        return (os.path.abspath(directory), prefix)


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
