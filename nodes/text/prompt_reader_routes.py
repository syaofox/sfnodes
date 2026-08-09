"""SFPromptReader 路由（/api/sfnodes/prompt_reader/extract + /list）。

实时读出版本：节点前端在文件变化时调此接口，让用户在运行前就看到提示词。
复刻 Pixaroma 的 /pixaroma/api/prompt_reader/extract：
- Query: ?filename=<image-name>（支持 ComfyUI 的 [input]/[output] 后缀）
- 解析目录内的路径并返回提取出的正向提示词；读不到时返回简短说明
- 恒 200 OK，前端按 `text` / `message` 渲染即可，无需分支 HTTP 状态

目录切换：前端 IN/OUT 按钮按 ?type=input|output 拉取文件列表（图片 + 视频，
递归子目录），output 项由前端拼 [output] 注解（get_annotated_filepath 原生
解析，extract 路由的 allowed_roots 已含 output）。

注册方式沿用 preview_routes.py 先例：模块导入时（__init__.py import）副作用
注册，try/except 包裹，环境异常时降级不注册。
"""

import os

import folder_paths

from ...sf_utils.prompt_reader import read_prompt_from_image, resolve_input_image_name


def _media_dir(source_type):
    """按类型返回目录根：output -> output/，其他 -> input/。"""
    if source_type == "output":
        return folder_paths.get_output_directory()
    return folder_paths.get_input_directory()


def _list_media_recursive(source_type="input"):
    """列出目录下的全部媒体文件（递归子目录，正斜杠相对路径）。

    与节点 INPUT_TYPES 的初始列表同构（filter image+video），供目录切换按钮
    拉取 input/ 或 output/ 的文件清单。失败返回空列表（与 INPUT_TYPES 的
    except 兜底一致）。
    """
    base_dir = _media_dir(source_type)
    files = []
    try:
        if os.path.isdir(base_dir):
            for root, _dirs, fnames in os.walk(base_dir):
                rel_root = os.path.relpath(root, base_dir)
                for fname in fnames:
                    rel = fname if rel_root == "." else os.path.join(rel_root, fname)
                    files.append(rel.replace("\\", "/"))
        files = folder_paths.filter_files_content_types(files, ["image", "video"])
    except Exception:
        files = []
    return sorted(files)


def _is_path_under(path, *roots):
    """realpath 后判断 path 是否位于任一 root 之下（路径穿越防护）。

    folder_paths.get_annotated_filepath 是 ComfyUI 标准解析器，但多用户部署 /
    隧道实例下额外 realpath 校验值得做：路由只读 PNG chunks，一个看起来像
    图片的路径仍可能泄漏文件存在性与可读性信息。
    """
    if not path:
        return False
    try:
        real = os.path.realpath(path)
    except Exception:
        return False
    for root in roots:
        if not root:
            continue
        try:
            real_root = os.path.realpath(root)
            if real == real_root or real.startswith(real_root.rstrip(os.sep) + os.sep):
                return True
        except Exception:
            continue
    return False


def _register_routes():
    try:
        from server import PromptServer
        from aiohttp import web

        ins = getattr(PromptServer, "instance", None)
        if ins is None or not hasattr(ins, "routes"):
            print("[sfnodes] PromptServer instance not available, prompt_reader routes not registered")
            return
        routes = ins.routes

        @routes.get("/api/sfnodes/prompt_reader/extract")
        async def api_sf_prompt_reader_extract(request):
            """实时读出端点：?filename=<image-name> → {found, text|message}。"""
            filename = request.query.get("filename", "")
            if not filename:
                return web.json_response({
                    "found": False,
                    "message": "No image selected.",
                })
            try:
                image_path = folder_paths.get_annotated_filepath(filename)
            except Exception:
                image_path = None
            # Fall back to the resolver for a bare / extension-less name (e.g. a
            # value wired from Load Image SF's filename output) so the live
            # readout can follow a connected node even when it hands us
            # "BunnyExplorer" rather than "BunnyExplorer.png".
            if not image_path or not os.path.isfile(image_path):
                resolved = resolve_input_image_name(filename)
                if resolved:
                    try:
                        image_path = folder_paths.get_annotated_filepath(resolved)
                    except Exception:
                        image_path = None
            if not image_path or not os.path.isfile(image_path):
                return web.json_response({
                    "found": False,
                    "message": "Image file not found in the input folder.",
                })
            allowed_roots = [
                folder_paths.get_input_directory(),
                folder_paths.get_output_directory(),
                folder_paths.get_temp_directory(),
            ]
            if not _is_path_under(image_path, *allowed_roots):
                return web.json_response({
                    "found": False,
                    "message": "Image path is outside the allowed directories.",
                })
            try:
                result = read_prompt_from_image(image_path)
            except Exception as e:
                return web.json_response({
                    "found": False,
                    "message": f"Could not read metadata: {e}",
                })
            return web.json_response(result)

        @routes.get("/api/sfnodes/prompt_reader/list")
        async def api_sf_prompt_reader_list(request):
            """目录文件列表（目录切换按钮）：?type=input|output → [rel/path.png, ...]。

            返回纯相对路径（正斜杠），前端按目录类型决定是否拼 [output] 注解。
            失败返回空列表（200），与 INPUT_TYPES 的 except 兜底一致。
            """
            source_type = request.query.get("type", "input")
            if source_type not in ("input", "output"):
                source_type = "input"
            try:
                files = _list_media_recursive(source_type)
            except Exception:
                files = []
            return web.json_response(files)

        print("[sfnodes] prompt_reader routes registered (/api/sfnodes/prompt_reader/extract, /list)")
    except Exception as e:
        print(f"[sfnodes] prompt_reader route registration failed: {e}")


_register_routes()
