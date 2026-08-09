"""SFPromptReader 提取路由（/api/sfnodes/prompt_reader/extract）。

实时读出版本：节点前端在文件变化时调此接口，让用户在运行前就看到提示词。
复刻 Pixaroma 的 /pixaroma/api/prompt_reader/extract：
- Query: ?filename=<image-name>（支持 ComfyUI 的 [input] 后缀）
- 解析 input/ 目录内的路径并返回提取出的正向提示词；读不到时返回简短说明
- 恒 200 OK，前端按 `text` / `message` 渲染即可，无需分支 HTTP 状态

注册方式沿用 preview_routes.py 先例：模块导入时（__init__.py import）副作用
注册，try/except 包裹，环境异常时降级不注册。
"""

import os

import folder_paths

from ...sf_utils.prompt_reader import read_prompt_from_image, resolve_input_image_name


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

        print("[sfnodes] prompt_reader route registered (/api/sfnodes/prompt_reader/extract)")
    except Exception as e:
        print(f"[sfnodes] prompt_reader route registration failed: {e}")


_register_routes()
