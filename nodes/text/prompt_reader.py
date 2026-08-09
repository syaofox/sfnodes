"""SFPromptReader - extract the positive prompt embedded in an image.

复刻 Pixaroma Prompt Reader：读取 PNG tEXt chunks（ComfyUI workflow JSON 或
A1111 'parameters'），从采样器反推正向 CLIP-text-encode 链并返回底层文本。
仅 STRING 输出（无 IMAGE/MASK 侧）。图片没有内嵌 prompt 时返回一句说明文本，
下游节点仍能拿到可用值。
"""

import os

import folder_paths

from ...sf_utils.prompt_reader import read_prompt_from_image, resolve_input_image_name

_CATEGORY = "sfnodes/text"


class SFPromptReader:
    DESCRIPTION = (
        "SF Prompt Reader - 读取用 ComfyUI（或 Automatic1111 / Forge）生成的图片，"
        "恢复保存在 PNG 元数据里的正向提示词。无图片预览，只输出文本。\n\n"
        "把 PNG 拖到节点上、点 Upload Image 上传、或从文件下拉中选择——选中的瞬间"
        "即开始读取，运行前就能看到提示词。输出为 STRING，可直接接入 "
        "CLIPTextEncode 或其他文本输入复用。\n\n"
        "图片没有内嵌提示词（JPG、截图、或元数据丢失的 PNG）时，输出一句说明文本"
        "并显示在节点上，下游接线不受影响。\n\n"
        "支持带文本链的 ComfyUI 工作流（ConditioningCombine、StringConcatenate、"
        "SDXL 双文本编码器），也支持 Automatic1111 / Forge 的 'parameters' 格式，"
        "以及 SFPromptTags、SFValueDropdown、SFTextPreset、SFAnythingIndexSwitch、"
        "SFPauseText 等本插件文本节点；对 Pixaroma 生态节点（Prompt Stack / Multi "
        "/ Pack / Dropdown / Switch 等）生成的图片同样兼容。\n\n"
        "可选 'filename' 输入可接入图片文件名（例如 Load Image SF 的 filename 输出）"
        "——接线期间节点忽略自己的选择器，改读该图片的提示词；在节点上选择、上传或"
        "拖放文件即接管并自动断开该连线。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        # Walk input/ recursively so subfolder PNGs are listed too. Forward
        # slashes in the paths so folder_paths.get_annotated_filepath resolves
        # them correctly cross-platform. Mirrors node_load_image.py.
        input_dir = folder_paths.get_input_directory()
        files = []
        try:
            if os.path.isdir(input_dir):
                for root, _dirs, fnames in os.walk(input_dir):
                    rel_root = os.path.relpath(root, input_dir)
                    for fname in fnames:
                        rel = fname if rel_root == "." else os.path.join(rel_root, fname)
                        files.append(rel.replace("\\", "/"))
            files = folder_paths.filter_files_content_types(files, ["image"])
        except Exception:
            files = []
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True, "tooltip": "要读取提示词的图片。上传、拖放或选择用 ComfyUI / Automatic1111 / Forge 生成的 PNG，以便恢复其中内嵌的提示词。选中的瞬间即开始读取。"}),
            },
            "optional": {
                # Wire-only (no widget). When connected it drives the read and
                # the picker above is ignored. Load Image SF's filename
                # output is extension-less, so read() resolves it back to the
                # real file via resolve_input_image_name.
                "filename": ("STRING", {"forceInput": True, "tooltip": "可选。把图片文件名接入此处（例如 Load Image SF 的 filename 输出）以自动读取该图片的提示词。接线期间节点忽略自己的选择器；在节点上选择、上传或拖放文件即接管并断开连线。"}),
            },
        }

    CATEGORY = _CATEGORY
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_TOOLTIPS = ("从图片元数据恢复的提示词；未找到时返回一句说明文本。",)
    FUNCTION = "read"
    OUTPUT_NODE = True

    @staticmethod
    def _effective_name(image, filename):
        """Pick which image to read: the wired filename wins over the picker.

        Returns (name, error_message). When a filename is wired but cannot be
        matched to a real file, name is None and error_message explains it so
        read() can surface that to the user instead of silently falling back to
        the picker (which would be confusing).
        """
        wired = filename.strip() if isinstance(filename, str) else ""
        if wired:
            resolved = resolve_input_image_name(wired)
            if not resolved:
                return None, (
                    f"Could not find an image named '{wired}' in the input "
                    "folder. Make sure the image sent by the connected node "
                    "is present in ComfyUI's input folder."
                )
            return resolved, None
        return image, None

    def read(self, image: str, filename: str = None):
        name, err = self._effective_name(image, filename)
        if err:
            return {"ui": {"text": [err]}, "result": (err,)}
        try:
            image_path = folder_paths.get_annotated_filepath(name)
        except Exception:
            text = "Image file not found in the input folder."
            return {"ui": {"text": [text]}, "result": (text,)}

        result = read_prompt_from_image(image_path)
        if result.get("found"):
            text = result.get("text") or ""
        else:
            text = result.get("message") or "No prompt found in this image."
        return {"ui": {"text": [text]}, "result": (text,)}

    @classmethod
    def IS_CHANGED(cls, image, filename=None):
        # Use (mtime, size) instead of a full-file SHA hash. ComfyUI's native
        # LoadImage hashes the file content, but we only need to know whether
        # the file changed - a 50MB PNG hashed on every run is wasteful.
        # mtime+size catches every realistic edit (the only false-negative is
        # an in-place byte swap that preserves size AND mtime, which doesn't
        # happen in practice when ComfyUI re-saves or the user re-uploads).
        # Reflect the EFFECTIVE file (wired filename wins) so a change on the
        # connected image also invalidates the cache and re-runs.
        name, _err = cls._effective_name(image, filename)
        if not name:
            wired = filename.strip() if isinstance(filename, str) else ""
            if wired:
                # Wired but unresolvable - key on the raw name so it re-checks
                # when the file appears / the wire changes.
                return f"unresolved:{wired}"
            # Nothing selected at all - always re-run (nan), same as before.
            return float("nan")
        try:
            image_path = folder_paths.get_annotated_filepath(name)
            st = os.stat(image_path)
            return f"{st.st_mtime_ns}:{st.st_size}"
        except Exception:
            return f"name:{name}"

    @classmethod
    def VALIDATE_INPUTS(cls, image=None, filename=None):
        # Never hard-block the graph: the node always runs and reports any
        # problem (missing file, no metadata) via its readout / output string,
        # so downstream wiring keeps working. This also means a wired filename
        # driving the read is never blocked by a stale picker value, and an
        # uploaded file not yet in the combo list is accepted.
        return True


NODE_CLASS_MAPPINGS = {"SFPromptReader": SFPromptReader}
NODE_DISPLAY_NAME_MAPPINGS = {"SFPromptReader": "SF Prompt Reader"}
