import os

_CATEGORY = "sfnodes/utils"


def _parse_entry(s: str) -> tuple:
    norm = (s or "").replace("\\", "/")
    if "/" in norm:
        dirname, filename = norm.rsplit("/", 1)
    else:
        dirname, filename = "", norm
    stem, ext = os.path.splitext(filename)
    return dirname, filename, ext, stem


class SFParsePath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "paths": ("STRING", {
                    "default": "",
                    "tooltip": "文件路径或文件名（自动识别）；连接列表源（如 SF Load Images Path 的 filenames/file_paths）时自动逐项解析输出列表",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("path", "filename", "extension", "stem")
    OUTPUT_IS_LIST = (True, True, True, True)
    OUTPUT_TOOLTIPS = ("路径（不含文件名；纯文件名时为空字符串）", "文件名（含扩展名）", "扩展名（含点，如 .png；无扩展名为空）", "不含扩展名的文件名")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "解析文件路径或文件名（自动识别全路径/纯文件名），输出路径、文件名、扩展名、不含扩展名的文件名；连接列表源时逐项解析输出列表"

    def execute(self, paths):
        return tuple([x] for x in _parse_entry(paths))
