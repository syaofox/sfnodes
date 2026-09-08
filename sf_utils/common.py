class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


def _parse_fill_color(fill_color):
    """Parse a fill color into an (r, g, b) tuple of 0-255 ints.

    Accepts a "#rrggbb" / "rrggbb" hex string or any 3-sequence of ints.
    (从 nodes/mask/masks.py 提升的公共实现——SFMaskFill 与 SFImageCropExpand 共用；
    前端取色器输出恒为 6 位 hex，故不支持 #RGB 缩写与 "r,g,b" 字符串。)
    """
    if isinstance(fill_color, str):
        hex_color = fill_color.lstrip("#")
        return (
            int(hex_color[0:2], 16),
            int(hex_color[2:4], 16),
            int(hex_color[4:6], 16),
        )
    return tuple(fill_color)
