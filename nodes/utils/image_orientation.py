_CATEGORY = "sfnodes/utils"


class ImageOrientation:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("is_portrait",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "判断图像方向：竖向（高>宽）返回 True，横向（高<=宽）返回 False"

    def execute(self, image):
        height = image.shape[1]
        width = image.shape[2]
        return (height > width,)
