import torch

_CATEGORY = "sfnodes/mask"


class ConditionalInvertMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "condition": (
                    ["all_black", "all_white"],
                    {"default": "all_black"},
                ),
            },
            "optional": {},
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "根据条件反转Mask:当Mask全黑(all_black)或全白(all_white)时才反转,否则返回原Mask"

    def execute(self, mask, condition):
        mask_flat = mask.reshape(-1)
        
        if condition == "all_black":
            should_invert = torch.all(mask_flat == 0.0).item()
        else:
            should_invert = torch.all(mask_flat == 1.0).item()
        
        if should_invert:
            result = 1.0 - mask
        else:
            result = mask
        
        return (result,)


NODE_CLASS_MAPPINGS = {
    "SFConditionalInvertMask": ConditionalInvertMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SFConditionalInvertMask": "SF Conditional Invert Mask",
}
