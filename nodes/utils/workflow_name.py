_CATEGORY = "sfnodes/utils"


class SFWorkflowName:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workflow_name": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("workflow_name",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "获取当前工作流的名称（由前端自动填充）"

    def execute(self, workflow_name=""):
        return (workflow_name,)
