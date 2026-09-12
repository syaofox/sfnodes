_CATEGORY = "sfnodes/utils"


class SFIgnoreGroups:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ()
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = "复刻孤海忽略多组：节点内嵌画布编组开关面板，一键旁路/禁用整组（默认多选/始终开一个/最多开一个三模式），控制逻辑全在前端"

    def execute(self):
        return ()
