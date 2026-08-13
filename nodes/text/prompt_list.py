_CATEGORY = "sfnodes/text"


class SFPromptList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "multiline_text": ("STRING", {"multiline": True, "default": "body_text", "tooltip": "多行文本，每行将作为列表的一项"}),
                "prepend_text": ("STRING", {"multiline": False, "default": "", "tooltip": "添加到每行前面的文本"}),
                "append_text": ("STRING", {"multiline": False, "default": "", "tooltip": "添加到每行后面的文本"}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 9999, "tooltip": "起始行索引"}),
                "max_rows": ("INT", {"default": 1000, "min": 1, "max": 9999, "tooltip": "最大行数"}),
                "skip_empty": ("BOOLEAN", {"default": True, "tooltip": "过滤空白行（去掉首尾空白后为空的行）"}),
                "wrap_text": ("BOOLEAN", {"default": False, "tooltip": "编辑器自动换行（仅编辑体验，不影响输出）"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "body_text")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "make_list"
    CATEGORY = _CATEGORY
    DESCRIPTION = "将多行文本按行拆分，每行可添加前后缀，支持索引切片与空白行过滤，输出字符串列表"

    def make_list(self, multiline_text, prepend_text="", append_text="", start_index=0, max_rows=9999, skip_empty=True, wrap_text=False):
        lines = multiline_text.split('\n')

        if skip_empty:
            lines = [line for line in lines if line.strip()]

        start_index = max(0, min(start_index, len(lines) - 1))
        end_index = min(start_index + max_rows, len(lines))

        selected_rows = lines[start_index:end_index]
        prompt_list = [prepend_text + line + append_text for line in selected_rows]
        body_list = selected_rows

        return (prompt_list, body_list)
