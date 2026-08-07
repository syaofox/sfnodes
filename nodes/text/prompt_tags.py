"""SFPromptTags - a prompt box where @tags expand to reusable snippets.

A multi-line prompt with a single text output, plus a personal library of named
tags (saved in ComfyUI's user settings, private to the machine). Type @name
(with autocomplete) to insert a short tag; at queue time the frontend swaps each
@tag for the longer prompt it stands for. It also has an OPTIONAL text input:
wire another prompt in and it is JOINED with your prompt - you pick which comes
first and how they are separated.

Division of labour (the Pixaroma Prompt pattern):
  * @tag EXPANSION happens on the frontend, in the app.graphToPrompt hook in
    web/sf_prompt_tags.js, reading the tag library from ComfyUI settings. The
    expanded prompt + the join order + separator are injected into the hidden
    PromptState input.
  * This node parses PromptState and JOINS it with the wired text_in (which is a
    real link, only known here at execution).

Consequences (both deliberate, matching Pixaroma Prompt):
  * A pure API / headless run (no browser) sends PromptState "{}" -> the typed
    prompt is empty there; wire text_in for headless pipelines, or type into a
    plain Text node.
  * The tag library lives on the user's machine (ComfyUI settings), never in the
    workflow, so a shared workflow keeps the author's prompts private.
"""

import json

_CATEGORY = "sfnodes/text"


def _clean_str(v):
    return v if isinstance(v, str) else ""


class SFPromptTags:
    DESCRIPTION = (
        "SF Prompt Tags - 带个人标签库的提示词框：单个文本输出，外加可复用的"
        "@tags 库与一个可拼接的可选文本输入。\n\n"
        "输入你的提示词，把常用部分存成标签后用 @标签名 插入。保存一个名为 "
        "oilpainting 的标签（全文是一长串 'oil painting, thick brush strokes, "
        "...'），之后只需输入 @oilpainting。输入 @ 弹出可搜索列表；已知标签"
        "高亮，未知名称以波浪线提示拼写错误。*分类 每次 run 从该分类随机选一个"
        "标签，#名称 从保存为 List 的标签里随机取一行。每个 @tag 在运行时替换为"
        "它的完整文本，输入框因此保持简短。打开 Show expanded 可预览实际发送的"
        "内容。\n\n"
        "把另一个提示词接入 text 输入后，将与你输入框中的内容拼接——可选"
        "「My prompt first / Wired first」顺序与分隔符。不接线时输出就是你的"
        "提示词。\n\n"
        "用 Tags 按钮管理标签库：分类面板支持增删改。你的库保存在 ComfyUI 的"
        "设置里，只属于你、随插件更新存活；可通过 Export / Import 主动分享。\n\n"
        "无浏览器的纯 API 运行无法展开 @tags——这类场景请改用普通 Text 节点，"
        "或接入 text 输入。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "text_in": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": (
                            "可选。接入另一个提示词后，将与本节点输入框中的提示词拼接"
                            "（顺序与分隔符在节点上设置）。不接线时仅输出输入框内容。"
                        ),
                    },
                ),
            },
            # 前端 graphToPrompt hook 注入：{"text": <展开后的提示词>, "order": "mine"|"wired",
            # "sep": <分隔符>}。见 web/sf_prompt_tags.js。
            "hidden": {"PromptState": ("STRING", {"default": "{}"})},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_TOOLTIPS = (
        "每个 @tag 展开后的提示词；若接入了 text_in，则按顺序与分隔符与其拼接后的结果。",
    )
    FUNCTION = "run"
    CATEGORY = _CATEGORY

    @staticmethod
    def _parse_state(raw):
        try:
            state = json.loads(raw) if isinstance(raw, str) else {}
        except (ValueError, TypeError, RecursionError):
            state = {}
        if not isinstance(state, dict):
            state = {}
        mine = _clean_str(state.get("text"))
        order = state.get("order")
        order = order if order in ("mine", "wired") else "mine"
        sep = state.get("sep")
        # 手改的 API 文件可能传非字符串 / 离谱长度的分隔符。
        sep = sep if isinstance(sep, str) and len(sep) <= 16 else ", "
        return mine, order, sep

    def run(self, text_in=None, PromptState="{}"):
        mine, order, sep = self._parse_state(PromptState)
        # 某些上游节点传入的 STRING 可能是长度为 1 的列表。
        if isinstance(text_in, (list, tuple)):
            text_in = text_in[0] if text_in else ""
        other = _clean_str(text_in)

        # 未接线（或解析为空）-> 仅输出输入框内容。
        if not other.strip():
            return (mine,)
        if not mine.strip():
            return (other,)
        if order == "wired":
            return (other + sep + mine,)
        return (mine + sep + other,)
