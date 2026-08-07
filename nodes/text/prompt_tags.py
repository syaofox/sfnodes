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
        "SF Prompt Tags - a prompt box with a single text output, plus a personal "
        "library of reusable @tags and an optional text input you can join with.\n\n"
        "Type your prompt and drop in @tags for the parts you reuse a lot. Save a tag "
        "called oilpainting whose full text is a long 'oil painting, thick brush "
        "strokes, ...' and then just type @oilpainting. Type @ in the box for a "
        "searchable list; known tags glow, unknown ones warn you of a typo. *category "
        "picks a random tag from that category each run, #name picks a random line "
        "from a tag saved as a List. Each @tag is swapped for its full text at run "
        "time, so the box stays short. Turn on Show expanded to preview what is sent.\n\n"
        "Wire a prompt into the text input and it is joined with yours - choose My "
        "prompt first or Wired first, and the separator. With nothing wired, the "
        "output is just your prompt.\n\n"
        "Manage tags with the Tags button: a library panel with categories. Your "
        "library is saved in ComfyUI's settings, so it stays private to you and "
        "survives updating the plugin; share it on purpose with Export / Import.\n\n"
        "A workflow run without a browser (pure API) cannot expand @tags. Type into a "
        "plain Text node for those, or wire the text input."
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
