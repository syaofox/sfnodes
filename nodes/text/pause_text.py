"""SFPauseText - an inline STRING gate that pauses a workflow.

复刻 Pixaroma Pause Text：把它放在文本源（LLM / 提示词生成节点）与工作流其余
部分之间。Pause 模式：run 停在此节点并显示模型文本；编辑后按 Continue，把编辑
后的文本送往下游（模型被跳过，快）。Pass 模式：整条工作流一次跑完，透传模型
文本。Keep 模式：每次 run 都复用当前文本（模型被跳过），可快速批量出图。

决策在前端 JS（Pattern #9，与 SFPromptTags 同模式）：app.graphToPrompt hook 注入
生效模式与当前编辑文本到隐藏 PauseState 输入，并在提交时修剪 prompt。本节点只
对拿到的东西做出反应：

  - pause / pass 且有接线输入：输出输入（模型）文本并 emit 到 UI，让盒子显示新文本
  - pause / pass 且无接线：保留前端盒子的文本（不 emit，fresh Run 不会冲掉手打内容）
  - continue：输入连线已被前端剪掉，输出 PauseState 中携带的编辑文本

无磁盘快照——文本足够小，随隐藏输入携带，Python 侧无状态。
"""

import json

_CATEGORY = "sfnodes/text"


class SFPauseText:
    DESCRIPTION = (
        "SF Pause Text - 内联文本闸门：在工作流中此节点处停下，让你在其余部分运行前"
        "阅读并修正一段文本。专为来自语言模型的文本设计（你无法控制确切措辞）。"
        "把文本源接入输入，把下一节点接到输出。\n\n"
        "开关在 Pause 时，按 Run 会停在这里并显示模型文本，工作流其余部分不运行。"
        "编辑文本后按 Continue，只有下游运行，喂给你确认过的确切措辞——模型被跳过，"
        "所以很快。按 Regenerate 获取新文本：节点沿连线回溯，找到生成文本的节点并"
        "把它的种子滚到新随机值，得到不同结果。开关拨到 Pass 则整条工作流一次跑完。"
        "拨到 Keep 则每次 Run 都复用当前文本（模型被跳过），每次 Run 用同一提示词"
        "出图，快速批量变体而不丢失你的编辑。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                # Optional + forceInput：纯连线 STRING 输入（无 widget）。Continue 模式
                # 前端剪掉此链接（模型被跳过），节点以 text=None 运行并返回编辑文本；
                # Pause / Pass 模式则带有线文本。
                "text": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "要闸门的文本。接入你的文本源（LLM / 提示词节点）。",
                    },
                ),
            },
            "hidden": {
                # 前端 app.graphToPrompt hook 注入（Pattern #9）：JSON 字符串
                # {"mode": "pause"|"continue"|"pass", "text": "<box>"}。
                # "text" 是前端盒子当前内容：Continue（输入被剪）与未接线的
                # Pause/Pass（保留盒子）时使用。
                "PauseState": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_TOOLTIPS = (
        "继续往下游的文本：Pause/Pass 是模型文本，Continue 是你编辑后的文本。",
    )
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = _CATEGORY

    # 有意不设 IS_CHANGED。曾经返回 float("nan")：
    # NaN 永不等同于自己 → ComfyUI 认为此节点每次 Run 都变化 → 节点缓存键折叠了
    # 每个祖先的 IS_CHANGED（caching.py::get_node_signature）→ 闸门下游全部失效。
    # 在文生图图上意味着固定种子下每次 Run 仍完整跑一遍采样器。
    # 去掉它没有损失：缓存的节点仍会重发 ui payload 到前端（文本框照样刷新）；
    # 模式与编辑文本都在隐藏 PauseState 输入里，属于 inputs、已在缓存键中，
    # 真正的变化仍会重跑本节点及其下游。

    @staticmethod
    def _as_text(v):
        """把任何接线值转成字符串（None -> ""）。"""
        if v is None:
            return ""
        return v if isinstance(v, str) else str(v)

    def run(self, text=None, PauseState=""):
        try:
            state = json.loads(PauseState) if PauseState else {}
        except Exception:
            state = {}
        if not isinstance(state, dict):
            state = {}
        mode = state.get("mode", "pause")
        box_text = self._as_text(state.get("text", ""))

        if mode == "continue":
            # 输入连线已被前端剪掉；输出编辑后的盒子文本。
            return {"result": (box_text,)}

        # Pause 或 Pass。
        if text is not None:
            # 有线喂来模型文本：透传并 emit，让前端盒子显示新文本（fresh Run 替换编辑）。
            out_text = self._as_text(text)
            return {"ui": {"sf_pause_text": [out_text]}, "result": (out_text,)}

        # 未接线：保留前端盒子里的内容（不 emit，fresh Run 不会冲掉手打文本）。
        return {"result": (box_text,)}
