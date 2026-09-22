"""Qwen Image 2.1 提示词增强协议纯逻辑（无 torch / app 依赖，可直跑单测）。

供 SFQwenImage21PromptEnhancer 使用：任务 profile（官方采样参数 / 图片要求）、
官方 PE 系统提示词（qwen21_prompts.py 原文）与通用自写协议的选择、本地聊天文本
构造（`<|im_start|>` 原文 + 视觉占位 + thinking 预填）、输出语言指令、思考链分割
与答案解析（严格 JSON → 正则字段恢复 → 原文兜底）。

官方 PE 契约（Qwen/Qwen-Image-2.1-PE-T2I·I2I）：
- 文生图 {"rewritten_prompt": ..., "wh_ratio": "16:9"}
- 图生图 {"rewritten_prompt": ..., "wh_ratio": "", "ratio_follow": "<image1>"}
  wh_ratio 与 ratio_follow 互斥，恰好一个带值。

与 sf_utils/llm_client.py 分工：那边管 OpenAI 请求形状与网络，这里管消息文字、
本地聊天模板、答案契约解析；与 prompt_rewrite.py（§115 观察者协议）协议不同，
不复用其常量与校验（仅节点复用 llm_client）。
"""

import json
import re
from dataclasses import dataclass

from .qwen21_prompts import OFFICIAL_EDIT_SYSTEM_PROMPT, OFFICIAL_T2I_SYSTEM_PROMPT

TASK_T2I = "t2i"
TASK_EDIT = "edit"

LANGUAGE_EN = "en"
LANGUAGE_ZH = "zh"

MAX_IMAGES = 8

# Qwen Image 2.1 官方 7 档比例（模型卡 Supported Aspect Ratios，顺序一致）
RATIO_OPTIONS = ("1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16")

VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"

_RATIO_FOLLOW_RE = re.compile(r"^<image([1-9]\d*)>$")


@dataclass(frozen=True)
class TaskProfile:
    """任务差异集中一处：官方生产推理参数（不同任务不可互换）。"""

    name: str
    requires_images: bool
    has_ratio_follow: bool
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    max_new_tokens: int = 24000


PROFILES = {
    TASK_T2I: TaskProfile(
        name=TASK_T2I,
        requires_images=False,
        has_ratio_follow=False,
        presence_penalty=1.5,
        max_new_tokens=16256,
    ),
    TASK_EDIT: TaskProfile(
        name=TASK_EDIT,
        requires_images=True,
        has_ratio_follow=True,
        presence_penalty=0.0,
        max_new_tokens=24000,
    ),
}


def get_profile(task):
    """按任务取 profile；未知任务抛 ValueError。"""
    try:
        return PROFILES[task]
    except KeyError:
        raise ValueError(f"未知任务：{task!r}（允许 {', '.join(PROFILES)}）") from None


# ── 系统提示词 ──────────────────────────────────────────────────────────
# 通用协议为自写实现（方法吸收官方 PE 的契约语义，措辞不复制）：官方提示词随
# PE 微调权重发布，用于「本地官方PE」模式；通用/API 模式用下面精简版。
GENERIC_T2I_SYSTEM_PROMPT = """You are an expert prompt writer for the Qwen-Image 2.1 \
text-to-image model. Turn the user's brief, in any language, into one detailed, \
self-contained prompt that describes the finished image the way an observer would see \
it. Decide everything the brief leaves open; never contradict what it fixes.

Rules:
- Preserve every fact the user fixed, character for character: any text that must \
appear in the image (keep its own language, wording and punctuation), named objects, \
counts, colours, positions, and a ratio if one was given. Job instructions ("sharp \
text", "no watermark") are applied silently and never repeated in the prompt.
- If the user did not ask for readable text, do not invent titles, logos, labels, \
captions, watermarks or decorative lettering; describe the area as blank or unlettered \
when it matters.
- Be concrete: medium, style, subject, environment, composition, camera, lighting, \
colours and materials. Name real visual detail instead of quality boosters such as \
"masterpiece", "8K" or "award-winning".
- Write one flowing paragraph. Never put a ratio, a resolution or a pixel count inside \
the prompt text.
- Choose the recommended canvas from {ratios} and report it only in the "wh_ratio" field.

Output only a single valid JSON object, nothing before or after:
{{"rewritten_prompt": "<the prompt>", "wh_ratio": "<e.g. 3:2>"}}"""

GENERIC_EDIT_SYSTEM_PROMPT = """You are an expert prompt writer for the Qwen-Image 2.1 \
image-editing model. Rewrite the user's editing instruction into a precise, \
unambiguous directive that a downstream editor can execute without guessing. The \
input image(s) are the authoritative source: anchor every spatial, tonal and \
contextual claim on what is visibly there, and invent nothing.

Rules:
- Refer to the input images as <image1>, <image2>, ... in connection order. With \
several images, name each one explicitly; never merge them into "the images".
- Change exactly the attribute(s) the instruction names, at a strong and unmistakable \
degree, and hold everything else at input fidelity: faces, products, garments, \
readable text, medium and framing all survive unless the user targets them.
- Never invent readable text. Existing text stays unchanged unless the user asked to \
change it; if they supplied replacement wording, quote it exactly in its own language.
- When the user wants a new picture of the subject (a shoot, poster, collage, or a \
placement into a new scene), build the scene, lighting, composition and layout \
concretely. When they want this picture changed, constrain the change and let the \
rest stand.
- Write the directive as one flowing paragraph. Never put a ratio, a resolution or a \
pixel count inside it.
- Decide the output canvas. If the edit keeps the source framing, set "ratio_follow" \
to the canvas image's tag (e.g. "<image1>") and leave "wh_ratio" empty. If the output \
is a new composition, set "wh_ratio" to one of {ratios} and leave "ratio_follow" \
empty. Exactly one of the two carries a value.

Output only a single valid JSON object, nothing before or after:
{{"rewritten_prompt": "<the directive>", "wh_ratio": "<ratio or empty>", \
"ratio_follow": "<imageN or empty>"}}"""


def language_directive(language):
    """输出语言指令（追加在系统提示词末尾；自写，官方 PE 中文输出需覆盖其英文默认）。"""
    if language == LANGUAGE_ZH:
        return (
            "Output language (takes precedence over any earlier statement in this "
            "prompt about the description's language): write every word of the "
            "descriptive prose in Chinese. Text that the user supplied to appear "
            "inside the image keeps its own language and exact wording; never "
            "translate it."
        )
    return (
        "Output language: write every word of the descriptive prose in English. "
        "Text that the user supplied to appear inside the image keeps its own "
        "language and exact wording; never translate it."
    )


def official_system_prompt(task, language=LANGUAGE_EN):
    """官方 PE 系统提示词（原文；中文输出追加覆盖指令，英文保持逐字不变）。"""
    get_profile(task)
    base = OFFICIAL_T2I_SYSTEM_PROMPT if task == TASK_T2I else OFFICIAL_EDIT_SYSTEM_PROMPT
    if language == LANGUAGE_ZH:
        return base + "\n\n" + language_directive(LANGUAGE_ZH)
    return base


def generic_system_prompt(task, language=LANGUAGE_EN):
    """通用自写协议 + 输出语言指令。"""
    get_profile(task)
    template = GENERIC_T2I_SYSTEM_PROMPT if task == TASK_T2I else GENERIC_EDIT_SYSTEM_PROMPT
    return template.format(ratios=", ".join(RATIO_OPTIONS)) + "\n\n" + language_directive(language)


# ── 本地聊天文本 ────────────────────────────────────────────────────────
def build_local_chat_text(system, user, image_count=0, thinking=True):
    """构造 Qwen 聊天原文（本地 clip.tokenize 直送，跳过 tokenizer 默认模板）。

    对齐官方 chat_template：system/user 轮 + assistant 预填。thinking=True 预填
    `<think>\\n`（官方 enable_thinking=True）；False 预填空 think 块抑制推理。
    视觉占位按图片数量置于 user 轮开头（官方顺序：图片先于文本）。
    """
    count = max(0, int(image_count or 0))
    text = "<|im_start|>system\n" + str(system or "").strip() + "<|im_end|>\n"
    text += "<|im_start|>user\n" + VISION_BLOCK * count + str(user or "").strip() + "<|im_end|>\n"
    text += "<|im_start|>assistant\n"
    text += "<think>\n" if thinking else "<think>\n\n</think>\n\n"
    return text


# ── 答案解析 ────────────────────────────────────────────────────────────
def split_thinking(text):
    """按 `</think>` 分割为 (thinking, answer)；未闭合视为只生成思考，answer 为空。"""
    value = str(text or "")
    if "</think>" in value:
        think, _, answer = value.partition("</think>")
        if "<think>" in think:
            think = think.partition("<think>")[2]
        return think.strip(), answer.strip()
    if "<think>" in value:
        return value.partition("<think>")[2].strip(), ""
    return "", value.strip()


def _balanced_spans(text):
    """所有顶层平衡 `{...}` 片段（跳过字符串字面量内的花括号），按出现顺序。"""
    spans = []
    depth = 0
    start = -1
    in_str = False
    escaped = False
    for i, ch in enumerate(text):
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                spans.append(text[start:i + 1])
    return spans


def _read_json_string(text, start):
    """从开引号后的 start 扫描到配对结束引号，返回 (原始内容, 结束下标)；容忍裸换行。"""
    out = []
    escaped = False
    i = start
    while i < len(text):
        ch = text[i]
        if escaped:
            out.append("\\" + ch)
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == '"':
            return "".join(out), i + 1
        else:
            out.append(ch)
        i += 1
    return "".join(out), i


def _decode_json_string(raw):
    """按 JSON 字符串语义反转义；裸控制字符先转义，失败退回原始内容。"""
    fixed = raw.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")
    try:
        return json.loads('"' + fixed + '"')
    except (TypeError, ValueError):
        return raw


_FIELD_RE = re.compile(
    r'"(?P<key>rewritten_prompt|rewrited_prompt|wh_ratio|ratio_follow)"\s*:\s*"',
    re.DOTALL,
)


def _recover_fields(text):
    """严格 JSON 失败时的字段级恢复（逐字段读引号串，容忍裸换行/截断）。"""
    fields = {}
    for match in _FIELD_RE.finditer(text):
        key = match.group("key")
        if key in fields:
            continue
        raw, _ = _read_json_string(text, match.end())
        fields[key] = _decode_json_string(raw)
    return fields


def normalize_ratios(wh_ratio, ratio_follow, task, image_count=0):
    """归一比例字段：非法值清空并给出警告，官方互斥约定（恰好一个带值）。

    返回 (wh_ratio, ratio_follow, warning)。文生图恒无 ratio_follow。
    """
    profile = get_profile(task)
    wh = str(wh_ratio or "").strip()
    follow = str(ratio_follow or "").strip() if profile.has_ratio_follow else ""
    warnings = []
    if wh and wh not in RATIO_OPTIONS:
        warnings.append(f"wh_ratio 非法（{wh}），已清空")
        wh = ""
    if follow:
        match = _RATIO_FOLLOW_RE.match(follow)
        if not match or int(match.group(1)) > int(image_count or 0):
            warnings.append(f"ratio_follow 非法（{follow}），已清空")
            follow = ""
    if wh and follow:
        warnings.append("wh_ratio 与 ratio_follow 同时有值，按官方互斥约定保留 wh_ratio")
        follow = ""
    return wh, follow, "；".join(warnings)


def parse_answer(answer, task, image_count=0):
    """解析答案段，返回 {rewritten_prompt, wh_ratio, ratio_follow, parse_ok, recovered}。

    严格 JSON（逆序平衡括号扫描，容忍前置/后置散文）→ parse_ok=True；
    失败则字段级恢复（recovered=True，parse_ok 仍 False）；再失败用原文兜底。
    比例字段经 normalize_ratios 归一，warning 由调用方从返回值合并。
    """
    value = str(answer or "").strip()
    profile = get_profile(task)
    for candidate in reversed(_balanced_spans(value)):
        try:
            obj = json.loads(candidate)
        except (TypeError, ValueError):
            continue
        if not isinstance(obj, dict):
            continue
        rewritten = obj.get("rewritten_prompt") or obj.get("rewrited_prompt")
        if not isinstance(rewritten, str) or not rewritten.strip():
            continue
        wh, follow, warning = normalize_ratios(
            obj.get("wh_ratio"), obj.get("ratio_follow") if profile.has_ratio_follow else "",
            task, image_count,
        )
        return {
            "rewritten_prompt": rewritten.strip(),
            "wh_ratio": wh,
            "ratio_follow": follow,
            "parse_ok": True,
            "recovered": False,
            "warning": warning,
        }
    fields = _recover_fields(value)
    rewritten = fields.get("rewritten_prompt") or fields.get("rewrited_prompt") or ""
    if rewritten.strip():
        wh, follow, warning = normalize_ratios(
            fields.get("wh_ratio"), fields.get("ratio_follow") if profile.has_ratio_follow else "",
            task, image_count,
        )
        return {
            "rewritten_prompt": rewritten.strip(),
            "wh_ratio": wh,
            "ratio_follow": follow,
            "parse_ok": False,
            "recovered": True,
            "warning": warning,
        }
    return {
        "rewritten_prompt": value,
        "wh_ratio": "",
        "ratio_follow": "",
        "parse_ok": False,
        "recovered": False,
        "warning": "",
    }
