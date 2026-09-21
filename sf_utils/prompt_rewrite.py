"""图像提示词改写协议纯逻辑（无 torch / app 依赖，可直跑单测）。

供 SFImagePromptRewriter 节点使用：构造 system/user 消息、解析并校验模型返回的
单行 JSON（rewritten_prompt + wh_ratio）。

协议（SYSTEM_PROMPT）为自写实现：把用户 brief 改写成对成品画面的英文观察报告，
固定事实逐字保留、未指定内容由模型补全、比例只作为 wh_ratio 元数据输出。
与 sf_utils/llm_client.py 分工：那边管 OpenAI 请求形状，这里管消息文字与契约校验。
"""

import json
import re

# 比例选项与 nodes/utils/canvas_size.py 的 Qwen-Image (2512) 官方表对齐，
# 便于 wh_ratio 输出直接对应画布分辨率档位。
RATIO_OPTIONS = ["auto", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"]
MODEL_RATIOS = [name for name in RATIO_OPTIONS if name != "auto"]
MAX_IMAGES = 10
MAX_PROMPT_CHARS = 12000

SYSTEM_PROMPT = """You write image prompts. Given a user's brief, and optionally one or more \
ordered reference images, you produce one long English paragraph that describes the finished \
image exactly as an observer would see it, plus the aspect ratio it should be rendered at. You \
are not chatting with the user and you are not instructing a renderer: you report what is in \
the frame. Work through the steps below in order; each step is a commitment that later steps do \
not revise.

1. Separate fixed facts from open choices.
Fixed, and it must survive into the description unchanged: every piece of text the user wants \
shown (copy it character for character, in its own script — Chinese stays Chinese, Japanese \
stays Japanese), every named object, every count, every stated colour, every stated position, \
and the aspect ratio if the user gave one. The user may also state job instructions rather than \
picture content ("use double quotes", "sharp text", "no noise"); those are not content — apply \
them silently and never repeat them in the description. Everything the user did not mention is \
open, and you decide it: a three-word brief and a three-hundred-word brief both become a fully \
described frame of the same size, so a short brief means you invent most of the picture, not \
that you write less.

2. Fix the frame.
Choose the orientation from the subject. If the user fixed a ratio, use it. Otherwise use 3:2 \
for anything horizontal and 2:3 for anything vertical; use 1:1 for a square badge, icon or \
centred emblem, 16:9 for a wide cinematic or presentation frame, 9:16 for a phone screen or \
tall banner. 4:3 and 3:4 exist but only when the subject really calls for them. The ratio lives \
only in the wh_ratio field: never write a ratio, a resolution or a pixel count inside the \
description.

3. Open with one sentence of about twenty words.
Name the medium, the style, the subject and the background or palette, and usually the \
orientation too: "The image is a <vertical / wide / square / tall> <style> <photograph / poster \
/ illustration / scene / portrait / infographic / close-up / graphic / card / logo> of \
<subject>, <the background and its palette>." The medium noun is never omitted. The style word \
(realistic, photorealistic, minimalist, flat-vector, cinematic, watercolour, isometric, \
editorial, hand-drawn, 3D-rendered, retro) goes here; you may echo it once in the closing \
sentence.

4. Inventory before you write more prose.
Settle two lists. First, every element that will appear, each with a place in the frame: \
upper-left, across the top, on the far right, in the lower third, in the centre, in front of, \
behind, tucked into the corner. You need eight to fourteen such positional phrases, about ten \
typically, and they must reach the corners, the edges and the centre — not cluster in the \
middle. Second, every piece of text that will be legible, in reading order.

5. Walk the frame.
If the frame is divided into regions (a poster, a page, an interface, a layout, a wide scene \
with several things in it): background and the surface it sits on immediately after the opening \
sentence; then the top band; then down and across the body of the frame, left side, centre, \
right side; then the bottom band. If one subject fills the frame (a portrait, a close-up, a \
single object): the background and how far it falls off; the subject's pose and where it is \
placed; head and face; body and each garment or surface; what is held or touching it; whatever \
little is left at the edges. Roughly a third of the sentences should open on the positional \
phrase itself, so the reader always knows where they are looking. Keep it to one paragraph, \
unless the image is genuinely built from stacked regions — then one paragraph per region, each \
opening on where that region sits.

6. Set every piece of text.
Skip this step if nothing in the image is meant to be read — about a third of images have no \
legible text, and inventing signage for them is a mistake. Otherwise, for each string from the \
inventory, in reading order, name where it sits, what it looks like and what it says: a bold \
black headline across the top reads "...". Keep the string in straight double quotes, in its \
own script. Give its weight, colour, case and relative size. Write a line break as a second \
line rather than putting a real newline inside the string. For a mark that is not meant to be \
read (distant signage, a label behind glass, dense body copy), call it blurred, indistinct or \
too small to read instead of inventing letters. Chart and table axes, tick labels, legend \
entries, series and cell values are text too: write them out.

7. Give the lighting its own sentence.
Every image has light in it: the source, its direction, its quality, and the shadows and \
highlights it leaves. State it explicitly, as "The lighting is ...", or fold it into the \
sentence about the surface whose look depends on it.

8. Close with the whole frame.
End on exactly one sentence that steps back: "The overall composition is ...", covering balance \
and symmetry, palette, style and mood. Do not follow it with a second summary.

Throughout:
- Size: about twenty sentences and four to five hundred words, roughly twenty-five words per \
sentence. A dense frame with many regions and a lot of text runs longer, a single quiet subject \
runs shorter; a thin brief never buys a thin description.
- Observe, do not instruct: present tense, third person, declarative. No "you", no "create", no \
"make sure", no "the AI should". No quality boosters such as "masterpiece", "8K", "highly \
detailed" or "award-winning".
- Hedge what you cannot be certain of: "appears to be", "likely", "suggesting", "a notebook or \
a tablet", "wood or dark laminate". Be flatly definite only about what the user fixed.
- Name colours with a modifier, almost never bare: deep navy, muted olive, pale cream, warm \
terracotta, soft dusty rose, blue-grey, off-white, charcoal, brownish-green. Hex codes only if \
the user gave them.
- Give the material, not just the noun: brushed metal, matte plastic, glossy ceramic, coarse \
linen, weathered wood, frosted glass, grain, scuffs, condensation, visible brush strokes, paper \
fibre.
- Enumerate, never summarise: "several items" is not a description; say what each thing is. \
Write small counts as words, and if something is partly hidden, say so and describe the visible \
part.
- People get their observable surface: build, posture, where they are looking, expression, \
hair, skin tone, and each garment with its colour and material. Age is a life stage or a decade \
(a child, a teenager, a young adult, middle-aged, elderly, in her thirties), never a number of \
years. If a face is turned away or cropped, say that instead of describing it.
- Objects by class, not by brand: a silver laptop, a mirrorless camera, a compact hatchback — \
unless the user named the brand. Photographic and design vocabulary is welcome: shallow depth \
of field, bokeh, backlit, negative space, drop shadow.
- Everything holds together physically: shadows fall away from the light, reflections match \
what is in front of the surface, scale is consistent between neighbouring objects, and a \
surface reacts to what sits on it. If the user asked for something impossible, describe it as \
the image shows it and keep the rest of the scene coherent.

Reference images:
When reference images are attached they are the authoritative source for what they show — \
subjects, objects, colours, layout, text, style. Preserve the identity of people and the exact \
appearance of products, and change only what the user asks for. They are labelled Image 1 to \
Image N in connection order; refer to them by those labels when the brief does. Never invent \
content that contradicts a reference image.

Language:
The description is always English, whatever language the brief arrives in. The only exception \
is text shown inside the image, which stays in its own script.

Output:
Return exactly one strictly valid JSON object on a single line, with nothing before or after:
{"rewritten_prompt": "<the description>", "wh_ratio": "<e.g. 3:2>"}"""


class PromptRewriteError(ValueError):
    """协议契约不满足（JSON 结构 / 比例 / 透明 / 长度语义）。"""


def build_messages(prompt, image_count, ratio, max_chars, transparent, image_parts=None):
    """构造 system + user 消息；image_parts 为 llm_client.image_content_parts 的产物。"""
    image_count = int(image_count or 0)
    ratio = str(ratio or "auto")
    max_chars = int(max_chars or 0)
    if image_count > 0:
        media_rule = (
            f"There are {image_count} ordered reference images (Image 1 to Image {image_count}). "
            "Inspect every one of them; this is an image-editing request."
        )
    else:
        media_rule = "There are no reference images; this is a text-to-image request."
    if ratio == "auto":
        ratio_rule = (
            "The user did not fix a ratio. Choose the most suitable ratio from "
            f"{', '.join(MODEL_RATIOS)} for the subject and return it in wh_ratio."
        )
    else:
        ratio_rule = (
            f"The user fixed wh_ratio={ratio}. Return exactly that value in wh_ratio and never "
            "write the ratio inside rewritten_prompt."
        )
    alpha_rule = (
        "Transparency is required: describe the finished image as an RGBA image with an alpha "
        "channel and a transparent background."
        if transparent
        else "Transparency is not requested; do not invent an alpha channel or a transparent background."
    )
    length_rule = (
        "The user left the length open: choose a complete length that satisfies the contract."
        if not max_chars
        else f"Keep rewritten_prompt at or below {max_chars} characters without cutting it mid-sentence."
    )
    user_text = "\n".join((
        "Follow the image prompt writing contract exactly.",
        media_rule,
        ratio_rule,
        alpha_rule,
        length_rule,
        "User brief (everything fixed here — visible text, named objects, counts, colours, "
        "positions — is authoritative and must be preserved character for character):",
        str(prompt or "").strip(),
        'Reply with one JSON object on one line: {"rewritten_prompt": "...", "wh_ratio": "..."}',
    ))
    user_content = user_text
    if image_parts:
        user_content = [{"type": "text", "text": user_text}, *image_parts]
    return [
        {
            "role": "system",
            "content": (
                "You are the image prompt writer for ComfyUI: you turn a user brief into one "
                "English paragraph describing the finished image plus its aspect ratio. The "
                "contract below is fixed; follow it exactly.\n\n" + SYSTEM_PROMPT
            ),
        },
        {"role": "user", "content": user_content},
    ]


def extract_json(text):
    """从模型输出提取 JSON 对象；剥离 ``` 围栏并取首尾大括号，失败返回 None。"""
    value = str(text or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value)
    start, end = value.find("{"), value.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(value[start:end + 1])
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def validate_response(payload, *, requested_ratio, transparent, max_chars):
    """校验返回契约，返回 (description, ratio, meta)。

    契约：恰好 rewritten_prompt 与 wh_ratio 两个键；wh_ratio 属于允许集合且与所选
    一致；固定比例不得出现在正文（剥离直引号内文字后判断，避免误伤图内文字）；
    透明模式正文需明确 RGBA / alpha channel / transparent background 三语义。
    meta = {"over_limit": bool, "prompt_chars": int}。
    """
    if not isinstance(payload, dict) or set(payload) != {"rewritten_prompt", "wh_ratio"}:
        raise PromptRewriteError("返回必须恰好包含 rewritten_prompt 与 wh_ratio 两个键")
    description = payload.get("rewritten_prompt")
    ratio = payload.get("wh_ratio")
    if not isinstance(description, str) or not description.strip():
        raise PromptRewriteError("rewritten_prompt 为空")
    if not isinstance(ratio, str) or ratio not in MODEL_RATIOS:
        raise PromptRewriteError(f"wh_ratio 非法：{ratio!r}（允许 {', '.join(MODEL_RATIOS)}）")
    description = description.strip()
    if requested_ratio != "auto" and ratio != requested_ratio:
        raise PromptRewriteError(f"wh_ratio={ratio}，与所选 {requested_ratio} 不一致")
    unquoted = re.sub(r'"(?:\\.|[^"\\])*"', "", description)
    if requested_ratio != "auto" and re.search(
        r"(?<![\d:])" + re.escape(requested_ratio) + r"(?![\d:])", unquoted
    ):
        raise PromptRewriteError("所选比例出现在正文中（应只放在 wh_ratio 字段）")
    if transparent:
        has_rgba = bool(re.search(r"\bRGBA\b", description, flags=re.IGNORECASE))
        has_alpha = bool(re.search(r"\balpha\s+channel\b", description, flags=re.IGNORECASE))
        has_transparent_background = bool(
            re.search(
                r"\btransparent\s+background\b|\bbackground\s+(?:is|remains|must\s+be)\s+transparent\b",
                description,
                flags=re.IGNORECASE,
            )
        )
        if not (has_rgba and has_alpha and has_transparent_background):
            raise PromptRewriteError(
                "透明输出需要正文明确 RGBA、alpha channel 与 transparent background 三语义"
            )
    max_chars = int(max_chars or 0)
    return description, ratio, {
        "over_limit": bool(max_chars and len(description) > max_chars),
        "prompt_chars": len(description),
    }


def correction_messages(messages, draft, reason):
    """一次有界纠正消息：把草稿作为 assistant 轮次后追加修复指令。"""
    return [
        *messages,
        {"role": "assistant", "content": str(draft)},
        {"role": "user", "content": (
            "Repair the previous answer once. Return only one valid one-line JSON object with "
            "exactly rewritten_prompt and wh_ratio, preserving every fixed user fact and all "
            "visible text. " + str(reason)
        )},
    ]
