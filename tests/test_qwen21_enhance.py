# sf_utils.qwen21_enhance 纯逻辑测试（python tests/test_qwen21_enhance.py）
# 覆盖：常量/profile、官方与通用系统提示词选择（含中文覆盖指令）、本地聊天文本
# （视觉占位/thinking 预填两态）、思考链分割、答案解析（严格 JSON/多对象/字符串花括号/
# 拼写兼容/比例归一与互斥/字段级恢复/原文兜底）。
import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

from sf_utils.qwen21_enhance import (  # noqa: E402
    GENERIC_EDIT_SYSTEM_PROMPT,
    GENERIC_T2I_SYSTEM_PROMPT,
    LANGUAGE_EN,
    LANGUAGE_ZH,
    MAX_IMAGES,
    PROFILES,
    RATIO_OPTIONS,
    TASK_EDIT,
    TASK_T2I,
    VISION_BLOCK,
    build_local_chat_text,
    generic_system_prompt,
    get_profile,
    language_directive,
    normalize_ratios,
    official_system_prompt,
    parse_answer,
    split_thinking,
)
from sf_utils.qwen21_prompts import (  # noqa: E402
    OFFICIAL_EDIT_SYSTEM_PROMPT,
    OFFICIAL_T2I_SYSTEM_PROMPT,
)

failures = []


def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")


def raises(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
        return False
    except Exception:
        return True


# ── 常量 / profile ──
check("RATIO_OPTIONS = 官方 7 档", RATIO_OPTIONS == ("1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16"))
check("MAX_IMAGES=8", MAX_IMAGES == 8)
check("PROFILES 键", set(PROFILES) == {TASK_T2I, TASK_EDIT})
t2i, edit = get_profile(TASK_T2I), get_profile(TASK_EDIT)
check("t2i 官方生产参数", (t2i.presence_penalty, t2i.max_new_tokens, t2i.temperature, t2i.top_p, t2i.top_k)
      == (1.5, 16256, 1.0, 0.95, 20))
check("edit 官方生产参数", (edit.presence_penalty, edit.max_new_tokens) == (0.0, 24000))
check("t2i 无图无 ratio_follow", t2i.requires_images is False and t2i.has_ratio_follow is False)
check("edit 需图有 ratio_follow", edit.requires_images is True and edit.has_ratio_follow is True)
check("未知任务抛错", raises(get_profile, "t2v"))

# ── 系统提示词 ──
check("官方 t2i 原文", official_system_prompt(TASK_T2I) == OFFICIAL_T2I_SYSTEM_PROMPT)
check("官方 edit 原文", official_system_prompt(TASK_EDIT) == OFFICIAL_EDIT_SYSTEM_PROMPT)
check("官方 t2i 头", official_system_prompt(TASK_T2I).startswith("# Image Prompt Rewriting Expert"))
check("官方 edit 头", official_system_prompt(TASK_EDIT).startswith("# Edit Prompt Enhancer"))
check("官方 t2i 契约", '"wh_ratio"' in OFFICIAL_T2I_SYSTEM_PROMPT and "rewritten_prompt" in OFFICIAL_T2I_SYSTEM_PROMPT)
check("官方 edit 契约", "ratio_follow" in OFFICIAL_EDIT_SYSTEM_PROMPT)
zh_official = official_system_prompt(TASK_T2I, LANGUAGE_ZH)
check("官方中文追加覆盖指令", zh_official.startswith(OFFICIAL_T2I_SYSTEM_PROMPT)
      and "Chinese" in zh_official and "precedence" in zh_official)
check("官方英文不追加", official_system_prompt(TASK_T2I, LANGUAGE_EN) == OFFICIAL_T2I_SYSTEM_PROMPT)

check("通用 t2i 模板已格式化", "{ratios}" not in GENERIC_T2I_SYSTEM_PROMPT.format(ratios="x"))
gen_t2i = generic_system_prompt(TASK_T2I, LANGUAGE_EN)
gen_edit = generic_system_prompt(TASK_EDIT, LANGUAGE_EN)
check("通用 t2i 含 7 档与契约", "1:1" in gen_t2i and "16:9" in gen_t2i and '"wh_ratio"' in gen_t2i)
check("通用 edit 含 ratio_follow 互斥", "ratio_follow" in gen_edit and "Exactly one of the two" in gen_edit)
check("通用协议区分任务", "text-to-image" in gen_t2i and "image-editing" in gen_edit)
check("通用中文语言指令", "Chinese" in generic_system_prompt(TASK_T2I, LANGUAGE_ZH))
check("language_directive 英文", "English" in language_directive(LANGUAGE_EN)
      and "never" in language_directive(LANGUAGE_EN))

# ── 本地聊天文本 ──
chat = build_local_chat_text("SYS", "USER", image_count=2, thinking=True)
check("聊天 system/user 轮", chat.startswith("<|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\n"))
check("聊天视觉占位数量", chat.count(VISION_BLOCK) == 2)
check("聊天图片先于文本", chat.index(VISION_BLOCK) < chat.index("USER"))
check("聊天 thinking 预填", chat.endswith("<|im_start|>assistant\n<think>\n"))
check("聊天关闭 thinking 空块", build_local_chat_text("S", "U", thinking=False).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
check("聊天无图无占位", VISION_BLOCK not in build_local_chat_text("S", "U", image_count=0))

# ── split_thinking ──
check("完整思考块", split_thinking("reasoning\n</think>\nANSWER") == ("reasoning", "ANSWER"))
check("带 think 开标签", split_thinking("<think>r</think>A") == ("r", "A"))
check("未闭合思考", split_thinking("<think>r") == ("r", ""))
check("无思考", split_thinking("ANSWER") == ("", "ANSWER"))
check("空输入", split_thinking("") == ("", ""))
check("仅闭合标签", split_thinking("r</think>") == ("r", ""))

# ── parse_answer ──
r = parse_answer('{"rewritten_prompt": "a cat", "wh_ratio": "3:2"}', TASK_T2I)
check("t2i 严格 JSON", r["rewritten_prompt"] == "a cat" and r["wh_ratio"] == "3:2"
      and r["ratio_follow"] == "" and r["parse_ok"] and not r["recovered"])
r = parse_answer('{"rewritten_prompt": "edit", "wh_ratio": "", "ratio_follow": "<image2>"}', TASK_EDIT, 2)
check("edit ratio_follow", r["ratio_follow"] == "<image2>" and r["wh_ratio"] == "" and r["parse_ok"])
r = parse_answer('{"rewritten_prompt": "edit", "wh_ratio": "", "ratio_follow": "<image1>"}', TASK_T2I, 1)
check("t2i 丢弃 ratio_follow", r["ratio_follow"] == "" and r["parse_ok"])
r = parse_answer('前言 {"rewritten_prompt": "a {brace} prompt", "wh_ratio": "16:9"} 后记', TASK_T2I)
check("散文包裹 + 字符串花括号", r["rewritten_prompt"] == "a {brace} prompt" and r["parse_ok"])
r = parse_answer('{"rewritten_prompt": "first", "wh_ratio": "1:1"}\n{"rewritten_prompt": "last", "wh_ratio": "2:3"}', TASK_T2I)
check("多对象取最后", r["rewritten_prompt"] == "last" and r["wh_ratio"] == "2:3")
r = parse_answer('```json\n{"rewrited_prompt": "typo key", "wh_ratio": "1:1"}\n```', TASK_T2I)
check("拼写兼容 + 围栏", r["rewritten_prompt"] == "typo key" and r["parse_ok"])
r = parse_answer('{"rewritten_prompt": "p", "wh_ratio": "21:9"}', TASK_T2I)
check("非法 wh_ratio 清空并警告", r["wh_ratio"] == "" and "wh_ratio 非法" in r["warning"] and r["parse_ok"])
r = parse_answer('{"rewritten_prompt": "p", "wh_ratio": "", "ratio_follow": "<image9>"}', TASK_EDIT, 2)
check("越界 ratio_follow 清空", r["ratio_follow"] == "" and "ratio_follow 非法" in r["warning"])
r = parse_answer('{"rewritten_prompt": "p", "wh_ratio": "16:9", "ratio_follow": "<image1>"}', TASK_EDIT, 1)
check("互斥保留 wh_ratio", r["wh_ratio"] == "16:9" and r["ratio_follow"] == "" and "互斥" in r["warning"])
r = parse_answer("这里没有 JSON，只有一段普通文本", TASK_T2I)
check("无 JSON 原文兜底", r["rewritten_prompt"] == "这里没有 JSON，只有一段普通文本"
      and not r["parse_ok"] and not r["recovered"] and r["wh_ratio"] == "")
r = parse_answer('{"rewritten_prompt": "line1\nline2", "wh_ratio": "1:1"}', TASK_T2I)
check("裸换行字段级恢复", r["recovered"] and not r["parse_ok"] and r["rewritten_prompt"] == "line1\nline2"
      and r["wh_ratio"] == "1:1")
r = parse_answer('{"rewritten_prompt": "p", "wh_ratio": "1:1"', TASK_T2I)
check("截断 JSON 字段级恢复", r["recovered"] and r["rewritten_prompt"] == "p")
r = parse_answer("", TASK_T2I)
check("空答案兜底", r["rewritten_prompt"] == "" and not r["parse_ok"])

# ── normalize_ratios ──
wh, follow, warning = normalize_ratios("3:2", "", TASK_T2I, 0)
check("normalize 合法值直通", (wh, follow, warning) == ("3:2", "", ""))
wh, follow, warning = normalize_ratios("", "<image1>", TASK_EDIT, 1)
check("normalize ratio_follow 直通", (wh, follow) == ("", "<image1>") and not warning)
wh, follow, warning = normalize_ratios("", "<image0>", TASK_EDIT, 1)
check("normalize image0 非法", follow == "" and warning)

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL PASS")
