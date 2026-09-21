# sf_utils.prompt_rewrite 纯逻辑测试（python tests/test_prompt_rewrite.py）
# 覆盖：比例常量、消息构造（模式/比例/透明/长度/图片部分）、JSON 提取（围栏/前后缀/非法）、
# 契约校验（键集合/比例合法与一致/正文比例引号豁免/透明三语义/超长标记）、纠正消息。
import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

from sf_utils.prompt_rewrite import (  # noqa: E402
    MAX_IMAGES,
    MAX_PROMPT_CHARS,
    MODEL_RATIOS,
    RATIO_OPTIONS,
    SYSTEM_PROMPT,
    PromptRewriteError,
    build_messages,
    correction_messages,
    extract_json,
    validate_response,
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


# ── 常量 ──
check("RATIO_OPTIONS = auto + Qwen 官方 7 档", RATIO_OPTIONS == ["auto", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"])
check("MODEL_RATIOS 去 auto", MODEL_RATIOS == RATIO_OPTIONS[1:])
check("MAX_IMAGES=10", MAX_IMAGES == 10)
check("MAX_PROMPT_CHARS=12000", MAX_PROMPT_CHARS == 12000)
check("SYSTEM_PROMPT 含输出契约", '"wh_ratio"' in SYSTEM_PROMPT and "rewritten_prompt" in SYSTEM_PROMPT)

# ── build_messages ──
msgs = build_messages("一只猫", 0, "auto", 0, False)
check("两条消息", len(msgs) == 2 and msgs[0]["role"] == "system" and msgs[1]["role"] == "user")
check("system 含协议", "image prompt writer" in msgs[0]["content"] and "Walk the frame" in msgs[0]["content"])
user_text = msgs[1]["content"]
check("无图 = text-to-image", "text-to-image" in user_text and "no reference images" in user_text)
check("auto 比例交给模型", "did not fix a ratio" in user_text and "wh_ratio" in user_text)
check("brief 原文嵌入", "一只猫" in user_text)
check("无图时 content 为纯文本", isinstance(user_text, str))

msgs = build_messages("brief", 3, "16:9", 500, True, [{"type": "image_url"}])
user_content = msgs[1]["content"]
check("有图 = 编辑 + 标注范围", "Image 1 to Image 3" in user_content[0]["text"] and "image-editing" in user_content[0]["text"])
check("固定比例规则", "wh_ratio=16:9" in user_content[0]["text"] and "never write the ratio" in user_content[0]["text"])
check("透明三语义要求", "RGBA" in user_content[0]["text"] and "alpha channel" in user_content[0]["text"])
check("长度上限写入", "at or below 500 characters" in user_content[0]["text"])
check("图片部分追加在文本后", user_content[0]["type"] == "text" and user_content[1]["type"] == "image_url" and len(user_content) == 2)

msgs = build_messages("b", 0, "auto", 0, False)
check("透明关闭时禁止发明 alpha", "do not invent an alpha channel" in msgs[1]["content"])
check("长度 0 = 交给模型", "left the length open" in msgs[1]["content"])

# ── extract_json ──
check("裸 JSON", extract_json('{"rewritten_prompt": "x", "wh_ratio": "3:2"}')["wh_ratio"] == "3:2")
check("```json 围栏", extract_json('```json\n{"a": 1}\n```') == {"a": 1})
check("前后夹文字", extract_json('好的：{"a": 1} 完毕') == {"a": 1})
check("非法 JSON 返回 None", extract_json("not json") is None)
check("数组返回 None", extract_json("[1, 2]") is None)
check("空返回 None", extract_json("") is None)

# ── validate_response ──
desc = "A wide realistic photograph of a harbour at dusk. The lighting is warm."
out = validate_response(
    {"rewritten_prompt": desc, "wh_ratio": "3:2"},
    requested_ratio="auto", transparent=False, max_chars=0,
)
check("auto 合法通过", out[0] == desc and out[1] == "3:2" and out[2] == {"over_limit": False, "prompt_chars": len(desc)})

check("键集合严格", raises(
    validate_response, {"rewritten_prompt": desc}, requested_ratio="auto", transparent=False, max_chars=0))
check("空描述拒绝", raises(
    validate_response, {"rewritten_prompt": "  ", "wh_ratio": "3:2"}, requested_ratio="auto", transparent=False, max_chars=0))
check("非法比例拒绝", raises(
    validate_response, {"rewritten_prompt": desc, "wh_ratio": "21:9"}, requested_ratio="auto", transparent=False, max_chars=0))
check("固定比例不一致拒绝", raises(
    validate_response, {"rewritten_prompt": desc, "wh_ratio": "2:3"}, requested_ratio="3:2", transparent=False, max_chars=0))

ratio_in_prose = "A square badge, 1:1 in proportion. The lighting is even."
check("固定比例出现在正文拒绝", raises(
    validate_response, {"rewritten_prompt": ratio_in_prose, "wh_ratio": "1:1"},
    requested_ratio="1:1", transparent=False, max_chars=0))

quoted_ratio = 'A square badge. A small sign reads "1:1". The lighting is even.'
out = validate_response(
    {"rewritten_prompt": quoted_ratio, "wh_ratio": "1:1"},
    requested_ratio="1:1", transparent=False, max_chars=0,
)
check("引号内比例豁免", out[1] == "1:1")

time_not_ratio = "A wall clock reads 4:30 in the lower third. The lighting is even."
out = validate_response(
    {"rewritten_prompt": time_not_ratio, "wh_ratio": "4:3"},
    requested_ratio="4:3", transparent=False, max_chars=0,
)
check("数字边界避免 4:30 误判", out[1] == "4:3")

check("透明缺语义拒绝", raises(
    validate_response, {"rewritten_prompt": desc, "wh_ratio": "3:2"},
    requested_ratio="auto", transparent=True, max_chars=0))

rgba_desc = "A logo on a transparent background, delivered as an RGBA image with an alpha channel."
out = validate_response(
    {"rewritten_prompt": rgba_desc, "wh_ratio": "1:1"},
    requested_ratio="auto", transparent=True, max_chars=0,
)
check("透明三语义通过", out[1] == "1:1")

long_desc = "x" * (MAX_PROMPT_CHARS + 1)
out = validate_response(
    {"rewritten_prompt": long_desc, "wh_ratio": "3:2"},
    requested_ratio="auto", transparent=False, max_chars=100,
)
check("超长标记 over_limit", out[2]["over_limit"] is True and out[2]["prompt_chars"] == len(long_desc))

out = validate_response(
    {"rewritten_prompt": desc, "wh_ratio": "3:2"},
    requested_ratio="auto", transparent=False, max_chars=100000,
)
check("限内不标超长", out[2]["over_limit"] is False)

# ── correction_messages ──
base = build_messages("b", 0, "auto", 0, False)
fixed = correction_messages(base, "bad draft", "please fix")
check("纠正消息四段", len(fixed) == 4)
check("草稿作为 assistant", fixed[2]["role"] == "assistant" and fixed[2]["content"] == "bad draft")
check("原因写入 user", fixed[3]["role"] == "user" and "please fix" in fixed[3]["content"] and "one-line JSON" in fixed[3]["content"])
check("PromptRewriteError 是 ValueError", issubclass(PromptRewriteError, ValueError))

print(f"\nFAILURES: {len(failures)}")
if failures:
    sys.exit(1)
print("test_prompt_rewrite: all assertions passed")
