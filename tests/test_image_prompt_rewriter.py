# SFImagePromptRewriter 后端测试（python tests/test_image_prompt_rewriter.py）
# 覆盖：INPUT_TYPES/元数据、空 prompt/非法比例/超 10 图校验、无图与多图消息构造
# （有序 Image N + image_url 部分数）、temperature/seed/cache_key_extra 透传、
# 结构失败一次纠正（温度 0.1）、纠正失败保留原文、超长纠正与保留、报告字段。
# 网络调用打桩（chat_completion_sync），不发真实请求；图片走真实 numpy/PIL 编码。
import json
import os
import sys
import types

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

# ── 最小 mock：folder_paths（llm_client 设置读取兜底）──
fp = types.ModuleType("folder_paths")
fp.get_user_directory = lambda: "/tmp/sfnodes_test_user"
sys.modules["folder_paths"] = fp

import numpy as np  # noqa: E402

from nodes.text import image_prompt_rewriter as mod  # noqa: E402
from nodes.text.image_prompt_rewriter import SFImagePromptRewriter  # noqa: E402

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


NODE = SFImagePromptRewriter()


def base_args(**overrides):
    args = dict(
        prompt="a cat on a chair", wh_ratio="auto", max_output_chars=0,
        transparent_alpha=False, temperature=0.2, max_tokens=4096,
        vision_megapixels=1.0, detail="auto", seed=7, send_seed=False,
    )
    args.update(overrides)
    return args


# ── INPUT_TYPES / 元数据 ──
it = SFImagePromptRewriter.INPUT_TYPES()
req = it["required"]
check("prompt 多行", req["prompt"][0] == "STRING" and req["prompt"][1].get("multiline") is True)
check("wh_ratio 选项", req["wh_ratio"][0] == ["auto", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"])
check("max_output_chars 0-12000", req["max_output_chars"][0] == "INT" and req["max_output_chars"][1]["max"] == 12000)
check("透明布尔", req["transparent_alpha"][0] == "BOOLEAN" and req["transparent_alpha"][1]["default"] is False)
check("seed 带 control_after_generate", req["seed"][1].get("control_after_generate") is True)
check("send_seed 默认 False", req["send_seed"][0] == "BOOLEAN" and req["send_seed"][1]["default"] is False)
check("optional image_1..10", set(it["optional"].keys()) == {f"image_{i}" for i in range(1, 11)})
check("optional 全 IMAGE", all(v[0] == "IMAGE" for v in it["optional"].values()))
check("RETURN_TYPES", SFImagePromptRewriter.RETURN_TYPES == ("STRING", "STRING", "STRING"))
check("RETURN_NAMES", SFImagePromptRewriter.RETURN_NAMES == ("rewritten_prompt", "wh_ratio", "report_json"))
check("FUNCTION", SFImagePromptRewriter.FUNCTION == "rewrite")
check("CATEGORY", SFImagePromptRewriter.CATEGORY == "sfnodes/text")
check("DESCRIPTION 非空", bool(SFImagePromptRewriter.DESCRIPTION))

# ── 输入校验 ──
check("空 prompt 抛错", raises(NODE.rewrite, **base_args(prompt="  ")))
check("非法比例抛错", raises(NODE.rewrite, **base_args(wh_ratio="21:9")))
check("超 10 图抛错（批次逐帧计数）", raises(NODE.rewrite, **base_args(image_1=np.zeros((11, 2, 2, 3), dtype="float32"))))

# ── 打桩调用 ──
calls = []
real_chat = mod.chat_completion_sync
real_cfg = mod.get_llm_config
queue = []


def fake_chat(config, messages, **kwargs):
    calls.append({"config": config, "messages": messages, "kwargs": kwargs})
    if queue:
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
    return '{"rewritten_prompt": "A wide realistic photograph of a calm harbour at dusk. The lighting is warm and low.", "wh_ratio": "3:2"}'


mod.chat_completion_sync = fake_chat
mod.get_llm_config = lambda: {"provider": "deepseek", "base_url": "u", "model": "m", "api_key": "k"}

try:
    # 成功（无图）
    calls.clear(); queue.clear()
    rewritten, ratio, report = NODE.rewrite(**base_args())
    check("无图输出描述", rewritten.startswith("A wide realistic photograph"))
    check("无图输出比例", ratio == "3:2")
    rep = json.loads(report)
    check("report structured", rep["structured_response"] is True and rep["correction_calls"] == 0)
    check("report image_count 0", rep["image_count"] == 0 and rep["wh_ratio"] == "3:2")
    check("report 不含密钥", "api_key" not in report and "k" != rep.get("model"))
    check("一次请求", len(calls) == 1)
    check("config 来自 get_llm_config", calls[0]["config"]["api_key"] == "k")
    check("temperature 透传", calls[0]["kwargs"]["temperature"] == 0.2)
    check("max_tokens 透传", calls[0]["kwargs"]["max_tokens"] == 4096)
    check("send_seed=False 不下发", calls[0]["kwargs"]["seed"] is None)
    check("seed 进缓存键", calls[0]["kwargs"]["cache_key_extra"] == (7,))
    user_text = calls[0]["messages"][1]["content"]
    check("无图 user 为纯文本", isinstance(user_text, str) and "text-to-image" in user_text)

    # 多图（image_1 批次 2 帧 + image_2 1 帧，按序）
    calls.clear(); queue.clear()
    rewritten, ratio, report = NODE.rewrite(**base_args(
        image_1=np.zeros((2, 4, 6, 3), dtype="float32"),
        image_2=np.ones((1, 4, 6, 3), dtype="float32"),
    ))
    rep = json.loads(report)
    check("多图 image_count=3", rep["image_count"] == 3)
    content = calls[0]["messages"][1]["content"]
    check("1 文本 + 3 图部分", len(content) == 4 and content[0]["type"] == "text" and all(p["type"] == "image_url" for p in content[1:]))
    check("图按序为 data URL", all(p["image_url"]["url"].startswith("data:image/jpeg;base64,") for p in content[1:]))
    check("system 标注 Image 1..3", "Image 1 to Image 3" in content[0]["text"])

    # 结构失败一次纠正（第二次温度 0.1）
    calls.clear(); queue.clear()
    queue.append("这不是 JSON")
    queue.append('{"rewritten_prompt": "A vertical minimalist poster of a lone tree. The lighting is soft.", "wh_ratio": "2:3"}')
    rewritten, ratio, report = NODE.rewrite(**base_args())
    rep = json.loads(report)
    check("纠正后结构化成功", rep["structured_response"] is True and rep["correction_calls"] == 1)
    check("纠正结果生效", rewritten.startswith("A vertical minimalist poster") and ratio == "2:3")
    check("纠正温度 0.1", calls[1]["kwargs"]["temperature"] == 0.1)
    check("纠正消息含草稿", calls[1]["messages"][2]["role"] == "assistant" and calls[1]["messages"][2]["content"] == "这不是 JSON")

    # 两次都失败：保留原文 + 空比例 + report 记错
    calls.clear(); queue.clear()
    queue.append("完全不合法")
    queue.append("还是不合法")
    rewritten, ratio, report = NODE.rewrite(**base_args())
    rep = json.loads(report)
    check("失败保留首稿原文", rewritten == "完全不合法" and ratio == "")
    check("失败 structured=False", rep["structured_response"] is False)
    check("失败记错", bool(rep["error"]) and rep["correction_calls"] == 1)

    # 空原文才抛错
    calls.clear(); queue.clear()
    queue.append("")
    queue.append("")
    check("空响应抛错", raises(NODE.rewrite, **base_args()))

    # 超长：一次长度纠正
    calls.clear(); queue.clear()
    queue.append('{"rewritten_prompt": "' + "x" * 300 + '", "wh_ratio": "3:2"}')
    queue.append('{"rewritten_prompt": "short enough", "wh_ratio": "3:2"}')
    rewritten, ratio, report = NODE.rewrite(**base_args(max_output_chars=200))
    rep = json.loads(report)
    check("超长纠正后达标", rewritten == "short enough" and rep["over_limit"] is False and rep["correction_calls"] == 1)
    check("长度原因写入纠正", "at or below 200 characters" in calls[1]["messages"][3]["content"])

    # 超长纠正失败：保留首次有效结果 + 标记 over_limit + 记错
    calls.clear(); queue.clear()
    long_desc = "y" * 300
    queue.append(json.dumps({"rewritten_prompt": long_desc, "wh_ratio": "3:2"}))
    queue.append("bad repair")
    rewritten, ratio, report = NODE.rewrite(**base_args(max_output_chars=200))
    rep = json.loads(report)
    check("纠正失败保留原结果", rewritten == long_desc and ratio == "3:2")
    check("保留结果标 over_limit", rep["over_limit"] is True and rep["structured_response"] is True)
    check("纠正失败原因入 report", "长度纠正未生效" in rep["error"])

    # 固定比例：正文含比例触发纠正
    calls.clear(); queue.clear()
    queue.append('{"rewritten_prompt": "A square badge in 1:1 proportion. The lighting is even.", "wh_ratio": "1:1"}')
    queue.append('{"rewritten_prompt": "A square badge. The lighting is even.", "wh_ratio": "1:1"}')
    rewritten, ratio, report = NODE.rewrite(**base_args(wh_ratio="1:1"))
    rep = json.loads(report)
    check("比例入正文被纠正", ratio == "1:1" and rep["correction_calls"] == 1 and "1:1" not in rewritten)

    # send_seed=True 下发 seed
    calls.clear(); queue.clear()
    NODE.rewrite(**base_args(send_seed=True, seed=42))
    check("send_seed=True 下发", calls[0]["kwargs"]["seed"] == 42 and calls[0]["kwargs"]["cache_key_extra"] == (42,))
finally:
    mod.chat_completion_sync = real_chat
    mod.get_llm_config = real_cfg

print(f"\nFAILURES: {len(failures)}")
if failures:
    sys.exit(1)
print("test_image_prompt_rewriter: all assertions passed")
