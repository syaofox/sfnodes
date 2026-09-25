# sf_utils/text_chunk.py 纯逻辑测试（python tests/test_text_chunk.py）
# 覆盖：分句（中英标点/换行/后置引号/无标点）、次级切分、超长单元硬切与孤标点合并、
# 贪心打包（余量/CJK 与英文拼接空格）、空输入。
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

failures = []


def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")


spec = importlib.util.spec_from_file_location(
    "text_chunk", os.path.join(root, "sf_utils", "text_chunk.py")
)
TC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TC)


def est(text):
    """假估计：0.1 秒/字符（便于手算打包结果）。"""
    return len(str(text)) * 0.1


# ── 分句 ──
check("中文句末切句", TC.split_sentences("你好。今天天气不错！") == ["你好。", "今天天气不错！"])
check("保留后置引号", TC.split_sentences("他说：“你好。”然后走了。") == ["他说：“你好。”", "然后走了。"])
check("英文句末切句", TC.split_sentences("Hello. How are you?") == ["Hello.", "How are you?"])
check("换行硬切", TC.split_sentences("第一行\n第二行") == ["第一行", "第二行"])
check("空输入", TC.split_sentences("  \n \n") == [])
check("无标点整段", TC.split_sentences("没有标点的一句话") == ["没有标点的一句话"])
check("次级切分", TC.split_clauses("前半，后半；尾巴") == ["前半，", "后半；", "尾巴"])

# ── 打包 ──
chunks = TC.pack_chunks("你好。世界。再见。", est, max_seconds=0.45, margin=0.0)
check("按上限逐句打包", [c["text"] for c in chunks] == ["你好。", "世界。", "再见。"])
check("段估计时长", all(abs(c["seconds"] - 0.3) < 1e-9 for c in chunks))

chunks = TC.pack_chunks("Hello. How are you?", est, max_seconds=5.0, margin=0.0)
check("英文句间补空格", [c["text"] for c in chunks] == ["Hello. How are you?"])

chunks = TC.pack_chunks("Hello. 你好。World.", est, max_seconds=1.0, margin=0.0)
check("CJK 直接相连", [c["text"] for c in chunks] == ["Hello.你好。", "World."])

chunks = TC.pack_chunks("这是一句非常长的话，没有任何句号", est, max_seconds=1.0, margin=0.0)
check("超长句按逗号切", [c["text"] for c in chunks] == ["这是一句非常长的话，", "没有任何句号"])

chunks = TC.pack_chunks("啊" * 25, est, max_seconds=1.0, margin=0.0)
check("无标点超长按字符硬切", [len(c["text"]) for c in chunks] == [10, 10, 5])

chunks = TC.pack_chunks("你好。世界。", est, max_seconds=0.25, margin=0.0)
check("硬切孤标点并入前片", [c["text"] for c in chunks] == ["你好。", "世界。"])

chunks = TC.pack_chunks("一二三四五六七八九十。", est, max_seconds=1.0, margin=0.2)
check("余量收紧上限", [c["text"] for c in chunks] == ["一二三四五六七八", "九十。"])

check("空文本返回空列表", TC.pack_chunks("", est, 10.0) == [])

if failures:
    print(f"\n{len(failures)} 项失败：")
    for name in failures:
        print("  -", name)
    sys.exit(1)
print("\nOK")
