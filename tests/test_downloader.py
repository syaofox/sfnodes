# downloader.download_model 防御测试（H10）：
#  - 网络异常被吞并返回 False（不炸调用方）
#  - 下载中断不留下被 is_file() 误判为"已下载"的半成品
#  - 成功时原子替换目标文件
# mock：requests（不联网）
# 运行：python tests/test_downloader.py
import importlib.util
import os
import sys
import tempfile
import types

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

failures = []

def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")

# ── mock requests ──
requests = types.ModuleType("requests")

class _Resp:
    def __init__(self, status=200, chunks=None, headers=None, exc=None):
        self.status_code = status
        self._chunks = chunks or []
        self.headers = headers or {"content-length": "0"}
        self._exc = exc
    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(str(self.status_code))
    def iter_content(self, block_size):
        if self._exc:
            raise self._exc
        for c in self._chunks:
            yield c

class _ReqModule:
    def __init__(self):
        self.calls = []
        self.next = None
        self._exc = None
    def get(self, url, stream=False, timeout=None):
        self.calls.append((url, timeout))
        if self._exc:
            raise self._exc
        return self.next

requests.exceptions = types.ModuleType("requests.exceptions")
requests.exceptions.RequestException = type("RequestException", (Exception,), {})
requests.exceptions.HTTPError = type("HTTPError", (requests.exceptions.RequestException,), {})

_req = _ReqModule()
requests.get = _req.get
sys.modules["requests"] = requests
sys.modules["requests.exceptions"] = requests.exceptions

# ── mock tqdm ──
tqdm = types.ModuleType("tqdm")
class _Bar:
    def __init__(self, *a, **k): pass
    def update(self, n): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass
tqdm.tqdm = _Bar
sys.modules["tqdm"] = tqdm

# ── mock logger ──
for name, path in [("sfnodes", "."), ("sfnodes.sf_utils", "sf_utils")]:
    if name not in sys.modules:
        m = types.ModuleType(name)
        m.__path__ = [os.path.join(root, path)]
        sys.modules[name] = m

logger_mod = types.ModuleType("sfnodes.sf_utils.logger")
import logging
logger_mod.get_logger = lambda name: logging.getLogger(name)
sys.modules["sfnodes.sf_utils.logger"] = logger_mod

spec = importlib.util.spec_from_file_location("sfnodes.sf_utils.downloader", os.path.join(root, "sf_utils", "downloader.py"))
dl = importlib.util.module_from_spec(spec)
sys.modules["sfnodes.sf_utils.downloader"] = dl
spec.loader.exec_module(dl)

with tempfile.TemporaryDirectory() as tmp:
    # 1) 成功下载：目标文件出现，无 .part 残留
    _req.next = _Resp(status=200, chunks=[b"hello", b" world"], headers={"content-length": "11"})
    ok = dl.download_model("https://example.com/model.safetensors", tmp, "model.safetensors")
    check("成功下载返回 True", ok is True)
    check("目标文件内容正确", open(os.path.join(tmp, "model.safetensors"), "rb").read() == b"hello world")
    check("无 .part 残留", not os.path.exists(os.path.join(tmp, "model.safetensors.part")))
    check("get 带 timeout", _req.calls[0][1] is not None)

    # 2) 已存在：直接返回 True 不请求
    _req.calls.clear()
    ok = dl.download_model("https://example.com/model.safetensors", tmp, "model.safetensors")
    check("已存在直接 True", ok is True and len(_req.calls) == 0)

    # 3) 流中断：返回 False 且不留半成品（下次 is_file 不误判）
    _req.next = _Resp(status=200, chunks=[b"partial"], exc=requests.exceptions.RequestException("conn reset"))
    ok = dl.download_model("https://example.com/model2.safetensors", tmp, "model2.safetensors")
    check("流中断返回 False", ok is False)
    check("无半成品文件", not os.path.exists(os.path.join(tmp, "model2.safetensors")) and not os.path.exists(os.path.join(tmp, "model2.safetensors.part")))

    # 4) 连接异常（requests.get 抛错）：返回 False 不炸
    _req._exc = requests.exceptions.RequestException("DNS fail")
    ok = dl.download_model("https://example.com/model3.safetensors", tmp, "model3.safetensors")
    check("连接异常返回 False", ok is False)
    _req._exc = None

    # 5) HTTP 错误：返回 False
    _req.next = _Resp(status=404)
    ok = dl.download_model("https://example.com/model4.safetensors", tmp, "model4.safetensors")
    check("404 返回 False", ok is False)

    # ── HF resolve URL → huggingface_hub（方案 A：官方缓存 + 复制到约定路径）──
    hf = types.ModuleType("huggingface_hub")
    hf_state = {"cached": None, "exc": None, "calls": 0}
    def _fake_hf_download(repo_id=None, filename=None, revision=None, **kw):
        hf_state["calls"] += 1
        if hf_state["exc"]:
            raise hf_state["exc"]
        return hf_state["cached"]
    hf.hf_hub_download = _fake_hf_download
    sys.modules["huggingface_hub"] = hf

    # parse_hf_url 解析
    check("parse_hf_url 基础", dl.parse_hf_url("https://huggingface.co/Syaofox/sfnodes/resolve/main/xseg_1.onnx") == ("Syaofox/sfnodes", "main", "xseg_1.onnx"))
    check("parse_hf_url 子目录", dl.parse_hf_url("https://huggingface.co/Syaofox/sfnodes/resolve/main/antelopev2/1k3d68.onnx") == ("Syaofox/sfnodes", "main", "antelopev2/1k3d68.onnx"))
    check("parse_hf_url 非 main rev", dl.parse_hf_url("https://huggingface.co/Syaofox/sfnodes/resolve/v1.0.0/model.onnx") == ("Syaofox/sfnodes", "v1.0.0", "model.onnx"))
    check("parse_hf_url 非 HF URL", dl.parse_hf_url("https://example.com/model.onnx") is None)
    check("parse_hf_url 非 str", dl.parse_hf_url(None) is None)

    # HF 成功：缓存文件 → 复制到约定路径，不请求 requests
    cached_file = os.path.join(tmp, "_hf_cache", "model.onnx")
    os.makedirs(os.path.dirname(cached_file), exist_ok=True)
    with open(cached_file, "wb") as f:
        f.write(b"hf model bytes")
    hf_state["cached"] = cached_file
    hf_state["exc"] = None
    _req.calls.clear()
    ok = dl.download_model("https://huggingface.co/Syaofox/sfnodes/resolve/main/model5.onnx", tmp, "model5.onnx")
    check("HF 下载成功返回 True", ok is True)
    check("HF 落盘位置正确", open(os.path.join(tmp, "model5.onnx"), "rb").read() == b"hf model bytes")
    check("HF 路径不请求 requests", len(_req.calls) == 0)

    # HF 失败：返回 False 且不回退 requests
    hf_state["exc"] = Exception("network down")
    _req.calls.clear()
    ok = dl.download_model("https://huggingface.co/Syaofox/sfnodes/resolve/main/model6.onnx", tmp, "model6.onnx")
    check("HF 失败返回 False", ok is False)
    check("HF 失败不回退 requests", len(_req.calls) == 0)
    hf_state["exc"] = None

    # HF 目标已存在：直接 True，不调 hf_hub_download
    hf_state["calls"] = 0
    ok = dl.download_model("https://huggingface.co/Syaofox/sfnodes/resolve/main/model5.onnx", tmp, "model5.onnx")
    check("HF 目标已存在直接 True", ok is True and hf_state["calls"] == 0)

if failures:
    print(f"\n{failures}")
    sys.exit(1)
print("\nALL PASS")
