"""为 VHS_LoadVideo 追加 filename 输出（运行时补丁，不修改 ComfyUI-VideoHelperSuite）。

ComfyUI-VideoHelperSuite 的 4 个 LoadVideo 节点（VHS_LoadVideo / VHS_LoadVideoPath /
VHS_LoadVideoFFmpeg / VHS_LoadVideoFFmpegPath）没有文件名输出（上游 main 同样没有），
而任务约束不允许改动 VHS 文件。这里在 sfnodes 导入时对节点注册表里的类对象做原地补丁：

- `RETURN_TYPES` / `RETURN_NAMES` 末尾追加 `STRING` / `filename`——工作流按槽索引
  连线，末尾追加不破坏已有连接；`server.node_info()` 每次 /object_info 请求都惰性读
  类属性（server.py `node_info()`），所以补丁在请求前生效即可让前端建出新槽。
- 包装 `load_video`：调用原方法前取 `kwargs["video"]` 原始 widget 值，返回时追加为
  第 5 个输出（upload 变体是 `sub/clip.mp4` 相对名，Path 变体是路径/URL 原样）。

守卫：仅补丁 `__module__` 含 videohelpersuite 的类；已有 filename 输出、幂等标记、
FUNCTION 不符一律跳过；filename 计算异常回退空串/`str(raw)`，绝不阻断 VHS 执行。

加载时序：自定义节点按 `os.listdir` 顺序加载，sfnodes 不保证在 VHS 之后，故
`install()` 先查 `nodes.NODE_CLASS_MAPPINGS`，缺 key 时 `install_deferred()` 在事件
循环上有限重试。VHS 升级改变类结构时补丁因守卫静默跳过（告警），不影响 VHS 运行。
"""

import functools

from .logger import get_logger

logger = get_logger("sfnodes.vhs_filename")

_MARK = "_sf_filename_output_patched"
_TARGET_KEYS = (
    "VHS_LoadVideo",
    "VHS_LoadVideoPath",
    "VHS_LoadVideoFFmpeg",
    "VHS_LoadVideoFFmpegPath",
)
_OUTPUT_TYPE = "STRING"
_OUTPUT_NAME = "filename"
_MODULE_TAG = "videohelpersuite"


def _extract_filename(value):
    """filename 输出语义 = 节点上显示的原始值：字符串原样，None 给空串，其余转 str。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _make_load_video(original):
    @functools.wraps(original)
    def load_video(self, **kwargs):
        raw = kwargs.get("video")
        result = original(self, **kwargs)
        try:
            filename = _extract_filename(raw)
        except Exception:
            filename = ""
        if isinstance(result, tuple):
            return (*result, filename)
        return (result, filename)

    return load_video


def patch_class(cls):
    """给单个 VHS LoadVideo 类追加 filename 输出；实际补丁成功返回 True。"""
    if getattr(cls, _MARK, False):
        return False
    try:
        if _MODULE_TAG not in (getattr(cls, "__module__", "") or ""):
            return False
        if getattr(cls, "FUNCTION", None) != "load_video":
            return False
        return_names = getattr(cls, "RETURN_NAMES", None)
        if return_names is not None and _OUTPUT_NAME in tuple(return_names):
            setattr(cls, _MARK, True)
            return False
        return_types = tuple(getattr(cls, "RETURN_TYPES", ()) or ())
        if not return_types:
            return False
        if return_names is not None and len(return_names) != len(return_types):
            return False
        original = cls.__dict__.get("load_video") or getattr(cls, "load_video", None)
        if not callable(original):
            return False
        cls.RETURN_TYPES = return_types + (_OUTPUT_TYPE,)
        if return_names is not None:
            cls.RETURN_NAMES = tuple(return_names) + (_OUTPUT_NAME,)
        output_is_list = getattr(cls, "OUTPUT_IS_LIST", None)
        if isinstance(output_is_list, tuple) and len(output_is_list) == len(return_types):
            cls.OUTPUT_IS_LIST = tuple(output_is_list) + (False,)
        cls.load_video = _make_load_video(original)
        setattr(cls, _MARK, True)
        return True
    except Exception as exc:
        logger.warning("VHS filename 输出注入失败：%s", exc)
        return False


def _registry_or_none():
    try:
        import nodes as comfy_nodes
        return comfy_nodes.NODE_CLASS_MAPPINGS
    except Exception as exc:
        logger.warning("VHS filename 输出补丁暂不可用（无法读取节点注册表）：%s", exc)
        return None


def _vhs_ready(registry):
    try:
        return all(key in registry for key in _TARGET_KEYS)
    except Exception:
        return False


def install(registry=None):
    """扫描注册表补丁 4 个 LoadVideo 类；返回实际补丁数（VHS 未加载/已补丁时为 0）。"""
    if registry is None:
        registry = _registry_or_none()
        if registry is None:
            return 0
    count = 0
    for key in _TARGET_KEYS:
        cls = registry.get(key)
        if cls is not None and patch_class(cls):
            count += 1
    if count > 0:
        logger.info("VHS LoadVideo filename 输出补丁：%d 个节点", count)
    return count


def install_deferred(registry=None, loop=None, retries=60, interval=0.5):
    """VHS 尚未加载（key 缺失）时在事件循环上有限重试安装；返回是否已就绪。"""
    if registry is None:
        registry = _registry_or_none()
        if registry is None:
            return False
    install(registry)
    if _vhs_ready(registry):
        return True
    if loop is None:
        try:
            import server as comfy_server
            loop = getattr(comfy_server.PromptServer.instance, "loop", None)
        except Exception:
            loop = None
    if loop is None or retries <= 0:
        logger.warning("VHS 尚未加载且无事件循环可用，filename 输出补丁未安装")
        return False

    state = {"left": int(retries)}

    def _retry():
        install(registry)
        if _vhs_ready(registry):
            return
        state["left"] -= 1
        if state["left"] <= 0:
            logger.warning("等待 VHS 加载超时，filename 输出补丁未安装")
            return
        loop.call_later(interval, _retry)

    loop.call_later(interval, _retry)
    return False
