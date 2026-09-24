"""给原生 LoadImage 追加 filename(STRING) 输出（运行时注册表补丁，不改核心文件）。

原生 ``nodes.LoadImage`` 只有 IMAGE/MASK 两个输出，工作流拿不到所加载图片的
原始路径。这里在 sfnodes 导入时把注册表里的 LoadImage 换成追加了 filename
输出的子类：

- 开关 ``sfnodes.LoadImage.FilenameOutput.Enabled``（comfy.settings.json，默认开）
  在**启动时**读取：关闭则不替换注册表，LoadImage 完全原生；改动需重启生效。
- **不能原地包装基类方法**（§79 VHS 同款在此不可用）：``LoadImageMask.load_image_mask``
  与 ``comfy_extras/nodes_webcam.py::WebcamCapture`` 解包/透传 ``super().load_image()``
  的 2 元组、``comfy_extras/nodes_load_3d.py`` 实例调用解包 2 元组——基类方法
  返回 3 值会让这三处立刻崩溃。替换注册表条目为子类则原类零改动，继承者
  （LoadImageMask/LoadImageOutput/WebcamCapture）与直接引用 ``nodes.LoadImage``
  的代码全部不受影响；前端 comfyClass 仍是 "LoadImage"（注册表键不变，
  §71/§119/§120 的前端补丁照常命中）。
- filename 语义 = image widget 原始值原样（input 相对路径，可直连 LoadImage；
  [output]/[temp] 注解值原样输出），对齐 §79 的"节点上显示的原始值"语义。
- ``server.node_info()`` 每次 /object_info 请求惰性读类属性 → 输出槽追加在末尾
  （IMAGE=0/MASK=1 不动），旧工作流按槽索引连线安全，无需前端 JS。

守卫：开关关闭 / 注册表缺 key / 注册表被换成非 LoadImage 子类 / 第三方子类
覆盖了 load_image（返回结构未知）/ 已带 filename 输出 / FUNCTION 不符 / 幂等
标记 → 跳过告警，绝不阻断启动。
"""

from .logger import get_logger

logger = get_logger("sfnodes.load_image_filename")

SETTING_ENABLED = "sfnodes.LoadImage.FilenameOutput.Enabled"

_KEY = "LoadImage"
_OUTPUT_TYPE = "STRING"
_OUTPUT_NAME = "filename"
_MARK = "_sf_filename_output_patched"


def read_enabled(settings=None):
    """读开关（默认开）。settings 可注入（测试/复用）。"""
    if settings is None:
        try:
            from .llm_client import read_comfy_settings

            settings = read_comfy_settings()
        except Exception:
            settings = {}
    if not isinstance(settings, dict):
        settings = {}
    return bool(settings.get(SETTING_ENABLED, True))


def _build_patched_class(base):
    """以 base 为基类创建追加 filename 输出的子类（原类零改动）。"""

    class SFLoadImage(base):
        RETURN_TYPES = tuple(base.RETURN_TYPES) + (_OUTPUT_TYPE,)
        RETURN_NAMES = tuple(getattr(base, "RETURN_NAMES", None) or base.RETURN_TYPES) + (_OUTPUT_NAME,)

        def load_image(self, image):
            result = super().load_image(image)
            return (*result, image if isinstance(image, str) else "")

    output_is_list = getattr(base, "OUTPUT_IS_LIST", None)
    if isinstance(output_is_list, tuple) and len(output_is_list) == len(base.RETURN_TYPES):
        SFLoadImage.OUTPUT_IS_LIST = tuple(output_is_list) + (False,)
    return SFLoadImage


def install(registry=None, settings=None):
    """按开关替换注册表中的 LoadImage 类；返回是否实际替换。

    registry/settings 可注入（测试）；缺省时分别取 ``nodes.NODE_CLASS_MAPPINGS``
    与 comfy.settings.json（读盘复用 ``llm_client.read_comfy_settings``）。
    """
    if not read_enabled(settings):
        return False
    try:
        import nodes as comfy_nodes
    except Exception as exc:
        logger.warning("LoadImage filename 输出补丁暂不可用（无法读取核心 nodes 模块）：%s", exc)
        return False
    if registry is None:
        registry = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", None)
        if registry is None:
            return False
    try:
        cur = registry.get(_KEY)
        if cur is None:
            logger.warning("LoadImage 不在节点注册表中，filename 输出补丁未安装")
            return False
        if getattr(cur, _MARK, False):
            return False
        base = comfy_nodes.LoadImage
        if not (isinstance(cur, type) and issubclass(cur, base)):
            logger.warning("注册表中的 LoadImage 不是核心 LoadImage 子类，filename 输出补丁跳过")
            return False
        if getattr(cur, "FUNCTION", None) != "load_image":
            logger.warning("LoadImage.FUNCTION 不是 load_image，filename 输出补丁跳过")
            return False
        names = getattr(cur, "RETURN_NAMES", None)
        if names is not None and _OUTPUT_NAME in tuple(names):
            setattr(cur, _MARK, True)
            return False
        if cur is not base and "load_image" in cur.__dict__:
            logger.warning("LoadImage.load_image 已被其他扩展覆盖，filename 输出补丁跳过")
            return False
        return_types = tuple(getattr(cur, "RETURN_TYPES", ()) or ())
        if return_types[:2] != ("IMAGE", "MASK"):
            logger.warning("LoadImage.RETURN_TYPES 结构不符，filename 输出补丁跳过")
            return False
        if names is not None and len(tuple(names)) != len(return_types):
            logger.warning("LoadImage.RETURN_NAMES 长度与 RETURN_TYPES 不符，filename 输出补丁跳过")
            return False
        patched = _build_patched_class(cur)
        setattr(patched, _MARK, True)
        registry[_KEY] = patched
        logger.info("原生 LoadImage 已追加 filename 输出（开关：%s，关闭需重启生效）", SETTING_ENABLED)
        return True
    except Exception as exc:
        logger.warning("LoadImage filename 输出补丁失败：%s", exc)
        return False
