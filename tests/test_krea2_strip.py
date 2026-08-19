import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---- comfy stub (test-only, no real ComfyUI runtime) ----
comfy = types.ModuleType("comfy")
comfy.utils = types.ModuleType("comfy.utils")
comfy.cli_args = types.ModuleType("comfy.cli_args")
comfy.cli_args.args = types.SimpleNamespace()
sys.modules["comfy"] = comfy
sys.modules["comfy.utils"] = comfy.utils
sys.modules["comfy.cli_args"] = comfy.cli_args
sys.modules["torch"] = types.ModuleType("torch")
# ------------------------------------------------

from nodes.model.krea2 import _strip_qwen3_thinking  # noqa: E402


def run():
    # 基础：<think>思考</think>答案
    assert _strip_qwen3_thinking(
        "<think>用户思考...</think>最终答案。"
    ) == "最终答案。"

    # 行内 / 多行均可
    assert _strip_qwen3_thinking(
        "<think>思考内容</think>__char__ 是一位女性。"
    ) == "__char__ 是一位女性。"
    assert _strip_qwen3_thinking(
        "<think>\n思考内容\n</think>\n\n答案。"
    ) == "答案。"

    # 思考内容含"response"一词：不误判（不是纯文本 response 闭合）
    assert _strip_qwen3_thinking(
        "<think>用户思考 no response tag</think>答案。"
    ) == "答案。"

    # max_length 截断、未及 </think>：返回原始文本（兜底，不返回空）
    assert _strip_qwen3_thinking("<think>用户思考没写完") == "<think>用户思考没写完"

    # 无思考块：原样返回
    assert _strip_qwen3_thinking("A normal description.") == "A normal description."

    # 真实运行时输出（用户提供，</think> 闭合）
    real_out = """<think>
用户现在需要详细描述图片中的内容，特别是关于人物的性别、动作姿势、穿着、拍摄角度、镜头距离、背景、光照和照片色调，还要注意用__char__指代女性人物，不能提长相、人种、高矮胖瘦这些。首先得仔细看图片里的元素。

首先，性别：因为是女性，所以用__char__。动作和姿势：她手里拿着筷子，另一只手端着碗，应该是在吃饭的动作，可能正在夹菜或者准备吃。穿着：黑色T恤，上面有白色的数字"5"，袖子有点宽松。拍摄角度：应该是平视角度，因为画面中人物和背景的相对位置，看起来是正面或者稍微侧一点的视角。镜头距离：比较近，可能是中近景，能看清人物上半身和手部动作。背景：白色的墙壁，左边有个黑色的开关面板，桌子上有红色的纸杯。光照：看起来是室内光线，比较柔和，没有强烈的阴影，可能自然光或者室内灯。照片色调：整体偏暖色调，因为红色的杯子和黑色衣服对比，白色背景显得干净。

现在要组织语言，逐一对应每个部分：

性别：__char__ 是女性。

动作和姿势：__char__ 双手配合，右手握着两根筷子，左手托着一个碗，呈现出正在用筷子取食的动作姿态。

穿着：__char__ 穿着一件黑色短袖T恤，T恤正面印有白色的数字"5"。

拍摄角度：采用平视角度拍摄，使得人物与观众视线基本平行。

镜头距离：镜头距离较近，聚焦于人物上半身及手部动作细节。

背景：背景为简洁的白色墙面，左侧可见一个黑色的电源开关面板，前景处有一张桌子，桌上放置着一个红色的纸杯。

光照：室内光线均匀柔和，没有明显的强烈光源或阴影，营造出舒适的氛围。

照片色调：整体色调以黑白红为主，白色背景、黑色衣物与红色纸杯形成鲜明对比，色调温暖且富有生活气息。

检查有没有遗漏的部分，比如数量：碗一个，筷子两根，纸杯一个。形状：碗是圆形带花纹，筷子细长，纸杯圆柱形。纹理：碗有花纹，T恤是纯色，纸杯有文字图案。不过用户要求详细描述颜色、形状、大小、纹理、数量、文字、空间关系等，可能需要补充这些。

颜色：黑色（T恤）、白色（墙壁、数字"5"）、红色（纸杯）、浅棕色（筷子）。

形状：T恤是圆领短袖；碗是圆形，有花纹；筷子是细长圆柱形；纸杯是圆柱形。

大小：T恤覆盖上半身；碗大小适中，适合手持；筷子长度适中；纸杯高度适中。

纹理：T恤表面光滑；碗有花纹纹理；筷子表面光滑；纸杯表面有印刷文字纹理。

数量：1个碗，2根筷子，1个纸杯。

文字：T恤上的"5"，纸杯上的文字（虽然不清晰但有文字）。

空间关系：__char__ 手持碗在胸前，筷子在手中，纸杯放在桌子左下角，背景墙在后方，开关面板在左侧墙面。

现在把这些整合起来，确保每部分都覆盖到。
</think>

__char__ 是一位女性，她正进行用餐动作：右手握着两根细长的筷子，左手托着一只带有花纹的碗，呈现用筷子取食的姿态。

__char__ 穿着一件黑色短袖T恤，T恤正面印有白色的数字"5"。

拍摄角度为平视角度，镜头距离较近，聚焦于人物上半身及手部动作细节。

背景是一面简洁的白色墙面，左侧可见一个黑色的电源开关面板；前景处有一张桌子，桌上放置着一个红色的纸杯。

光照均匀柔和，无明显强烈光源或阴影，营造出舒适自然的室内氛围。照片整体色调以黑白红为主，白色背景、黑色衣物与红色纸杯形成鲜明对比，色调温暖且富有生活气息。

颜色方面，T恤为黑色，数字"5"和背景墙为白色，碗呈浅色系并带有花纹，筷子为浅棕色，纸杯主体为红色且印有文字。

形状上，T恤为圆领短袖；碗是圆形且带有花纹；筷子是细长圆柱形；纸杯为圆柱形。

大小方面，T恤覆盖上半身；碗大小适中，便于手持；筷子长度适中；纸杯高度适中。

纹理上，T恤表面光滑；碗有花纹纹理；筷子表面光滑；纸杯表面印有文字纹理。

数量上，碗1个，筷子2根，纸杯1个。

文字方面，T恤上的数字"5"清晰可见，纸杯表面也有文字装饰。

空间关系上，__char__ 手持碗位于胸前，筷子在手中，纸杯放置在桌子左下角，背景墙在人物后方，开关面板紧贴左侧墙面。"""
    result = _strip_qwen3_thinking(real_out)
    assert result.startswith("__char__ 是一位女性，她正进行用餐动作")
    assert "<think>" not in result
    assert "</think>" not in result
    assert "现在要组织语言" not in result
    assert "检查有没有遗漏" not in result

    print("test_krea2_strip.py: OK")


if __name__ == "__main__":
    run()
