"""SFVideoCompare：TE_MAN「TE MAN 视频对比」干净室复刻（两路 VIDEO 叠加播放对比）。

复刻 tl2012tl/TE_MAN 的 TE_Video_Comparer（Cython .pyd 闭源，仓库 LICENSE
禁止复制/衍生；能力清单自 README v3.7 与其 .pyd 字符串表还原，实现自写）：
两个可选 VIDEO 输入 → 各自落盘 temp mp4（H264，浏览器可播；源为 h264/mp4 时
save_to 内部纯 remux 不重编码）→ av 读帧数/帧率/时长 → `ui` 返回
a_videos/b_videos 供前端节点内叠加播放（分界线/进度条/静音/音频 A·B/
0.25x-2x 速度/同步帧/全屏/同步播放，信息存 node.properties 随工作流恢复）。

无输出槽：节点是纯对比查看器（OUTPUT_NODE，入队即执行并触发上游视频生成）。
"""

import os

_CATEGORY = "sfnodes/video"


class SFVideoCompare:
    DESCRIPTION = (
        "两路 VIDEO 在节点内叠加播放，鼠标移动分界线实时对比；"
        "支持点击暂停/继续、进度条拖动、静音/音频 A/音频 B、0.25x-2x 播放速度、"
        "同步帧（总帧数相同时按帧号对齐）、全屏对比与同步播放（从头播放、结束自动重播）；"
        "视频信息随工作流保存，刷新后自动恢复"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "video_a": ("VIDEO", {
                    "tooltip": "对比视频 A（可选）：显示在分界线左侧；与 B 同步播放。只接 A 时自动占满预览",
                }),
                "video_b": ("VIDEO", {
                    "tooltip": "对比视频 B（可选）：显示在分界线右侧；只接 B 时自动占满预览",
                }),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = _CATEGORY

    def execute(self, video_a=None, video_b=None):
        import folder_paths
        from comfy_api.latest import Types

        from ...sf_utils.video_compare import (
            build_entry,
            preview_filename,
            preview_path,
            probe_video_meta,
        )

        temp_dir = folder_paths.get_temp_directory()
        ui = {"a_videos": [], "b_videos": []}
        for key, label, video in (("a_videos", "A", video_a), ("b_videos", "B", video_b)):
            if video is None:
                continue
            filename = preview_filename()
            path = preview_path(temp_dir, filename)
            try:
                video.save_to(
                    path,
                    format=Types.VideoContainer.MP4,
                    codec=Types.VideoCodec.H264,
                )
                meta = probe_video_meta(path)
            except Exception as e:
                raise RuntimeError(f"[SFVideoCompare] 对比视频 {label} 预览生成失败：{e}") from e
            ui[key] = [build_entry(filename, "", "temp", meta)]
        return {"ui": ui}
