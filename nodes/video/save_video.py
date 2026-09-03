"""SFSaveVideoSwitchable — SaveVideo 强化版：开关控制是否落盘 + overwrite + temp 预览。

原生 SaveVideo（comfy_extras/nodes_video.py:135）始终写入 output 目录，
本节点在保留全部原生输入（video/filename_prefix/format/codec）基础上新增：

- save_enabled (BOOLEAN, default True): True=与原生一致写入 output/ 并 ui=PreviewVideo；False=跳过 output 落盘，仅透传 VIDEO，并在 temp/ 生成预览（复用 comfy_extras.nodes_video.save_video_preview），前端仍可见但不污染 output。
- overwrite (BOOLEAN, default False): 仅在 save_enabled=True 时生效。False=保留计数（永不覆盖，若目标文件已存在则递增 counter 直到空闲，默认安全）；True=允许覆盖（直接写入 get_save_image_path 返回的计数文件，若已存在则覆盖）。

无前端 JS，纯后端。CATEGORY sfnodes/video。
"""

import os

import folder_paths
from comfy.cli_args import args
from comfy_api.latest import io, ui, Input, Types

_CATEGORY = "sfnodes/video"


def _save_video_codec_input(supported_codecs: list[str], *, optional=False, hidden=False):
    """复刻 comfy_extras/nodes_video.py:_save_video_codec_input，禁内联漂移。

    原函数为 DynamicCombo 嵌套：codec -> encoding{auto/re-encode{crf}}。
    """
    codec_options = []
    if "auto" in supported_codecs:
        codec_options.append(io.DynamicCombo.Option("auto", []))
    if "h264" in supported_codecs:
        codec_options.append(
            io.DynamicCombo.Option(
                "h264",
                [
                    io.DynamicCombo.Input(
                        "encoding",
                        display_name="encoding mode",
                        options=[
                            io.DynamicCombo.Option("auto", []),
                            io.DynamicCombo.Option(
                                "re-encode",
                                [
                                    io.Float.Input("crf", default=23.0, min=0.0, max=51.0, step=1.0, tooltip="Lower values produce higher quality and larger files."),
                                ],
                            ),
                        ],
                        optional=True,
                        tooltip="Automatic preserves compatible H.264 streams. Re-encode applies custom encoding options.",
                    ),
                ],
            )
        )
    if "av1" in supported_codecs:
        codec_options.append(
            io.DynamicCombo.Option(
                "av1",
                [
                    io.DynamicCombo.Input(
                        "encoding",
                        display_name="encoding mode",
                        options=[
                            io.DynamicCombo.Option("auto", []),
                            io.DynamicCombo.Option(
                                "re-encode",
                                [
                                    io.Float.Input("crf", default=30.0, min=0.0, max=63.0, step=1.0, tooltip="Lower values produce higher quality and larger files."),
                                ],
                            ),
                        ],
                        optional=True,
                        tooltip="Automatic preserves compatible AV1 streams. Re-encode applies custom encoding options.",
                    ),
                ],
            )
        )
    return io.DynamicCombo.Input(
        "codec",
        options=codec_options,
        optional=optional,
        tooltip="The output video codec. Auto preserves a compatible source stream. H.264 and AV1 re-encoding support SDR, HDR (HLG), and HDR PQ.",
        extra_dict={"hidden": True} if hidden else None,
    )


class SFSaveVideoSwitchable(io.ComfyNode):
    DESCRIPTION = (
        "SF Save Video (Switchable) — SaveVideo 强化版：开关控制是否保存到 output。开启时与原生 SaveVideo 一致落盘；关闭时跳过 output 写入、仅透传 VIDEO 并在 temp 生成预览（不污染 output）。overwrite 控制计数文件已存在时是否覆盖，默认保留计数（不覆盖）。\n"
        "Inputs: video/filename_prefix/format(codec+encoding)/save_enabled/overwrite"
    )

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SFSaveVideoSwitchable",
            display_name="SF Save Video (Switchable)",
            category=_CATEGORY,
            essentials_category="Basics",
            description="Saves the input videos to your ComfyUI output directory, with a toggle to skip saving and an overwrite option.",
            inputs=[
                io.Video.Input("video", tooltip="The video to save."),
                io.String.Input("filename_prefix", default="video/ComfyUI", tooltip="The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."),
                io.DynamicCombo.Input(
                    "format",
                    options=[
                        io.DynamicCombo.Option("auto", [_save_video_codec_input(["auto", "h264", "av1"])]),
                        io.DynamicCombo.Option("mp4", [_save_video_codec_input(["auto", "h264", "av1"])]),
                        io.DynamicCombo.Option("mkv", [_save_video_codec_input(["auto", "h264", "av1"])]),
                        io.DynamicCombo.Option("webm", [_save_video_codec_input(["auto", "av1"])]),
                    ],
                    tooltip="The output container. Auto uses MP4 for Auto/H.264 and WebM for AV1. MP4, MKV, and WebM select a specific container.",
                ),
                _save_video_codec_input(["auto", "h264", "av1"], optional=True, hidden=True),
                io.Boolean.Input("save_enabled", default=True, label_on="save", label_off="skip", tooltip="True=保存到 output 目录（与原生一致）；False=跳过 output 落盘，仅透传 VIDEO 并在 temp 生成预览"),
                io.Boolean.Input("overwrite", default=False, label_on="overwrite", label_off="increment", tooltip="仅 save_enabled=True 时生效。False=保留计数（默认，目标已存在则递增 counter 直到空闲，永不覆盖）；True=允许覆盖已存在的计数文件"),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[io.Video.Output("video", tooltip="The input video, unchanged.")],
        )

    @classmethod
    def execute(
        cls,
        video: Input.Video,
        filename_prefix,
        format: io.DynamicCombo.Type | str,
        codec: io.DynamicCombo.Type | None = None,
        save_enabled: bool = True,
        overwrite: bool = False,
    ) -> io.NodeOutput:
        # ── B 分支：跳过落盘，仅 temp 预览透传 ──
        if not save_enabled:
            try:
                from comfy_extras.nodes_video import save_video_preview
            except Exception:
                # 兜底：无预览也透传，不崩
                return io.NodeOutput(video)
            try:
                preview = save_video_preview(video)
                return io.NodeOutput(video, ui=preview)
            except Exception as e:
                print(f"[SFSaveVideoSwitchable] temp preview failed: {e}")
                return io.NodeOutput(video)

        # ── 以下与原生 SaveVideo 1:1（增 overwrite 分支） ──
        if isinstance(format, dict):
            format_name = format["format"]
            codec = format.get("codec") or codec
        else:
            format_name = format
        if codec is None:
            codec = {"codec": "auto"}
        codec_name = codec["codec"]
        if format_name == "auto":
            format_name = "webm" if codec_name == "av1" else "mp4"
        encoding = codec.get("encoding") or {}
        width, height = video.get_dimensions()
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix,
            folder_paths.get_output_directory(),
            width,
            height,
        )
        saved_metadata = None
        if not args.disable_metadata:
            metadata = {}
            if cls.hidden.extra_pnginfo is not None:
                metadata.update(cls.hidden.extra_pnginfo)
            if cls.hidden.prompt is not None:
                metadata["prompt"] = cls.hidden.prompt
            if len(metadata) > 0:
                saved_metadata = metadata
        ext = Types.VideoContainer.get_extension(format_name)
        # overwrite=False（默认保留计数）：若目标已存在则递增 counter 直到空闲
        if not overwrite:
            # get_save_image_path 已找空闲，但为应对外部并发/用户手动放同名文件，做二次探空
            # 最多递增 100000 次，与 SFSaveImageExact 同上限
            for _ in range(100000):
                cand = f"{filename}_{counter:05}_.{ext}"
                full = os.path.join(full_output_folder, cand)
                if not os.path.exists(full):
                    break
                counter += 1
            else:
                raise RuntimeError("[SFSaveVideoSwitchable] 无法找到空闲文件名（超出尝试上限）")
        file = f"{filename}_{counter:05}_.{ext}"
        video.save_to(
            os.path.join(full_output_folder, file),
            format=Types.VideoContainer(format_name),
            codec=Types.VideoCodec(codec_name),
            metadata=saved_metadata,
            crf=encoding.get("crf"),
        )
        return io.NodeOutput(video, ui=ui.PreviewVideo([ui.SavedResult(file, subfolder, io.FolderType.output)]))
