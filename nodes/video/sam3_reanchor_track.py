_CATEGORY = "sfnodes/video"


def _parse_anchors(spec, total):
    """解析锚帧列表（逗号/分号/空白分隔），排序去重并校验范围。

    空串回退 `[0]`（等价于原生 SAM3_VideoTrack 的单段行为）。
    """
    if spec is None or not str(spec).strip():
        return [0]
    normalized = str(spec).replace(";", ",").replace("\n", ",").replace(" ", ",")
    anchors = []
    for part in normalized.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            anchor = int(part)
        except ValueError:
            raise ValueError(f"SF SAM3 Reanchor Track: 非法锚帧 '{part}'（应为整数）")
        if anchor < 0 or anchor >= total:
            raise ValueError(f"SF SAM3 Reanchor Track: 锚帧 {anchor} 越界（总帧数 {total}）")
        anchors.append(anchor)
    if not anchors:
        return [0]
    return sorted(set(anchors))


def _parse_prompts(spec):
    """逐行解析提示词（保留空行以保持与锚帧的行号对齐）。"""
    if not spec:
        return []
    return [line.strip() for line in str(spec).splitlines()]


class SFSAM3ReanchorTrack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "视频帧序列 [T,H,W,C]；每段锚帧后重新开始追踪，输出与输入同长度"}),
                "model": ("MODEL", {"tooltip": "SAM3.1 模型（CheckpointLoaderSimple 的 MODEL 输出）"}),
                "anchor_frames": ("STRING", {"default": "0",
                                             "tooltip": "锚帧列表（逗号/空格分隔），每个锚帧为一段起点：在该帧重新检测并以全新追踪状态向后传播；如 '0,120,300'"}),
            },
            "optional": {
                "clip": ("CLIP", {"tooltip": "SAM3 文本编码器（CheckpointLoaderSimple 的 CLIP），用于逐行编码 prompts；只用 conditioning 时可留空"}),
                "prompts": ("STRING", {"multiline": True, "default": "",
                                       "tooltip": "每行对应一个锚帧的提示词，可各不相同（如切镜后 person→woman）；空行/行数不足回退 conditioning，行数多余忽略"}),
                "conditioning": ("CONDITIONING", {"tooltip": "所有锚帧共用的现成文本条件（prompts 对应行为空时回退）"}),
                "initial_mask": ("MASK", {"tooltip": "种子遮罩（如 SAM3 Detect 输出）；默认仅作用于第一个锚帧，打开下方开关后每个锚段都用它建轨"}),
                "detection_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                                  "tooltip": "段内文本检测得分阈值（越高越严格）"}),
                "max_objects": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1,
                                        "tooltip": "每段最多追踪的对象数；输出跨对象并集塌成单一身份"}),
                "detect_interval": ("INT", {"default": 1, "min": 1, "max": 999, "step": 1,
                                            "tooltip": "段内检测间隔帧数（1=每帧检测；越大越省算力）"}),
                "initial_mask_always": ("BOOLEAN", {"default": False,
                                                    "tooltip": "打开后每个锚段都以 initial_mask 作种子（全程有效），适合机位/主体位置基本不变的视频；主体移动大时后续锚帧会因首帧形状错位。未接 initial_mask 时打开会报错"}),
            },
        }

    RETURN_TYPES = ("SAM3_TRACK_DATA",)
    RETURN_NAMES = ("track_data",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "按锚帧分段调用 SAM3 追踪：每个锚帧用对应提示词重新检测并以全新追踪状态向后传播，各段并集塌成单一身份；输出与输入同长度，可接 SAM3 Track to Mask / SCAIL-2 driving"

    def execute(self, images, model, anchor_frames="0", clip=None, prompts="", conditioning=None,
                initial_mask=None, detection_threshold=0.5, max_objects=1, detect_interval=1,
                initial_mask_always=False):
        import torch
        from comfy_extras.nodes_sam3 import SAM3_VideoTrack
        from ...sf_utils.common import node_result
        from ...sf_utils.track_data_ops import concat_track_data_segments

        if images is None or images.ndim != 4:
            raise ValueError("SF SAM3 Reanchor Track: images 必须是 [T,H,W,C] 的 IMAGE 张量")
        total = int(images.shape[0])
        if total <= 0:
            raise ValueError("SF SAM3 Reanchor Track: images 帧数为 0")

        anchors = _parse_anchors(anchor_frames, total)
        prompt_lines = _parse_prompts(prompts)

        if initial_mask_always and initial_mask is None:
            raise ValueError("SF SAM3 Reanchor Track: initial_mask_always 已打开，但未接 initial_mask")
        if initial_mask is not None and initial_mask.ndim == 2:
            initial_mask = initial_mask.unsqueeze(0)

        segments = []
        for index, anchor in enumerate(anchors):
            end = anchors[index + 1] if index + 1 < len(anchors) else total
            prompt = prompt_lines[index] if index < len(prompt_lines) else ""

            cond = None
            if prompt:
                if clip is None:
                    raise ValueError(
                        f"SF SAM3 Reanchor Track: 第 {index + 1} 行提示词 '{prompt}' 需要 clip 输入才能编码")
                cond = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
            elif conditioning is not None:
                cond = conditioning

            seed = initial_mask if (initial_mask is not None and (initial_mask_always or index == 0)) else None
            if seed is None and cond is None:
                raise ValueError(
                    f"SF SAM3 Reanchor Track: 锚帧 {anchor} 既无提示词，也无 conditioning/initial_mask")

            result = node_result(SAM3_VideoTrack.execute(
                images=images[anchor:end],
                model=model,
                initial_mask=seed,
                conditioning=cond,
                detection_threshold=float(detection_threshold),
                max_objects=int(max_objects),
                detect_interval=int(detect_interval),
            ))
            track_data = result[0] if result else None
            if not isinstance(track_data, dict) or "packed_masks" not in track_data:
                raise RuntimeError("SF SAM3 Reanchor Track: SAM3_VideoTrack 返回结构异常")
            segments.append((anchor, track_data))

        return (concat_track_data_segments(segments, total, torch=torch),)
