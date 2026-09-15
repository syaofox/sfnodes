"""SF SCAIL-2 四节点（复刻 ComfyUI-SCAIL2-Easy）。

不重写 SCAIL-2 推理：复用 ComfyUI 原生 SCAIL-2 支持（comfy_extras.nodes_scail
的 WanSCAILToVideo / SCAIL2ColoredMask / _extract_mask_to_28ch、comfy_extras.
nodes_sam3.SAM3_VideoTrack、comfy.context_windows），把易接错的分块、参考包、
colored mask 编排封装成 4 个节点：

- SFSCAIL2FitVideo       统一输入视频尺寸（512p/704p/custom，对齐 32）
- SFSCAIL2ReferencePack  多主体/多参考图 → SCAIL2_REFERENCE_PACK
- SFSCAIL2ReferenceSAMBuilder  跑 SAM3 给主体/参考图生成彩色蒙版
- SFSCAIL2SimpleVideo    主生成（animation/replacement + 长视频 chunk/context）

纯逻辑（帧数/尺寸数学、多主体拼图布局）在 sf_utils/scail2_easy.py；
上下文窗口采样补丁在 sf_utils/scail2_context.py。
"""

from __future__ import annotations

import gc
import json

import torch
import torch.nn.functional as F

from ...sf_utils.scail2_context import apply_scail2_easy_context
from ...sf_utils.scail2_easy import (
    LONG_VIDEO_MODES,
    MAX_LEGACY_REFERENCE_IMAGES_PER_SUBJECT,
    MAX_MIXED_REFERENCE_IMAGES,
    MAX_PREFIX_REFERENCE_IMAGES,
    MAX_REFERENCE_SUBJECTS,
    MAX_STAGE_REFERENCE_SOURCE_HEIGHT,
    REFERENCE_PACK_TYPE,
    RESOLUTION_PRESETS,
    SCAIL_COLOR_PALETTE,
    clamp_int as _clamp_int,
    infer_generation_size as _infer_generation_size_math,
    is_reference_pack as _is_reference_pack,
    layout_stage_entries as _layout_stage_entries,
    subject_color as _subject_color,
    target_size_for_video as _target_size_for_video_math,
    wan_frame_count_cover as _wan_frame_count_cover,
    wan_frame_count_floor as _wan_frame_count_floor,
)

_CATEGORY = "sfnodes/video"
REFERENCE_IMAGE_INPUT_TYPE = f"IMAGE,{REFERENCE_PACK_TYPE}"


def _shape(value):
    return list(value.shape) if hasattr(value, "shape") else []


def _track_object_count(track_data):
    if not isinstance(track_data, dict):
        return 0
    packed = track_data.get("packed_masks")
    if isinstance(packed, torch.Tensor) and packed.ndim >= 2:
        return int(packed.shape[1])
    scores = track_data.get("scores")
    if scores is not None:
        try:
            return int(len(scores))
        except TypeError:
            return 0
    return 0


def _trim_frames(value, frame_count):
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise TypeError("Image inputs must be ComfyUI IMAGE tensors.")
    if value.ndim != 4:
        raise ValueError("Image inputs must have shape [frames, height, width, channels].")
    return value[:frame_count].detach()


def _first_image(value, name):
    if not isinstance(value, torch.Tensor) or value.ndim != 4:
        raise ValueError(f"{name} must be a ComfyUI IMAGE tensor.")
    if value.shape[0] <= 0:
        raise ValueError(f"{name} has no images.")
    return value[:1].detach().contiguous()


def _node_result(value):
    if hasattr(value, "result"):
        result = value.result
        if result is None:
            return ()
        return tuple(result)
    if isinstance(value, tuple):
        return value
    return (value,)


def _get_scail_nodes_module():
    try:
        from comfy_extras import nodes_scail

        return nodes_scail
    except ImportError:
        from comfy_extras import nodes_wan

        return nodes_wan


def _extract_scail_mask_to_28ch(rgb_video):
    """复用核心 comfy_extras.nodes_scail._extract_mask_to_28ch（禁内联副本）。"""
    if rgb_video.ndim != 4 or rgb_video.shape[-1] < 3:
        raise ValueError("SCAIL mask video must have shape [frames, height, width, channels].")
    native_extract = getattr(_get_scail_nodes_module(), "_extract_mask_to_28ch", None)
    if native_extract is None:
        raise RuntimeError("This ComfyUI build does not provide SCAIL-2 _extract_mask_to_28ch.")
    return native_extract(rgb_video)


def _infer_generation_size(pose_video):
    if pose_video.ndim != 4:
        raise ValueError("pose_video must be a ComfyUI IMAGE tensor.")
    return _infer_generation_size_math(int(pose_video.shape[1]), int(pose_video.shape[2]))


def _empty_cache(force: bool = False):
    if force:
        gc.collect()
    try:
        import comfy.model_management

        comfy.model_management.cleanup_models_gc()
        try:
            comfy.model_management.soft_empty_cache(force=force)
        except TypeError:
            comfy.model_management.soft_empty_cache()
    except Exception:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _target_size_for_video(images, resolution, custom_width=832, custom_height=480):
    if images.ndim != 4:
        raise ValueError("video must be a ComfyUI IMAGE tensor.")
    return _target_size_for_video_math(
        int(images.shape[1]), int(images.shape[2]), resolution, custom_width, custom_height
    )


def _resize_to_target(images, target_width, target_height):
    if images.ndim != 4:
        raise ValueError("video must be a ComfyUI IMAGE tensor.")
    source_height = int(images.shape[1])
    source_width = int(images.shape[2])
    if source_width == target_width and source_height == target_height:
        return images.detach().contiguous()

    dtype = images.dtype
    chunks = []

    for start in range(0, int(images.shape[0]), 32):
        chunk = images[start : start + 32].detach().float().movedim(-1, 1)
        chunk = F.interpolate(
            chunk,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )
        chunks.append(chunk.movedim(1, -1).to(dtype=dtype).contiguous())

    return torch.cat(chunks, dim=0).clamp(0, 1)


def _resize_mask_like(mask, image):
    if mask.shape[1] == image.shape[1] and mask.shape[2] == image.shape[2]:
        return mask.detach()
    resized = F.interpolate(
        mask.detach().float().movedim(-1, 1),
        size=(int(image.shape[1]), int(image.shape[2])),
        mode="nearest",
    )
    return resized.movedim(1, -1).to(dtype=image.dtype).contiguous()


def _apply_reference_mask(reference_image, reference_image_mask):
    mask = _resize_mask_like(reference_image_mask[:1], reference_image[:1])
    alpha = (mask[..., :3].max(dim=-1, keepdim=True).values > 0.1).to(dtype=reference_image.dtype)
    return (reference_image[:1].detach() * alpha).contiguous()


def _float_list(value):
    return [round(float(item), 4) for item in value.detach().cpu().flatten().tolist()]


def _match_chunk_color_to_overlap(
    frames,
    current_overlap,
    reference_overlap,
    strength: float = 0.65,
):
    info = {"applied": False}
    if frames.ndim != 4 or frames.shape[-1] < 3 or frames.shape[0] <= 0:
        info["reason"] = "invalid_frames"
        return frames, info
    if current_overlap is None or reference_overlap is None:
        info["reason"] = "missing_overlap"
        return frames, info

    overlap_count = min(int(current_overlap.shape[0]), int(reference_overlap.shape[0]))
    if overlap_count <= 0:
        info["reason"] = "empty_overlap"
        return frames, info
    if current_overlap.ndim != 4 or reference_overlap.ndim != 4:
        info["reason"] = "invalid_overlap"
        return frames, info

    current = current_overlap[-overlap_count:, :, :, :3].detach()
    reference = reference_overlap[-overlap_count:, :, :, :3].detach()
    if current.shape[1:3] != reference.shape[1:3]:
        reference = F.interpolate(
            reference.float().movedim(-1, 1),
            size=(int(current.shape[1]), int(current.shape[2])),
            mode="bilinear",
            align_corners=False,
        ).movedim(1, -1)

    device = frames.device
    current = current.to(device=device, dtype=torch.float32)
    reference = reference.to(device=device, dtype=torch.float32)
    current_mean = current.mean(dim=(0, 1, 2))
    reference_mean = reference.mean(dim=(0, 1, 2))
    current_std = current.std(dim=(0, 1, 2), unbiased=False).clamp_min(1e-4)
    reference_std = reference.std(dim=(0, 1, 2), unbiased=False).clamp_min(1e-4)
    scale = (reference_std / current_std).clamp(0.7, 1.35)
    shift = (reference_mean - current_mean * scale).clamp(-0.25, 0.25)
    strength = max(0.0, min(1.0, float(strength)))

    corrected_rgb = frames[:, :, :, :3].to(torch.float32) * scale.view(1, 1, 1, 3) + shift.view(1, 1, 1, 3)
    blended_rgb = torch.lerp(frames[:, :, :, :3].to(torch.float32), corrected_rgb, strength).clamp(0, 1)
    corrected = frames.clone()
    corrected[:, :, :, :3] = blended_rgb.to(dtype=frames.dtype)

    info.update(
        {
            "applied": True,
            "overlap_frames": int(overlap_count),
            "strength": float(strength),
            "scale": _float_list(scale),
            "shift": _float_list(shift),
        }
    )
    return corrected.contiguous(), info


def _strip_latent_for_decode(latent):
    if not isinstance(latent, dict) or "samples" not in latent:
        raise RuntimeError("SamplerCustom returned an invalid latent output.")
    samples = latent["samples"]
    if hasattr(samples, "detach"):
        samples = samples.detach()
        if hasattr(samples, "contiguous"):
            samples = samples.contiguous()
    return {"samples": samples}


def _sample_for_decode(
    *,
    model,
    positive,
    negative,
    sampler,
    sigmas,
    latent,
    seed: int,
    cfg: float,
):
    from comfy_extras.nodes_custom_sampler import SamplerCustom

    sampled = _node_result(
        SamplerCustom.execute(
            model,
            True,
            int(seed),
            float(cfg),
            positive,
            negative,
            sampler,
            sigmas,
            latent,
        )
    )
    if not sampled:
        raise RuntimeError("SamplerCustom returned no latent output.")

    latent_to_decode = sampled[1] if len(sampled) > 1 else sampled[0]
    decode_latent = _strip_latent_for_decode(latent_to_decode)
    del sampled, latent_to_decode
    return decode_latent


def _decode_latent_to_frames(vae, latent_to_decode):
    import nodes

    decoded = nodes.VAEDecode().decode(vae, latent_to_decode)[0]
    frames = decoded.detach().cpu().contiguous().clamp(0, 1)
    del decoded, latent_to_decode
    _empty_cache()
    return frames


def _run_native_scail_chunk(
    *,
    model,
    positive,
    negative,
    vae,
    sampler,
    sigmas,
    reference_image,
    pose_video,
    clip_vision_output,
    pose_video_mask,
    reference_image_mask,
    previous_frames,
    width: int,
    height: int,
    length: int,
    video_frame_offset: int,
    previous_frame_count: int,
    replacement_mode: bool,
    seed: int,
    cfg: float,
    pose_strength: float,
):
    WanSCAILToVideo = _get_scail_nodes_module().WanSCAILToVideo
    scail_out = _node_result(
        WanSCAILToVideo.execute(
            positive,
            negative,
            vae,
            width,
            height,
            length,
            1,
            pose_strength,
            0.0,
            1.0,
            video_frame_offset,
            previous_frame_count,
            replacement_mode=replacement_mode,
            reference_image=reference_image,
            clip_vision_output=clip_vision_output,
            pose_video=pose_video,
            pose_video_mask=pose_video_mask,
            reference_image_mask=reference_image_mask,
            previous_frames=previous_frames,
        )
    )
    if len(scail_out) != 4:
        raise RuntimeError("WanSCAILToVideo returned an unexpected output shape.")

    chunk_positive, chunk_negative, latent, next_offset = scail_out
    latent_to_decode = _sample_for_decode(
        model=model,
        positive=chunk_positive,
        negative=chunk_negative,
        sampler=sampler,
        sigmas=sigmas,
        latent=latent,
        seed=seed,
        cfg=cfg,
    )
    del scail_out, chunk_positive, chunk_negative, latent
    reference_image = pose_video = clip_vision_output = pose_video_mask = reference_image_mask = previous_frames = None
    _empty_cache(force=True)
    frames = _decode_latent_to_frames(vae, latent_to_decode)
    summary = {
        "width": int(width),
        "height": int(height),
        "length": int(length),
        "seed": int(seed),
        "cfg": float(cfg),
        "input_video_frame_offset": int(video_frame_offset),
        "output_video_frame_offset": int(next_offset),
        "decoded_shape": _shape(frames),
    }
    return frames, int(next_offset), summary


def _create_scail_masks(driving_track_data, reference_track_data, replacement_mode: bool):
    SCAIL2ColoredMask = getattr(_get_scail_nodes_module(), "SCAIL2ColoredMask", None)
    if SCAIL2ColoredMask is None:
        raise RuntimeError("SCAIL2ColoredMask is unavailable in this ComfyUI build.")

    masks = _node_result(
        SCAIL2ColoredMask.execute(
            driving_track_data,
            "",
            "left_to_right",
            bool(replacement_mode),
            ref_track_data=reference_track_data,
        )
    )
    if len(masks) != 2:
        raise RuntimeError("SCAIL2ColoredMask returned an unexpected output shape.")
    return masks[0], masks[1]


def _create_driving_scail_mask(driving_track_data, replacement_mode: bool):
    SCAIL2ColoredMask = getattr(_get_scail_nodes_module(), "SCAIL2ColoredMask", None)
    if SCAIL2ColoredMask is None:
        raise RuntimeError("SCAIL2ColoredMask is unavailable in this ComfyUI build.")

    masks = _node_result(
        SCAIL2ColoredMask.execute(
            driving_track_data,
            "",
            "left_to_right",
            bool(replacement_mode),
        )
    )
    if not masks:
        raise RuntimeError("SCAIL2ColoredMask returned no driving mask.")
    return masks[0]


def _run_sam3_track(images, model, conditioning, detection_threshold: float, max_objects: int, detect_interval: int):
    from comfy_extras.nodes_sam3 import SAM3_VideoTrack

    if conditioning is None:
        raise ValueError("SCAIL-2 Reference SAM Builder needs a SAM3 conditioning input, usually CLIPTextEncode('person').")
    result = _node_result(
        SAM3_VideoTrack.execute(
            images,
            model,
            initial_mask=None,
            conditioning=conditioning,
            detection_threshold=float(detection_threshold),
            max_objects=int(max_objects),
            detect_interval=max(1, int(detect_interval)),
        )
    )
    if len(result) != 1:
        raise RuntimeError("SAM3_VideoTrack returned an unexpected output shape.")
    return result[0]


def _solid_color_mask_like(image, color, alpha=None):
    mask = torch.empty((1, int(image.shape[1]), int(image.shape[2]), 3), device=image.device, dtype=image.dtype)
    mask[..., 0] = color[0]
    mask[..., 1] = color[1]
    mask[..., 2] = color[2]
    if alpha is not None:
        mask = mask * alpha.to(device=image.device, dtype=image.dtype)
    return mask.contiguous()


def _render_reference_sam_mask(reference_image, track_data, subject_index: int):
    raw_mask = _create_driving_scail_mask(track_data, False)
    raw_mask = _resize_mask_like(raw_mask[:1], reference_image[:1])
    alpha = (raw_mask[..., :3].max(dim=-1, keepdim=True).values > 0.1).to(dtype=reference_image.dtype)
    return _solid_color_mask_like(reference_image[:1], _subject_color(subject_index), alpha)


def _render_mixed_reference_sam_mask(reference_image, track_data):
    from comfy.ldm.sam3.tracker import unpack_masks

    output = torch.zeros((1, int(reference_image.shape[1]), int(reference_image.shape[2]), 3), device=reference_image.device, dtype=reference_image.dtype)
    if not isinstance(track_data, dict):
        return output.contiguous()

    packed = track_data.get("packed_masks")
    if not isinstance(packed, torch.Tensor) or packed.ndim < 2 or int(packed.shape[1]) <= 0:
        return output.contiguous()

    masks = unpack_masks(packed.to(reference_image.device)).float()
    if masks.ndim != 4 or int(masks.shape[0]) <= 0 or int(masks.shape[1]) <= 0:
        return output.contiguous()

    frame_masks = masks[:1]
    n_obj = int(frame_masks.shape[1])
    height = int(reference_image.shape[1])
    width = int(reference_image.shape[2])
    frame_masks = F.interpolate(
        frame_masks.reshape(n_obj, 1, int(frame_masks.shape[-2]), int(frame_masks.shape[-1])),
        size=(height, width),
        mode="nearest",
    ).reshape(n_obj, height, width) > 0.5
    if not bool(frame_masks.any().item()):
        return output.contiguous()

    areas = frame_masks.float().sum(dim=(1, 2)).clamp(min=1.0)
    grid_x = torch.arange(width, device=frame_masks.device, dtype=torch.float32).view(1, width)
    centers = (frame_masks.float() * grid_x).sum(dim=(1, 2)) / areas
    order = torch.argsort(centers)
    frame_masks = frame_masks[order]

    any_mask = frame_masks.any(dim=0)
    obj_idx_map = frame_masks.to(torch.uint8).argmax(dim=0)
    colors = torch.tensor(
        [SCAIL_COLOR_PALETTE[index % len(SCAIL_COLOR_PALETTE)] for index in range(int(frame_masks.shape[0]))],
        device=reference_image.device,
        dtype=reference_image.dtype,
    )
    color_overlay = colors[obj_idx_map].unsqueeze(0)
    return torch.where(any_mask.unsqueeze(0).unsqueeze(-1), color_overlay, output).contiguous()


def _mask_with_background(mask, *, white_background: bool):
    if not white_background:
        return mask.contiguous().clamp(0, 1)
    alpha = (mask[..., :3].max(dim=-1, keepdim=True).values > 0.1).to(device=mask.device, dtype=mask.dtype)
    return (mask + (1.0 - alpha)).contiguous().clamp(0, 1)


def _encode_clip_vision(clip_vision, image):
    import nodes

    return nodes.CLIPVisionEncode().encode(clip_vision, image, "none")[0]


def _colored_reference_mask(item, image, *, white_background: bool):
    if item["kind"] == "background":
        return _solid_color_mask_like(image[:1], (1.0, 1.0, 1.0))

    color = _subject_color(int(item["subject_index"]))
    if item.get("mask") is not None:
        source_mask = _resize_mask_like(item["mask"][:1], image[:1])
        alpha = (source_mask[..., :3].max(dim=-1, keepdim=True).values > 0.1).to(dtype=image.dtype)
        mask = _solid_color_mask_like(image[:1], color, alpha)
        if white_background:
            mask = mask + (1.0 - alpha.to(device=mask.device, dtype=mask.dtype))
        return mask.clamp(0, 1).contiguous()
    return _solid_color_mask_like(image[:1], color)


def _resize_image_exact(image, width: int, height: int, *, mode: str = "bicubic"):
    if image.ndim != 4:
        raise ValueError("Reference images must have shape [frames, height, width, channels].")
    if int(image.shape[1]) == int(height) and int(image.shape[2]) == int(width):
        return image[:1, :, :, :3].detach().contiguous()
    tensor = image[:1, :, :, :3].detach().float().movedim(-1, 1)
    kwargs = {}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    resized = F.interpolate(tensor, size=(int(height), int(width)), mode=mode, **kwargs)
    return resized.movedim(1, -1).to(dtype=image.dtype).clamp(0, 1).contiguous()


def _resize_image_cover(image, width: int, height: int, *, mode: str = "bicubic"):
    if image.ndim != 4:
        raise ValueError("Reference images must have shape [frames, height, width, channels].")

    target_width = max(1, int(width))
    target_height = max(1, int(height))
    source_height = int(image.shape[1])
    source_width = int(image.shape[2])
    if source_height <= 0 or source_width <= 0:
        raise ValueError("Reference image dimensions must be greater than zero.")

    crop = image[:1, :, :, :3].detach()
    if source_width * target_height > source_height * target_width:
        crop_width = max(1, min(source_width, round(source_height * target_width / target_height)))
        left = max(0, (source_width - crop_width) // 2)
        crop = crop[:, :, left:left + crop_width, :]
    elif source_width * target_height < source_height * target_width:
        crop_height = max(1, min(source_height, round(source_width * target_height / target_width)))
        top = max(0, (source_height - crop_height) // 2)
        crop = crop[:, top:top + crop_height, :, :]

    return _resize_image_exact(crop, target_width, target_height, mode=mode)


def _resize_image_contain(
    image,
    width: int,
    height: int,
    *,
    fill=(0.0, 0.0, 0.0),
    mode: str = "bicubic",
):
    if image.ndim != 4:
        raise ValueError("Reference images must have shape [frames, height, width, channels].")

    target_width = max(1, int(width))
    target_height = max(1, int(height))
    source_height = int(image.shape[1])
    source_width = int(image.shape[2])
    if source_height <= 0 or source_width <= 0:
        raise ValueError("Reference image dimensions must be greater than zero.")

    scale = min(target_width / source_width, target_height / source_height)
    resized_width = max(1, min(target_width, int(round(source_width * scale))))
    resized_height = max(1, min(target_height, int(round(source_height * scale))))
    resized = _resize_image_exact(image, resized_width, resized_height, mode=mode)

    canvas = torch.empty((1, target_height, target_width, 3), device=resized.device, dtype=resized.dtype)
    canvas[..., 0] = fill[0]
    canvas[..., 1] = fill[1]
    canvas[..., 2] = fill[2]
    paste_x = max(0, (target_width - resized_width) // 2)
    paste_y = max(0, (target_height - resized_height) // 2)
    canvas[:, paste_y:paste_y + resized_height, paste_x:paste_x + resized_width, :] = resized[:, :, :, :3]
    return canvas.contiguous().clamp(0, 1)


def _resize_mask_contain_like(mask, source_image, target_image):
    target_width = int(target_image.shape[2])
    target_height = int(target_image.shape[1])
    source_height = int(source_image.shape[1])
    source_width = int(source_image.shape[2])
    scale = min(target_width / max(1, source_width), target_height / max(1, source_height))
    resized_width = max(1, min(target_width, int(round(source_width * scale))))
    resized_height = max(1, min(target_height, int(round(source_height * scale))))
    resized = _resize_image_exact(mask[:1], resized_width, resized_height, mode="nearest")
    canvas = torch.zeros((1, target_height, target_width, 3), device=resized.device, dtype=resized.dtype)
    paste_x = max(0, (target_width - resized_width) // 2)
    paste_y = max(0, (target_height - resized_height) // 2)
    canvas[:, paste_y:paste_y + resized_height, paste_x:paste_x + resized_width, :] = resized[:, :, :, :3]
    return canvas.contiguous().clamp(0, 1)


def _resize_mask_cover_like(mask, source_image, width: int, height: int):
    target_width = max(1, int(width))
    target_height = max(1, int(height))
    source_height = int(source_image.shape[1])
    source_width = int(source_image.shape[2])
    crop = mask[:1, :, :, :3].detach()
    if source_width * target_height > source_height * target_width:
        crop_width = max(1, min(source_width, round(source_height * target_width / target_height)))
        left = max(0, (source_width - crop_width) // 2)
        crop = crop[:, :, left:left + crop_width, :]
    elif source_width * target_height < source_height * target_width:
        crop_height = max(1, min(source_height, round(source_width * target_height / target_width)))
        top = max(0, (source_height - crop_height) // 2)
        crop = crop[:, top:top + crop_height, :, :]
    return _resize_image_exact(crop, target_width, target_height, mode="nearest")


def _resize_stage_subject_to_height(image, mask, target_height: int):
    source_height = max(1, int(image.shape[1]))
    source_width = max(1, int(image.shape[2]))
    target_height = max(1, int(target_height))
    target_width = max(1, int(round(source_width * (target_height / source_height))))
    if source_height == target_height:
        return image, _resize_mask_like(mask[:1], image[:1]) if mask is not None else None

    resized_image = _resize_image_exact(image[:1], target_width, target_height, mode="bilinear")
    resized_mask = None
    if mask is not None:
        mask_like_image = _resize_mask_like(mask[:1], image[:1])
        resized_mask = _resize_image_exact(mask_like_image, target_width, target_height, mode="nearest")
    return resized_image.contiguous(), resized_mask.contiguous() if resized_mask is not None else None


def _alpha_from_mask(mask, image):
    if mask is None:
        return torch.ones((1, int(image.shape[1]), int(image.shape[2]), 1), device=image.device, dtype=image.dtype)
    resized = _resize_mask_like(mask[:1], image[:1])
    return (resized[..., :3].max(dim=-1, keepdim=True).values > 0.1).to(device=image.device, dtype=image.dtype)


def _alpha_bbox(alpha, padding: float = 0.06):
    plane = alpha[0, :, :, 0] > 0.05
    coords = torch.nonzero(plane, as_tuple=False)
    height = int(alpha.shape[1])
    width = int(alpha.shape[2])
    if coords.numel() == 0:
        return 0, 0, width, height
    y0 = int(coords[:, 0].min().item())
    y1 = int(coords[:, 0].max().item()) + 1
    x0 = int(coords[:, 1].min().item())
    x1 = int(coords[:, 1].max().item()) + 1
    pad = int(max(width, height) * float(padding))
    return max(0, x0 - pad), max(0, y0 - pad), min(width, x1 + pad), min(height, y1 + pad)


def _subject_bbox_metrics(image, mask):
    alpha = _alpha_from_mask(mask, image)
    x0, y0, x1, y1 = _alpha_bbox(alpha)
    crop_w = max(1, int(x1 - x0))
    crop_h = max(1, int(y1 - y0))
    return {
        "crop_w": float(crop_w),
        "crop_h": float(crop_h),
        "image_w": float(max(1, int(image.shape[2]))),
        "image_h": float(max(1, int(image.shape[1]))),
    }


def _paste_subject_cutout(
    canvas,
    mask_canvas,
    image,
    mask,
    color,
    spec,
    scale_override=None,
):
    alpha = _alpha_from_mask(mask, image)
    x0, y0, x1, y1 = _alpha_bbox(alpha)
    crop = image[:, y0:y1, x0:x1, :3]
    crop_alpha = alpha[:, y0:y1, x0:x1, :]
    if crop.shape[1] <= 0 or crop.shape[2] <= 0:
        return canvas, mask_canvas

    canvas_w = int(canvas.shape[2])
    canvas_h = int(canvas.shape[1])
    target_w = max(1, int(round(canvas_w * float(spec["max_w"]))))
    target_h = max(1, int(round(canvas_h * float(spec["max_h"]))))
    if scale_override is None:
        scale = min(target_w / max(1, int(crop.shape[2])), target_h / max(1, int(crop.shape[1])))
    else:
        scale = min(
            float(scale_override),
            target_w / max(1, int(crop.shape[2])),
            target_h / max(1, int(crop.shape[1])),
        )
    new_w = max(1, min(canvas_w, int(round(crop.shape[2] * scale))))
    new_h = max(1, min(canvas_h, int(round(crop.shape[1] * scale))))

    resized_crop = F.interpolate(crop.float().movedim(-1, 1), size=(new_h, new_w), mode="bilinear", align_corners=False).movedim(1, -1)
    resized_alpha = F.interpolate(crop_alpha.float().movedim(-1, 1), size=(new_h, new_w), mode="nearest").movedim(1, -1)
    resized_crop = resized_crop.to(device=canvas.device, dtype=canvas.dtype).clamp(0, 1)
    resized_alpha = resized_alpha.to(device=canvas.device, dtype=canvas.dtype).clamp(0, 1)

    center_x = int(round(canvas_w * float(spec["x"])))
    baseline_y = int(round(canvas_h * float(spec["base"])))
    paste_x = center_x - new_w // 2
    paste_y = baseline_y - new_h
    paste_x = max(0, min(canvas_w - new_w, paste_x))
    paste_y = max(0, min(canvas_h - new_h, paste_y))

    region = canvas[:, paste_y:paste_y + new_h, paste_x:paste_x + new_w, :3]
    canvas[:, paste_y:paste_y + new_h, paste_x:paste_x + new_w, :3] = (
        resized_crop * resized_alpha + region * (1.0 - resized_alpha)
    ).contiguous()

    color_tensor = torch.tensor(color, device=mask_canvas.device, dtype=mask_canvas.dtype).view(1, 1, 1, 3)
    mask_region = mask_canvas[:, paste_y:paste_y + new_h, paste_x:paste_x + new_w, :3]
    mask_canvas[:, paste_y:paste_y + new_h, paste_x:paste_x + new_w, :3] = torch.where(
        resized_alpha > 0.05,
        color_tensor.expand_as(mask_region),
        mask_region,
    )
    return canvas.contiguous(), mask_canvas.contiguous()


def _compose_main_reference(reference_pack, *, width: int, height: int, replacement_mode: bool):
    subjects = list(reference_pack.get("subjects", []))
    if not subjects:
        raise ValueError("Reference Pack has no usable subject images.")

    scene_images = reference_pack.get("scene_images", [])
    scene_resized = _resize_image_cover(scene_images[0], width, height) if (scene_images and not replacement_mode) else None
    single_subject = len(subjects) == 1
    subject_summaries = []

    first_subject = subjects[0]
    first_image = _first_image(first_subject["images"][0], "subject_1_image")
    first_mask = (first_subject.get("masks") or [None])[0] if first_subject.get("masks") else None

    if single_subject and scene_resized is None and not replacement_mode:
        main_image = _resize_image_cover(first_image, width, height)
        main_mask_source = (
            _resize_mask_cover_like(first_mask, first_image, width, height)
            if first_mask is not None
            else None
        )
        main_mask = _colored_reference_mask(
            {
                "kind": "subject",
                "subject_index": int(first_subject.get("index", 0)),
                "mask": main_mask_source,
            },
            main_image,
            white_background=True,
        )
        subject_summaries.append(
            {
                "subject_index": int(first_subject.get("index", 0)),
                "main_shape": _shape(first_image),
                "mask_source": first_subject.get("mask_source", "full_image"),
                "canvas_role": "single_main_preserve_background_center_crop",
            }
        )
        return main_image, main_mask, scene_resized, subject_summaries

    if scene_resized is not None:
        canvas = scene_resized.clone()
        mask_canvas = _solid_color_mask_like(canvas, (1.0, 1.0, 1.0))
        base_role = "scene_background"
    elif replacement_mode:
        canvas = torch.zeros((1, int(height), int(width), 3), device=first_image.device, dtype=first_image.dtype)
        mask_canvas = _solid_color_mask_like(canvas, (0.0, 0.0, 0.0))
        base_role = "black_background"
    else:
        canvas = torch.ones((1, int(height), int(width), 3), device=first_image.device, dtype=first_image.dtype)
        mask_canvas = _solid_color_mask_like(canvas, (1.0, 1.0, 1.0))
        base_role = "white_background"

    stage_entries = []
    for subject_position, subject in enumerate(subjects):
        subject_index = int(subject.get("index", subject_position))
        images = subject.get("images", [])
        if not images:
            continue
        image = _first_image(images[0], f"subject_{subject_index + 1}_image")
        masks = subject.get("masks") or []
        mask = masks[0] if masks else None

        stage_entries.append(
            {
                "subject": subject,
                "subject_index": subject_index,
                "image": image,
                "mask": mask,
                "metrics": _subject_bbox_metrics(image, mask),
            }
        )

    if len(stage_entries) > 1:
        target_stage_height = min(
            MAX_STAGE_REFERENCE_SOURCE_HEIGHT,
            max(max(1, int(entry["image"].shape[1])) for entry in stage_entries),
        )
        for entry in stage_entries:
            image, mask = _resize_stage_subject_to_height(entry["image"], entry["mask"], target_stage_height)
            entry["image"] = image
            entry["mask"] = mask
            entry["metrics"] = _subject_bbox_metrics(image, mask)

    stage_specs, stage_scales = _layout_stage_entries(stage_entries, width, height)
    for entry, stage_spec, stage_scale in zip(stage_entries, stage_specs, stage_scales):
        subject = entry["subject"]
        subject_index = int(entry["subject_index"])
        image = entry["image"]
        mask = entry["mask"]

        canvas, mask_canvas = _paste_subject_cutout(
            canvas,
            mask_canvas,
            image,
            mask,
            _subject_color(subject_index),
            stage_spec,
            stage_scale,
        )
        subject_summaries.append(
            {
                "subject_index": subject_index,
                "main_shape": _shape(image),
                "mask_source": subject.get("mask_source", "full_image") if mask is not None else "full_image",
                "canvas_role": "stage_cutout",
                "stage_spec": stage_spec,
                "stage_scale": None if stage_scale is None else round(float(stage_scale), 6),
            }
        )

    return canvas.contiguous().clamp(0, 1), mask_canvas.contiguous().clamp(0, 1), scene_resized, [
        {"base_role": base_role, "subjects": subject_summaries}
    ]


def _mixed_reference_items(reference_pack, *, limit: int):
    items = []
    masks = reference_pack.get("reference_masks") or []
    for image_index, image in enumerate(reference_pack.get("reference_images", [])):
        items.append(
            {
                "kind": "mixed_reference",
                "image_index": int(image_index),
                "image": image,
                "mask": masks[image_index] if image_index < len(masks) else None,
                "mask_source": reference_pack.get("reference_mask_source", "full_image") if image_index < len(masks) else "full_image",
                "role": "prefix",
            }
        )
    dropped = max(0, len(items) - max(0, int(limit)))
    return items[: max(0, int(limit))], dropped


def _prepare_mixed_reference(
    item,
    *,
    width: int,
    height: int,
    replacement_mode: bool,
    scene_image=None,
):
    source_image = item["image"]
    if replacement_mode:
        image = _resize_image_contain(source_image, width, height)
        fitted_mask = _resize_mask_contain_like(item.get("mask"), source_image, image) if item.get("mask") is not None else None
    else:
        image = _resize_image_cover(source_image, width, height)
        fitted_mask = _resize_mask_cover_like(item.get("mask"), source_image, width, height) if item.get("mask") is not None else None
    alpha = _alpha_from_mask(fitted_mask, image) if fitted_mask is not None else None

    alpha_cropped = False
    scene_composited = False
    if alpha is not None and replacement_mode:
        image = (image * alpha.to(device=image.device, dtype=image.dtype)).contiguous()
        alpha_cropped = True
    elif alpha is not None and scene_image is not None:
        scene = scene_image.to(device=image.device, dtype=image.dtype)
        if scene.shape[1] != image.shape[1] or scene.shape[2] != image.shape[2]:
            scene = _resize_image_exact(scene, int(image.shape[2]), int(image.shape[1]))
        alpha = alpha.to(device=image.device, dtype=image.dtype)
        image = (image * alpha + scene * (1.0 - alpha)).contiguous()
        scene_composited = True

    if fitted_mask is not None:
        reference_mask = _mask_with_background(
            fitted_mask,
            white_background=not replacement_mode,
        )
    else:
        reference_mask = _solid_color_mask_like(image, (1.0, 1.0, 1.0) if not replacement_mode else (0.0, 0.0, 0.0))
    summary = {
        "kind": "mixed_reference",
        "conditioning_role": "prefix",
        "image_index": int(item["image_index"]),
        "source_shape": _shape(item["image"]),
        "mask_source": item.get("mask_source", "full_image"),
        "alpha_cropped": bool(alpha_cropped),
        "scene_composited": bool(scene_composited),
    }
    return image.contiguous().clamp(0, 1), reference_mask.contiguous().clamp(0, 1), summary


def _encode_reference_sequence(vae, images, masks, *, mask_condition_enabled: bool):
    latents = []
    mask_latents = []
    latent_shapes = []
    for image, mask in zip(images, masks):
        latent = vae.encode(image[:1, :, :, :3])
        if latent.ndim != 5:
            raise RuntimeError(f"SCAIL-2 reference VAE encode returned unexpected shape: {_shape(latent)}")
        latents.append(latent)
        latent_shapes.append(_shape(latent))
        if mask_condition_enabled:
            mask_latents.append(_extract_scail_mask_to_28ch(mask[:1, :, :, :3]))

    if not latents:
        raise ValueError("Reference Pack has no reference images to encode.")
    reference_latent = torch.cat(latents, dim=2).detach().cpu().contiguous()
    reference_mask = torch.cat(mask_latents, dim=1).detach().cpu().contiguous() if mask_condition_enabled and mask_latents else None
    return reference_latent, reference_mask, latent_shapes


def _prepare_scail_reference_pack(
    *,
    vae,
    reference_pack: dict,
    width: int,
    height: int,
    replacement_mode: bool,
):
    background_images = len(reference_pack.get("scene_images", []))
    include_background = not replacement_mode
    background_enabled = bool(include_background and background_images > 0)
    subject_count = len(reference_pack.get("subjects", []))
    subject_image_count = sum(1 for subject in reference_pack.get("subjects", []) if subject.get("images"))
    mixed_reference_count = len(reference_pack.get("reference_images", []))
    multi_reference = bool(mixed_reference_count > 0 or background_enabled or subject_count > 1)
    has_subject_masks = any(subject.get("masks") for subject in reference_pack.get("subjects", []))
    has_reference_masks = bool(reference_pack.get("reference_masks"))
    mask_condition_enabled = bool(replacement_mode or subject_count > 1)

    if mask_condition_enabled:
        missing_masks = []
        for subject_position, subject in enumerate(reference_pack.get("subjects", [])):
            images = subject.get("images", [])
            masks = subject.get("masks") or []
            for image_index in range(len(images)):
                if image_index >= len(masks) or masks[image_index] is None:
                    missing_masks.append((subject_position, image_index))
        reference_masks = reference_pack.get("reference_masks") or []
        for image_index, _image in enumerate(reference_pack.get("reference_images", [])):
            if image_index >= len(reference_masks) or reference_masks[image_index] is None:
                missing_masks.append(("reference", image_index))
        if missing_masks:
            raise ValueError(
                "This Reference Pack needs subject/reference masks. Connect Reference Pack through "
                "SCAIL-2 Reference SAM Builder before SCAIL-2 Simple Video."
            )

    main_image, main_mask, scene_resized, main_summary = _compose_main_reference(
        reference_pack,
        width=width,
        height=height,
        replacement_mode=replacement_mode,
    )

    prefix_budget = MAX_PREFIX_REFERENCE_IMAGES
    mixed_items, dropped_mixed = _mixed_reference_items(reference_pack, limit=prefix_budget)
    reference_images = [main_image]
    reference_masks = [main_mask]
    item_summaries = [
        {
            "kind": "main_reference",
            "conditioning_role": "primary",
            "subjects": int(subject_count),
            "source": main_summary,
            "background_composited": bool(scene_resized is not None),
            "mask_background": "black" if replacement_mode else "white",
        }
    ]

    for item in mixed_items:
        image, mask, summary_item = _prepare_mixed_reference(
            item,
            width=width,
            height=height,
            replacement_mode=replacement_mode,
            scene_image=scene_resized,
        )
        reference_images.append(image)
        reference_masks.append(mask)
        item_summaries.append(summary_item)

    # Scene images are folded into the primary reference collage. Feeding them
    # as a separate background-only prefix can pull attention on the first frame.
    background_prefix_used = False

    ref_latent, ref_mask_1f, latent_shapes = _encode_reference_sequence(
        vae,
        reference_images,
        reference_masks,
        mask_condition_enabled=mask_condition_enabled,
    )

    summary = {
        "reference_items": len(reference_images),
        "subject_items": int(subject_image_count),
        "mixed_reference_items": int(mixed_reference_count),
        "additional_items": max(0, len(reference_images) - 1),
        "primary_items": 1,
        "background_images": int(background_images),
        "background_enabled": bool(background_enabled),
        "background_prefix_used": bool(background_prefix_used),
        "background_ignored": bool(replacement_mode and background_images > 0),
        "background_fit": "center_crop_cover" if background_enabled else None,
        "mixed_reference_used": int(len(mixed_items)),
        "mixed_reference_dropped": int(dropped_mixed),
        "max_prefix_reference_images": int(MAX_PREFIX_REFERENCE_IMAGES),
        "generation_width": int(width),
        "generation_height": int(height),
        "reference_pixel_frames": int(len(reference_images)),
        "reference_latent_shape": _shape(ref_latent),
        "reference_latent_parts": latent_shapes,
        "reference_encoding": "main_collage_plus_single_frame_prefix_latents",
        "reference_mask_enabled": bool(mask_condition_enabled),
        "reference_mask_reason": (
            None
            if not mask_condition_enabled
            else (
                "replacement"
                if replacement_mode
                else ("multi_subject" if subject_count > 1 else "reference_sam_masks")
            )
        ),
        "items": item_summaries,
    }
    return {
        "reference_latent": ref_latent,
        "reference_mask_prefix": ref_mask_1f,
        "clip_image": main_image,
        "summary": summary,
    }


def _set_scail_reference_pack_conditioning(
    *,
    positive,
    negative,
    vae,
    latent,
    reference_pack: dict,
    width: int,
    height: int,
    replacement_mode: bool,
    prepared_reference_pack=None,
):
    import node_helpers

    prepared = prepared_reference_pack or _prepare_scail_reference_pack(
        vae=vae,
        reference_pack=reference_pack,
        width=width,
        height=height,
        replacement_mode=replacement_mode,
    )
    ref_latent = prepared["reference_latent"]
    positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [ref_latent]}, append=True)
    negative = node_helpers.conditioning_set_values(negative, {"reference_latents": [ref_latent]}, append=True)

    ref_mask_1f = prepared.get("reference_mask_prefix")
    ref_mask_shape = None
    if ref_mask_1f is not None:
        zeros = torch.zeros(
            (1, latent.shape[2], 28, ref_mask_1f.shape[-2], ref_mask_1f.shape[-1]),
            device=ref_mask_1f.device,
            dtype=ref_mask_1f.dtype,
        )
        ref_mask_28ch = torch.cat([ref_mask_1f, zeros], dim=1)
        positive = node_helpers.conditioning_set_values(positive, {"ref_mask_28ch": ref_mask_28ch})
        negative = node_helpers.conditioning_set_values(negative, {"ref_mask_28ch": ref_mask_28ch})
        ref_mask_shape = _shape(ref_mask_28ch)

    summary = dict(prepared["summary"])
    summary["reference_mask_shape"] = ref_mask_shape
    return positive, negative, summary


def _run_multi_ref_scail_chunk(
    *,
    model,
    positive,
    negative,
    vae,
    sampler,
    sigmas,
    reference_pack: dict,
    pose_video,
    clip_vision_output,
    pose_video_mask,
    previous_frames,
    width: int,
    height: int,
    length: int,
    video_frame_offset: int,
    previous_frame_count: int,
    replacement_mode: bool,
    seed: int,
    cfg: float,
    pose_strength: float,
    prepared_reference_pack=None,
):
    import comfy.model_management
    import comfy.utils
    import node_helpers

    latent = torch.zeros(
        [1, 16, ((length - 1) // 4) + 1, height // 8, width // 8],
        device=comfy.model_management.intermediate_device(),
    )
    noise_mask = None

    ref_mask_flag = not replacement_mode
    positive = node_helpers.conditioning_set_values(positive, {"ref_mask_flag": ref_mask_flag})
    negative = node_helpers.conditioning_set_values(negative, {"ref_mask_flag": ref_mask_flag})

    prev_trimmed = None
    if previous_frames is not None and previous_frames.shape[0] > 0:
        prev_trimmed = previous_frames[-previous_frame_count:]
        video_frame_offset = max(0, int(video_frame_offset) - int(prev_trimmed.shape[0]))

    positive, negative, ref_summary = _set_scail_reference_pack_conditioning(
        positive=positive,
        negative=negative,
        vae=vae,
        latent=latent,
        reference_pack=reference_pack,
        width=width,
        height=height,
        replacement_mode=replacement_mode,
        prepared_reference_pack=prepared_reference_pack,
    )

    if clip_vision_output is not None:
        positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
        negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

    if pose_video is not None:
        pose_video = None if pose_video.shape[0] <= video_frame_offset else pose_video[video_frame_offset:]
    if pose_video_mask is not None:
        pose_video_mask = None if pose_video_mask.shape[0] <= video_frame_offset else pose_video_mask[video_frame_offset:]

    ts = [v.shape[0] for v in (pose_video, pose_video_mask) if v is not None]
    if ts:
        kept = ((min(min(ts), length) - 1) // 4) * 4 + 1
        if pose_video is not None:
            pose_video = pose_video[:kept]
        if pose_video_mask is not None:
            pose_video_mask = pose_video_mask[:kept]

    if pose_video is not None:
        pose_video = comfy.utils.common_upscale(
            pose_video[:length].movedim(-1, 1),
            width // 2,
            height // 2,
            "area",
            "center",
        ).movedim(1, -1)
        pose_video_latent = vae.encode(pose_video[:, :, :, :3]) * pose_strength
        positive = node_helpers.conditioning_set_values_with_timestep_range(
            positive, {"pose_video_latent": pose_video_latent}, 0.0, 1.0
        )
        negative = node_helpers.conditioning_set_values_with_timestep_range(
            negative, {"pose_video_latent": pose_video_latent}, 0.0, 1.0
        )

    if pose_video_mask is not None:
        mask_video_hw = comfy.utils.common_upscale(
            pose_video_mask[:length].movedim(-1, 1),
            width // 2,
            height // 2,
            "area",
            "center",
        ).movedim(1, -1)
        driving_mask_28ch = _extract_scail_mask_to_28ch(mask_video_hw)
        positive = node_helpers.conditioning_set_values(positive, {"driving_mask_28ch": driving_mask_28ch})
        negative = node_helpers.conditioning_set_values(negative, {"driving_mask_28ch": driving_mask_28ch})

    if prev_trimmed is not None:
        previous = comfy.utils.common_upscale(prev_trimmed.movedim(-1, 1), width, height, "bicubic", "center").movedim(1, -1)
        prev_latent = vae.encode(previous[:, :, :, :3])
        prev_latent_frames = min(prev_latent.shape[2], latent.shape[2])
        latent[:, :, :prev_latent_frames] = prev_latent[:, :, :prev_latent_frames].to(latent.dtype)
        noise_mask = torch.ones(
            (1, 1, latent.shape[2], latent.shape[-2], latent.shape[-1]),
            device=latent.device,
            dtype=latent.dtype,
        )
        noise_mask[:, :, :prev_latent_frames] = 0.0

    out_latent = {"samples": latent}
    if noise_mask is not None:
        out_latent["noise_mask"] = noise_mask

    latent_to_decode = _sample_for_decode(
        model=model,
        positive=positive,
        negative=negative,
        sampler=sampler,
        sigmas=sigmas,
        latent=out_latent,
        seed=seed,
        cfg=cfg,
    )
    positive = negative = out_latent = latent = noise_mask = None
    reference_pack = clip_vision_output = prepared_reference_pack = None
    pose_video = pose_video_mask = pose_video_latent = mask_video_hw = driving_mask_28ch = None
    previous = prev_latent = prev_trimmed = None
    _empty_cache(force=True)
    frames = _decode_latent_to_frames(vae, latent_to_decode)
    summary = {
        "width": int(width),
        "height": int(height),
        "length": int(length),
        "seed": int(seed),
        "cfg": float(cfg),
        "input_video_frame_offset": int(video_frame_offset),
        "output_video_frame_offset": int(video_frame_offset + length),
        "decoded_shape": _shape(frames),
        "reference_pack": ref_summary,
    }
    return frames, int(video_frame_offset + length), summary


def _set_scail_single_reference_conditioning(
    *,
    positive,
    negative,
    vae,
    latent,
    reference_image,
    reference_image_mask,
    width: int,
    height: int,
    replacement_mode: bool,
):
    import comfy.utils
    import node_helpers

    ref_pixels = comfy.utils.common_upscale(
        reference_image[:1].movedim(-1, 1),
        width,
        height,
        "bilinear",
        "center",
    ).movedim(1, -1)
    ref_mask = None
    if replacement_mode and reference_image_mask is not None:
        ref_mask = comfy.utils.common_upscale(
            reference_image_mask[:1].movedim(-1, 1),
            width,
            height,
            "nearest-exact",
            "center",
        ).movedim(1, -1)
        alpha = (ref_mask[..., :3].max(dim=-1, keepdim=True).values > 0.1).to(dtype=ref_pixels.dtype)
        ref_pixels = ref_pixels * alpha

    ref_latent = vae.encode(ref_pixels[:, :, :, :3])
    positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [ref_latent]}, append=True)
    negative = node_helpers.conditioning_set_values(negative, {"reference_latents": [torch.zeros_like(ref_latent)]}, append=True)

    ref_mask_shape = None
    if replacement_mode and ref_mask is not None:
        ref_mask_1f = _extract_scail_mask_to_28ch(ref_mask)
        zeros = torch.zeros(
            (1, latent.shape[2], 28, ref_mask_1f.shape[-2], ref_mask_1f.shape[-1]),
            device=ref_mask_1f.device,
            dtype=ref_mask_1f.dtype,
        )
        ref_mask_28ch = torch.cat([ref_mask_1f, zeros], dim=1)
        positive = node_helpers.conditioning_set_values(positive, {"ref_mask_28ch": ref_mask_28ch})
        negative = node_helpers.conditioning_set_values(negative, {"ref_mask_28ch": ref_mask_28ch})
        ref_mask_shape = _shape(ref_mask_28ch)

    summary = {
        "reference_latent_shape": _shape(ref_latent),
        "reference_mask_shape": ref_mask_shape,
    }
    return positive, negative, summary


def _set_scail_pose_conditioning(
    *,
    positive,
    negative,
    vae,
    pose_video,
    pose_video_mask,
    width: int,
    height: int,
    length: int,
    pose_strength: float,
):
    import comfy.utils
    import node_helpers

    if pose_video.shape[0] < length:
        raise ValueError("pose_video is shorter than the requested generation length.")
    if pose_video_mask is not None and pose_video_mask.shape[0] < length:
        raise ValueError("pose_video_mask is shorter than the requested generation length.")

    pose_video = pose_video[:length]
    pose_video_hw = comfy.utils.common_upscale(
        pose_video.movedim(-1, 1),
        width // 2,
        height // 2,
        "area",
        "center",
    ).movedim(1, -1)
    pose_video_latent = vae.encode(pose_video_hw[:, :, :, :3]) * pose_strength
    positive = node_helpers.conditioning_set_values_with_timestep_range(
        positive,
        {"pose_video_latent": pose_video_latent},
        0.0,
        1.0,
    )
    negative = node_helpers.conditioning_set_values_with_timestep_range(
        negative,
        {"pose_video_latent": pose_video_latent},
        0.0,
        1.0,
    )

    driving_mask_shape = None
    if pose_video_mask is not None:
        mask_video_hw = comfy.utils.common_upscale(
            pose_video_mask[:length].movedim(-1, 1),
            width // 2,
            height // 2,
            "area",
            "center",
        ).movedim(1, -1)
        driving_mask_28ch = _extract_scail_mask_to_28ch(mask_video_hw)
        positive = node_helpers.conditioning_set_values(positive, {"driving_mask_28ch": driving_mask_28ch})
        negative = node_helpers.conditioning_set_values(negative, {"driving_mask_28ch": driving_mask_28ch})
        driving_mask_shape = _shape(driving_mask_28ch)

    summary = {
        "pose_video_latent_shape": _shape(pose_video_latent),
        "driving_mask_shape": driving_mask_shape,
    }
    return positive, negative, summary


def _run_context_scail(
    *,
    model,
    positive,
    negative,
    vae,
    sampler,
    sigmas,
    reference_image,
    reference_pack,
    pose_video,
    clip_vision_output,
    pose_video_mask,
    reference_image_mask,
    width: int,
    height: int,
    length: int,
    context_frames: int,
    context_overlap_frames: int,
    replacement_mode: bool,
    seed: int,
    cfg: float,
    pose_strength: float,
    prepared_reference_pack=None,
):
    import comfy.model_management
    import node_helpers

    latent = torch.zeros(
        [1, 16, ((length - 1) // 4) + 1, height // 8, width // 8],
        device=comfy.model_management.intermediate_device(),
    )
    ref_mask_flag = not replacement_mode
    positive = node_helpers.conditioning_set_values(positive, {"ref_mask_flag": ref_mask_flag})
    negative = node_helpers.conditioning_set_values(negative, {"ref_mask_flag": ref_mask_flag})

    if reference_pack is not None:
        positive, negative, reference_summary = _set_scail_reference_pack_conditioning(
            positive=positive,
            negative=negative,
            vae=vae,
            latent=latent,
            reference_pack=reference_pack,
            width=width,
            height=height,
            replacement_mode=replacement_mode,
            prepared_reference_pack=prepared_reference_pack,
        )
    elif reference_image is not None:
        positive, negative, reference_summary = _set_scail_single_reference_conditioning(
            positive=positive,
            negative=negative,
            vae=vae,
            latent=latent,
            reference_image=reference_image,
            reference_image_mask=reference_image_mask,
            width=width,
            height=height,
            replacement_mode=replacement_mode,
        )
    else:
        raise ValueError("reference_image or reference_pack is required.")

    if clip_vision_output is not None:
        positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
        negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

    positive, negative, pose_summary = _set_scail_pose_conditioning(
        positive=positive,
        negative=negative,
        vae=vae,
        pose_video=pose_video,
        pose_video_mask=pose_video_mask,
        width=width,
        height=height,
        length=length,
        pose_strength=pose_strength,
    )

    context_model, context_summary = apply_scail2_easy_context(
        model,
        context_frames=context_frames,
        context_overlap_frames=context_overlap_frames,
    )
    latent_to_decode = _sample_for_decode(
        model=context_model,
        positive=positive,
        negative=negative,
        sampler=sampler,
        sigmas=sigmas,
        latent={"samples": latent},
        seed=seed,
        cfg=cfg,
    )
    positive = negative = latent = pose_video = pose_video_mask = None
    reference_image = reference_pack = reference_image_mask = clip_vision_output = prepared_reference_pack = None
    context_model = None
    _empty_cache(force=True)
    frames = _decode_latent_to_frames(vae, latent_to_decode)
    summary = {
        "width": int(width),
        "height": int(height),
        "length": int(length),
        "seed": int(seed),
        "cfg": float(cfg),
        "decoded_shape": _shape(frames),
        "context": context_summary,
        "reference": reference_summary,
        "pose": pose_summary,
    }
    return frames, summary


class SFSCAIL2ReferencePack:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "subject_count": ("INT", {"default": 1, "min": 1, "max": MAX_REFERENCE_SUBJECTS, "step": 1, "tooltip": "主体数量（前端据此动态显示 主体N 输入，最多 6 个）"}),
            "reference_count": ("INT", {"default": 0, "min": 0, "max": MAX_MIXED_REFERENCE_IMAGES, "step": 1, "tooltip": "补充参考图数量（前端据此动态显示 参考图N 输入，最多 5 张）"}),
        }

        optional = {}
        for subject_index in range(1, MAX_REFERENCE_SUBJECTS + 1):
            optional[f"subject_{subject_index}_image"] = ("IMAGE", {"tooltip": f"主体{subject_index} 的主参考图（清晰、完整；多主体时按驱动视频里目标人物的顺序排列更稳定）"})
        for reference_index in range(1, MAX_MIXED_REFERENCE_IMAGES + 1):
            optional[f"reference_{reference_index}"] = ("IMAGE", {"tooltip": f"补充参考图{reference_index}（不同角度、服装细节、剧情画面或多人画面）"})
        optional["scene_image"] = ("IMAGE", {"tooltip": "可选背景图：animation 模式用于指定背景；replacement 模式忽略"})

        for subject_index in range(1, MAX_REFERENCE_SUBJECTS + 1):
            for image_index in range(1, MAX_LEGACY_REFERENCE_IMAGES_PER_SUBJECT + 1):
                optional[f"subject_{subject_index}_image_{image_index}"] = ("IMAGE", {"tooltip": f"主体{subject_index} 的第 {image_index} 张参考图（旧版输入，兼容保留）"})

        return {"required": required, "optional": optional}

    RETURN_TYPES = (REFERENCE_PACK_TYPE,)
    RETURN_NAMES = ("reference_pack",)
    FUNCTION = "pack"
    CATEGORY = _CATEGORY
    DESCRIPTION = "把多主体主图 / 补充参考图 / 场景图打包成 SCAIL2_REFERENCE_PACK，供 SF SCAIL-2 Simple Video 使用（前端按数量动态增删槽位）"

    def pack(self, subject_count: int, reference_count: int = 0, scene_image=None, **kwargs):
        active_subjects = _clamp_int(subject_count, 1, MAX_REFERENCE_SUBJECTS)
        active_references = _clamp_int(reference_count, 0, MAX_MIXED_REFERENCE_IMAGES)
        subjects = []
        connected_subject_indices = []

        for subject_index in range(1, active_subjects + 1):
            name = f"subject_{subject_index}_image"
            image = kwargs.get(name)
            legacy_name = f"subject_{subject_index}_image_1"
            if image is None:
                image = kwargs.get(legacy_name)
                name = legacy_name
            if image is None:
                continue
            first = _first_image(image, name)
            subjects.append({"index": len(subjects), "source_subject_index": subject_index - 1, "images": [first]})
            connected_subject_indices.append(subject_index)

        if not subjects:
            raise ValueError("Reference Pack needs at least one connected subject image.")

        reference_images = []
        for reference_index in range(1, active_references + 1):
            name = f"reference_{reference_index}"
            image = kwargs.get(name)
            if image is not None:
                reference_images.append(_first_image(image, name))

        if not reference_images:
            for subject_index in connected_subject_indices:
                for image_index in range(2, MAX_LEGACY_REFERENCE_IMAGES_PER_SUBJECT + 1):
                    legacy_name = f"subject_{subject_index}_image_{image_index}"
                    image = kwargs.get(legacy_name)
                    if image is not None:
                        reference_images.append(_first_image(image, legacy_name))
                    if len(reference_images) >= MAX_MIXED_REFERENCE_IMAGES:
                        break
                if len(reference_images) >= MAX_MIXED_REFERENCE_IMAGES:
                    break

        scene_images = []
        if scene_image is not None:
            scene_images.append(_first_image(scene_image, "scene_image"))

        reference_pack = {
            "type": REFERENCE_PACK_TYPE,
            "version": 3,
            "subjects": subjects,
            "reference_images": reference_images,
            "scene_images": scene_images,
            "subject_count": len(subjects),
            "configured_subject_count": active_subjects,
            "reference_count": len(reference_images),
            "max_subjects": MAX_REFERENCE_SUBJECTS,
            "max_reference_images": MAX_MIXED_REFERENCE_IMAGES,
        }
        return (reference_pack,)


class SFSCAIL2ReferenceSAMBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_pack": (REFERENCE_PACK_TYPE, {"tooltip": "来自 SF SCAIL-2 Reference Pack 的参考包"}),
                "sam_model": ("MODEL", {"tooltip": "SAM3 模型（通常由 Checkpoint 加载）"}),
                "conditioning": ("CONDITIONING", {"tooltip": "SAM3 文本条件，通常为 CLIPTextEncode('person')"}),
                "detection_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "文本检测得分阈值（越高越严格）"}),
                "max_objects": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1, "tooltip": "每张参考图最多追踪的对象数"}),
                "detect_interval": ("INT", {"default": 1, "min": 1, "max": 999, "step": 1, "tooltip": "检测间隔帧数（1=每帧检测；越大越省算力）"}),
            },
        }

    RETURN_TYPES = (REFERENCE_PACK_TYPE, "STRING")
    RETURN_NAMES = ("reference_pack", "summary")
    FUNCTION = "build"
    CATEGORY = _CATEGORY
    DESCRIPTION = "对 Reference Pack 的每张主体/参考图跑 SAM3_VideoTrack，生成 SCAIL2ColoredMask 需要的彩色参考蒙版后回填进 reference_pack"

    def build(
        self,
        reference_pack: dict,
        sam_model,
        conditioning,
        detection_threshold: float,
        max_objects: int,
        detect_interval: int,
    ):
        if not _is_reference_pack(reference_pack):
            raise ValueError("reference_pack must come from SCAIL-2 Reference Pack.")

        subjects = []
        summary_subjects = []
        for subject_position, subject in enumerate(reference_pack.get("subjects", [])):
            subject_index = int(subject.get("index", subject_position))
            images = list(subject.get("images", []))[:1]
            masks = []
            image_summaries = []
            for image_index, image in enumerate(images):
                first = _first_image(image, f"subject_{subject_index + 1}_image")
                track = _run_sam3_track(
                    first,
                    sam_model,
                    conditioning,
                    detection_threshold=float(detection_threshold),
                    max_objects=int(max_objects),
                    detect_interval=int(detect_interval),
                )
                mask = _render_reference_sam_mask(first, track, subject_index)
                masks.append(mask.detach().contiguous())
                image_summaries.append(
                    {
                        "image_index": int(image_index),
                        "image_shape": _shape(first),
                        "mask_shape": _shape(mask),
                    }
                )

            enriched_subject = dict(subject)
            enriched_subject["images"] = images
            enriched_subject["masks"] = masks
            enriched_subject["mask_source"] = "sam3"
            subjects.append(enriched_subject)
            summary_subjects.append(
                {
                    "subject": int(subject_index + 1),
                    "reference_images": len(images),
                    "masks": len(masks),
                    "images": image_summaries,
                }
            )

        reference_images = []
        reference_masks = []
        summary_references = []
        for reference_index, image in enumerate(reference_pack.get("reference_images", [])):
            first = _first_image(image, f"reference_{reference_index + 1}")
            track = _run_sam3_track(
                first,
                sam_model,
                conditioning,
                detection_threshold=float(detection_threshold),
                max_objects=int(max_objects),
                detect_interval=int(detect_interval),
            )
            mask = _render_mixed_reference_sam_mask(first, track)
            reference_images.append(first)
            reference_masks.append(mask.detach().contiguous())
            summary_references.append(
                {
                    "reference": int(reference_index + 1),
                    "image_shape": _shape(first),
                    "mask_shape": _shape(mask),
                    "objects": _track_object_count(track),
                }
            )

        enriched_pack = dict(reference_pack)
        enriched_pack["version"] = 4
        enriched_pack["subjects"] = subjects
        enriched_pack["reference_images"] = reference_images
        enriched_pack["reference_masks"] = reference_masks
        enriched_pack["reference_mask_source"] = "sam3"

        summary = {
            "type": REFERENCE_PACK_TYPE,
            "reference_mask_source": "sam3",
            "subject_count": len(subjects),
            "subjects": summary_subjects,
            "mixed_references": summary_references,
            "scene_images": len(reference_pack.get("scene_images", [])),
            "settings": {
                "detection_threshold": float(detection_threshold),
                "max_objects": int(max_objects),
                "detect_interval": int(detect_interval),
            },
        }
        return (enriched_pack, json.dumps(summary, indent=2))


class SFSCAIL2SimpleVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "SCAIL-2 扩散模型"}),
                "positive": ("CONDITIONING", {"tooltip": "正向条件"}),
                "negative": ("CONDITIONING", {"tooltip": "负向条件"}),
                "vae": ("VAE", {"tooltip": "Wan / SCAIL-2 的 VAE"}),
                "sampler": ("SAMPLER", {"tooltip": "采样器"}),
                "sigmas": ("SIGMAS", {"tooltip": "采样 sigmas 调度"}),
                "reference_image": (REFERENCE_IMAGE_INPUT_TYPE, {"tooltip": "参考图 IMAGE 或 SF SCAIL-2 Reference Pack：IMAGE=单图参考；Pack=多主体/多参考图（需先经 SAM Builder 生成蒙版）"}),
                "pose_video": ("IMAGE", {"tooltip": "驱动视频帧序列（其尺寸决定生成尺寸，内部对齐 32）"}),
                "clip_vision": ("CLIP_VISION", {"tooltip": "CLIP 视觉编码器（编码主参考图）"}),
                "seed": ("INT", {"default": 1, "min": 0, "max": 0xffffffffffffffff, "tooltip": "随机种子（分块时逐块 +1）"}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1, "tooltip": "CFG 引导系数"}),
                "mode": (["replacement", "animation"], {"default": "replacement", "tooltip": "模式：replacement=角色替换（需驱动/参考 track_data）；animation=动作迁移"}),
                "advanced": ("BOOLEAN", {"default": False, "tooltip": "展开高级参数（长视频方式/分段/上下文等）"}),
                "long_video_mode": (list(LONG_VIDEO_MODES), {"default": "chunk", "tooltip": "长视频方式：chunk=分段接续（逐段衔接上一段尾部帧）；context_sampling=上下文窗口采样"}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1, "tooltip": "最大生成帧数（0=不额外限制，按输入视频帧数）"}),
                "chunk_frames": ("INT", {"default": 81, "min": 17, "max": 321, "step": 4, "tooltip": "每段生成帧数（自动对齐 4n+1，如 81）"}),
                "overlap_frames": ("INT", {"default": 5, "min": 0, "max": 33, "step": 1, "tooltip": "段间重叠帧数（作为上一段历史上下文，0=不重叠）"}),
                "color_correction": ("BOOLEAN", {"default": False, "tooltip": "开启段间重叠区域色彩校正（减少接缝色差；overlap=0 时无效）"}),
                "context_frames": ("INT", {"default": 81, "min": 17, "max": 321, "step": 4, "tooltip": "上下文窗口帧数（context_sampling 模式，自动对齐 4n+1）"}),
                "context_overlap_frames": ("INT", {"default": 20, "min": 0, "max": 320, "step": 1, "tooltip": "上下文窗口重叠帧数（context_sampling 模式，0=不重叠）"}),
            },
            "optional": {
                "driving_track_data": ("SAM3_TRACK_DATA", {"tooltip": "驱动视频的 SAM3 追踪数据（replacement 模式必需；多主体 Reference Pack 也必需）"}),
                "reference_track_data": ("SAM3_TRACK_DATA", {"tooltip": "参考图的 SAM3 追踪数据（仅单图 replacement 模式需要）"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frames", "summary")
    FUNCTION = "generate"
    CATEGORY = _CATEGORY
    DESCRIPTION = "SCAIL-2 主生成节点：animation 动作迁移 / replacement 角色替换；内部自动分段拼接长视频，参考包（多主体/多参考图）自动生成 colored mask 与条件"

    def generate(
        self,
        model,
        positive,
        negative,
        vae,
        sampler,
        sigmas,
        reference_image,
        pose_video: torch.Tensor,
        clip_vision,
        seed: int,
        cfg: float,
        mode: str,
        long_video_mode: str = "chunk",
        advanced: bool = False,
        max_frames: int = 0,
        chunk_frames: int = 81,
        overlap_frames: int = 5,
        color_correction: bool = False,
        context_frames: int = 81,
        context_overlap_frames: int = 20,
        driving_track_data=None,
        reference_track_data=None,
    ):
        reference_is_pack = _is_reference_pack(reference_image)
        if pose_video.ndim != 4:
            raise ValueError("pose_video must be a ComfyUI IMAGE tensor.")
        if not reference_is_pack and (not isinstance(reference_image, torch.Tensor) or reference_image.ndim != 4):
            raise ValueError("reference_image must be a ComfyUI IMAGE tensor or a SCAIL-2 Reference Pack.")

        total_frames = int(pose_video.shape[0])
        if max_frames > 0:
            total_frames = min(total_frames, int(max_frames))
        if total_frames <= 0:
            raise ValueError("pose_video has no frames.")

        mask_replacement_mode = mode == "replacement"
        pose_video_mask = None
        reference_image_mask = None
        if reference_is_pack:
            subject_count = len(reference_image.get("subjects", []))
            reference_pack_identity_mask_required = bool(mask_replacement_mode or subject_count > 1)
            use_optional_animation_driving_mask = bool(
                not mask_replacement_mode
                and subject_count == 1
                and driving_track_data is not None
            )
            if reference_pack_identity_mask_required or use_optional_animation_driving_mask:
                if driving_track_data is None:
                    raise ValueError(
                        "replacement or multi-subject Reference Pack requires driving_track_data from SAM3_VideoTrack."
                    )
                driving_object_count = _track_object_count(driving_track_data)
                if subject_count > 1 and driving_object_count < subject_count:
                    raise ValueError(
                        "Reference Pack has "
                        f"{subject_count} subjects, but driving_track_data only contains "
                        f"{driving_object_count} tracked object(s). Increase SAM3_VideoTrack.max_objects "
                        "for the driving video and make sure each target person is detected."
                    )
                pose_video_mask = _create_driving_scail_mask(driving_track_data, mask_replacement_mode)
        elif mask_replacement_mode:
            if driving_track_data is None or reference_track_data is None:
                raise ValueError("replacement mode requires driving_track_data and reference_track_data from SAM3_VideoTrack.")
            pose_video_mask, reference_image_mask = _create_scail_masks(
                driving_track_data,
                reference_track_data,
                True,
            )
        pose_video = _trim_frames(pose_video, total_frames)
        pose_video_mask = _trim_frames(pose_video_mask, total_frames)
        width, height = _infer_generation_size(pose_video)
        reference_pack_cache = {}
        if reference_is_pack:
            reference_pack = reference_image
            initial_prepared_reference_pack = _prepare_scail_reference_pack(
                vae=vae,
                reference_pack=reference_pack,
                width=width,
                height=height,
                replacement_mode=mask_replacement_mode,
            )
            reference_pack_cache[mask_replacement_mode] = initial_prepared_reference_pack
            clip_vision_output = _encode_clip_vision(
                clip_vision,
                initial_prepared_reference_pack["clip_image"],
            )
        else:
            reference_pack = None
            initial_prepared_reference_pack = None
            reference_image = reference_image[:1].detach()
            if reference_image_mask is not None:
                reference_image_mask = reference_image_mask[:1].detach()
                reference_image = _apply_reference_mask(reference_image, reference_image_mask)
            clip_vision_output = _encode_clip_vision(clip_vision, reference_image)

        if long_video_mode not in LONG_VIDEO_MODES:
            long_video_mode = "chunk"
        pose_strength = 1.0

        if long_video_mode == "context_sampling":
            generation_length = _wan_frame_count_floor(total_frames)
            if generation_length <= 0:
                raise ValueError("pose_video has no usable 4n+1 frame range.")
            context_frames = _wan_frame_count_cover(max(17, int(context_frames)))
            requested_context_overlap = max(0, int(context_overlap_frames))
            context_overlap_frames = (
                0
                if requested_context_overlap == 0
                else min(_wan_frame_count_floor(requested_context_overlap), max(0, context_frames - 4))
            )
            context_pose_video = pose_video[:generation_length].contiguous()
            context_pose_video_mask = _trim_frames(pose_video_mask, generation_length)
            prepared_reference_pack = initial_prepared_reference_pack
            frames, context_summary = _run_context_scail(
                model=model,
                positive=positive,
                negative=negative,
                vae=vae,
                sampler=sampler,
                sigmas=sigmas,
                reference_image=None if reference_pack is not None else reference_image,
                reference_pack=reference_pack,
                pose_video=context_pose_video,
                clip_vision_output=clip_vision_output,
                pose_video_mask=context_pose_video_mask,
                reference_image_mask=reference_image_mask,
                width=width,
                height=height,
                length=generation_length,
                context_frames=context_frames,
                context_overlap_frames=context_overlap_frames,
                replacement_mode=mask_replacement_mode,
                seed=int(seed),
                cfg=float(cfg),
                pose_strength=float(pose_strength),
                prepared_reference_pack=prepared_reference_pack,
            )
            summary = {
                "mode": mode,
                "long_video_mode": long_video_mode,
                "mask_replacement_mode": bool(mask_replacement_mode),
                "frames": _shape(frames),
                "width": width,
                "height": height,
                "source_frames": int(total_frames),
                "generated_frames": int(frames.shape[0]),
                "dropped_tail_frames": int(total_frames - frames.shape[0]),
                "color_correction_enabled": False,
                "cfg": float(cfg),
                "seed_start": int(seed),
                "reference_input": "reference_pack" if reference_pack is not None else "image",
                "core": "SCAIL-2 context sampling -> SamplerCustom -> VAEDecode",
                "context_details": context_summary,
            }
            return (frames.contiguous().clamp(0, 1), json.dumps(summary, indent=2))

        chunk_frames = _wan_frame_count_cover(max(17, int(chunk_frames)))
        first_length = chunk_frames
        requested_overlap = max(0, int(overlap_frames))
        previous_frame_count = 0 if requested_overlap == 0 else max(1, min(_wan_frame_count_floor(requested_overlap), 33, max(1, chunk_frames - 4)))
        if chunk_frames <= previous_frame_count:
            raise ValueError("chunk_frames must be larger than overlap_frames.")
        color_correction_enabled = bool(color_correction and previous_frame_count > 0)

        stitched = []
        chunk_summaries = []
        previous_frames = None
        video_frame_offset = 0
        produced = 0
        chunk_index = 0
        max_chunks = max(1, (total_frames // max(1, chunk_frames - previous_frame_count)) + 4)

        while produced < total_frames and chunk_index < max_chunks:
            anchor_start = max(0, video_frame_offset - (previous_frame_count if previous_frames is not None else 0))
            remaining_from_anchor = max(1, total_frames - anchor_start)
            target_length = first_length if chunk_index == 0 else chunk_frames
            length = min(target_length, _wan_frame_count_floor(remaining_from_anchor))
            if previous_frames is not None and previous_frame_count > 0 and length <= previous_frame_count:
                break

            if reference_pack is not None:
                chunk_replacement_mode = bool(mask_replacement_mode)
                prepared_reference_pack = reference_pack_cache.get(chunk_replacement_mode)
                if prepared_reference_pack is None:
                    prepared_reference_pack = _prepare_scail_reference_pack(
                        vae=vae,
                        reference_pack=reference_pack,
                        width=width,
                        height=height,
                        replacement_mode=chunk_replacement_mode,
                    )
                    reference_pack_cache[chunk_replacement_mode] = prepared_reference_pack
                decoded, next_offset, chunk_summary = _run_multi_ref_scail_chunk(
                    model=model,
                    positive=positive,
                    negative=negative,
                    vae=vae,
                    sampler=sampler,
                    sigmas=sigmas,
                    reference_pack=reference_pack,
                    pose_video=pose_video,
                    clip_vision_output=clip_vision_output,
                    pose_video_mask=pose_video_mask,
                    previous_frames=previous_frames,
                    width=width,
                    height=height,
                    length=length,
                    video_frame_offset=video_frame_offset,
                    previous_frame_count=previous_frame_count,
                    replacement_mode=chunk_replacement_mode,
                    seed=int(seed) + chunk_index,
                    cfg=float(cfg),
                    pose_strength=float(pose_strength),
                    prepared_reference_pack=prepared_reference_pack,
                )
            else:
                decoded, next_offset, chunk_summary = _run_native_scail_chunk(
                    model=model,
                    positive=positive,
                    negative=negative,
                    vae=vae,
                    sampler=sampler,
                    sigmas=sigmas,
                    reference_image=reference_image,
                    pose_video=pose_video,
                    clip_vision_output=clip_vision_output,
                    pose_video_mask=pose_video_mask,
                    reference_image_mask=reference_image_mask,
                    previous_frames=previous_frames,
                    width=width,
                    height=height,
                    length=length,
                    video_frame_offset=video_frame_offset,
                    previous_frame_count=previous_frame_count,
                    replacement_mode=mask_replacement_mode,
                    seed=int(seed) + chunk_index,
                    cfg=float(cfg),
                    pose_strength=float(pose_strength),
                )

            discard_head = 0 if chunk_index == 0 else min(previous_frame_count, int(decoded.shape[0]))
            current_overlap = decoded[:discard_head].contiguous() if discard_head > 0 else None
            reference_overlap = previous_frames[-discard_head:].contiguous() if previous_frames is not None and discard_head > 0 else None
            kept = decoded[discard_head:].contiguous()
            remaining_output = total_frames - produced
            if kept.shape[0] > remaining_output:
                kept = kept[:remaining_output].contiguous()
            if kept.shape[0] <= 0:
                raise RuntimeError("A SCAIL-2 chunk produced no keepable frames.")

            color_summary = {"applied": False}
            if color_correction_enabled and discard_head > 0:
                kept, color_summary = _match_chunk_color_to_overlap(kept, current_overlap, reference_overlap)

            kept_cpu = kept.detach().cpu().contiguous()
            stitched.append(kept_cpu)
            produced += int(kept.shape[0])
            chunk_summary.update(
                {
                    "chunk_index": chunk_index,
                    "replacement_mode": bool(mask_replacement_mode),
                    "discard_head": int(discard_head),
                    "kept_frames": int(kept.shape[0]),
                    "produced_total": int(produced),
                }
            )
            if color_correction_enabled:
                chunk_summary["color_correction"] = color_summary
            chunk_summaries.append(chunk_summary)

            if previous_frame_count > 0:
                tail_parts = [kept_cpu] if previous_frames is None else [previous_frames, kept_cpu]
                previous_frames = torch.cat(tail_parts, dim=0)[-previous_frame_count:].contiguous()
            else:
                previous_frames = None
            video_frame_offset = int(next_offset)
            chunk_index += 1
            del decoded, kept, kept_cpu, current_overlap, reference_overlap
            _empty_cache(force=True)

        frames = torch.cat(stitched, dim=0).contiguous().clamp(0, 1)
        summary = {
            "mode": mode,
            "long_video_mode": long_video_mode,
            "mask_replacement_mode": bool(mask_replacement_mode),
            "chunk_replacement_mode": bool(mask_replacement_mode),
            "frames": _shape(frames),
            "width": width,
            "height": height,
            "source_frames": int(total_frames),
            "generated_frames": int(frames.shape[0]),
            "dropped_tail_frames": int(total_frames - frames.shape[0]),
            "chunks": len(chunk_summaries),
            "first_chunk_frames": int(first_length),
            "chunk_frames": int(chunk_frames),
            "overlap_frames": int(previous_frame_count),
            "color_correction_enabled": bool(color_correction_enabled),
            "cfg": float(cfg),
            "seed_start": int(seed),
            "reference_input": "reference_pack" if reference_pack is not None else "image",
            "core": "SCAIL-2 multi-reference conditioning -> SamplerCustom -> VAEDecode"
            if reference_pack is not None
            else "ComfyUI native WanSCAILToVideo -> SamplerCustom -> VAEDecode",
            "chunk_details": chunk_summaries,
        }
        return (frames, json.dumps(summary, indent=2))


class SFSCAIL2FitVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE", {"tooltip": "待统一的输入视频帧序列 [T,H,W,C]"}),
                "resolution": (list(RESOLUTION_PRESETS), {"default": "512p", "tooltip": "分辨率预设：512p/704p 把短边缩放到对应值、长边按原比例并对齐 32 的倍数；custom 手动指定 custom_width/custom_height"}),
                "custom_width": ("INT", {"default": 832, "min": 32, "max": 4096, "step": 32, "tooltip": "custom 分辨率下的输出宽度（对齐 32 的倍数）"}),
                "custom_height": ("INT", {"default": 480, "min": 32, "max": 4096, "step": 32, "tooltip": "custom 分辨率下的输出高度（对齐 32 的倍数）"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("video", "width", "height", "summary")
    FUNCTION = "fit"
    CATEGORY = _CATEGORY
    DESCRIPTION = "统一 SCAIL-2 输入视频尺寸：512p/704p 短边缩放长边按比例对齐 32，custom 手动指定（选 512p/704p 时前端隐藏宽高）"

    def fit(self, video: torch.Tensor, resolution: str, custom_width: int = 832, custom_height: int = 480):
        target_width, target_height = _target_size_for_video(video, resolution, custom_width, custom_height)
        fitted = _resize_to_target(video, target_width, target_height)
        summary = {
            "resolution": resolution,
            "custom_width": int(custom_width),
            "custom_height": int(custom_height),
            "source": _shape(video),
            "target": _shape(fitted),
            "width": target_width,
            "height": target_height,
            "fit": "preserve_aspect_resize_to_nearest_32",
        }
        return (fitted, target_width, target_height, json.dumps(summary, indent=2))


NODE_CLASS_MAPPINGS = {
    "SFSCAIL2FitVideo": SFSCAIL2FitVideo,
    "SFSCAIL2ReferencePack": SFSCAIL2ReferencePack,
    "SFSCAIL2ReferenceSAMBuilder": SFSCAIL2ReferenceSAMBuilder,
    "SFSCAIL2SimpleVideo": SFSCAIL2SimpleVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SFSCAIL2FitVideo": "SF SCAIL-2 Fit Video",
    "SFSCAIL2ReferencePack": "SF SCAIL-2 Reference Pack",
    "SFSCAIL2ReferenceSAMBuilder": "SF SCAIL-2 Reference SAM Builder",
    "SFSCAIL2SimpleVideo": "SF SCAIL-2 Simple Video",
}
