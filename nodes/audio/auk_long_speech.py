"""SFAuKLongSpeech：AuK 长文本语音合成（分句 → 滚动参考分段生成 → 拼接）。

30 秒是单次模型序列预算（参考/输入 + 生成长度），拼接后的总时长不受限。本节点：
  1. 长文本按标点分句、按本地时长估计打包为若干段（单段估计 ≤ 预算）；
  2. 第 1 段用 input_audio 参考（可裁剪），后续段用上一段尾部音频作滚动参考（音色/语气续接），
     也可切「同一参考」档；
  3. 逐段调用 AukInfer.generate（复用进度条与中断检查），段间短静音 + 5ms 淡入淡出后拼接；
  4. 输出拼接音频与分段报告（段数/每段文本与时长/参考来源）。

不走 Prompt Enhancer：节点直接套官方指令模板，时长用 pe.py 的本地 utf8 权重估计
（`estimate_speech_seconds`，与 PE 的 F5 估计同源）。
"""

import json

import torch

from .auk_generate import (
    MAX_SEQUENCE_SECONDS,
    normalize_audio,
    prepare_model_audio,
    validate_sequence_duration,
)
from .auk_loader import AUK_ENGINE

# 顶层包导入时 `...` 正常；测试以 `nodes.audio.auk_long_speech` 顶层导入时 `...` 越界，
# 回退绝对导入（image_interrogator_api.py 同款可移植性兜底）。
try:
    from ...sf_utils.text_chunk import pack_chunks
except Exception:  # pragma: no cover - 测试/移植性兜底
    from sf_utils.text_chunk import pack_chunks  # type: ignore

_CATEGORY = "sfnodes/audio"

MODE_REFERENCE = "参考音色 TTS"
MODE_DESCRIPTION = "声音描述 TTS"
MODE_OPTIONS = [MODE_REFERENCE, MODE_DESCRIPTION]

CONTINUITY_ROLLING = "滚动参考"
CONTINUITY_FIXED = "同一参考"
CONTINUITY_OPTIONS = [CONTINUITY_ROLLING, CONTINUITY_FIXED]

REFERENCE_TEMPLATE = 'Say the following with the same voice: "{text}"'
DESCRIPTION_TEMPLATE = (
    'Generate speech based on the following description: "{description}". '
    'The content to speak is: "{text}".'
)

TARGET_HEADROOM = 0.15  # 段时长头寸（防估计略紧截字；语速由 speech_rate 直接控制）


def _chunk_target_seconds(estimated_seconds, reference, sample_rate):
    """单段生成时长：估计 × 余量，并夹到「30s − 参考」预算内。"""
    ref_seconds = 0.0
    if reference is not None:
        ref_seconds = reference[0].shape[-1] / float(reference[1])
    allowed = MAX_SEQUENCE_SECONDS - ref_seconds - 0.2
    target = float(estimated_seconds) + TARGET_HEADROOM
    return max(0.5, min(target, allowed, MAX_SEQUENCE_SECONDS))


def _rolling_reference(waveform, sample_rate, seconds):
    """取已完成音频尾部 seconds 秒作下一段参考（[1, T] 张量）。"""
    frames = max(1, int(float(seconds) * sample_rate))
    return waveform[..., -frames:].contiguous()


def _fade_edges(waveform, sample_rate, fade_ms=5.0, fade_in=True, fade_out=True):
    """段首/段尾线性淡入淡出（防拼接爆音；只处理需要拼接的边）。"""
    count = max(1, int(float(fade_ms) / 1000.0 * sample_rate))
    count = min(count, waveform.shape[-1] // 2)
    if count <= 0:
        return waveform
    result = waveform.clone()
    if fade_in:
        ramp = torch.linspace(0.0, 1.0, count + 1)[:count]
        result[..., :count] = result[..., :count] * ramp
    if fade_out:
        ramp = torch.linspace(1.0, 0.0, count + 1)[1:]
        result[..., -count:] = result[..., -count:] * ramp
    return result


def _trim_trailing_silence(waveform, sample_rate, threshold=0.004, keep_seconds=0.05):
    """按 20ms 窗口能量裁掉段尾静音（防段间静音累积）；全静音或过短则原样返回。"""
    total = waveform.shape[-1]
    window = max(1, int(0.02 * sample_rate))
    usable = (total // window) * window
    if usable < window * 2:
        return waveform
    frames = waveform[..., :usable].reshape(-1, window)
    rms = frames.square().mean(dim=1).sqrt()
    active = (rms > threshold).nonzero()
    if active.numel() == 0:
        return waveform
    end = min(total, int(active[-1].item() + 1) * window + int(keep_seconds * sample_rate))
    if end < int(0.2 * sample_rate):
        return waveform
    return waveform[..., :end]


def _concat_chunks(chunks, sample_rate, pause_seconds):
    """段间静音 + 边缘淡化后拼接为 [1, T]。"""
    pause = max(0, int(round(float(pause_seconds) * sample_rate)))
    silence = torch.zeros(1, pause) if pause else None
    parts = []
    for index, chunk in enumerate(chunks):
        part = chunk
        if index > 0 or index < len(chunks) - 1:
            part = _fade_edges(part, sample_rate, fade_in=index > 0, fade_out=index < len(chunks) - 1)
        if index > 0 and silence is not None:
            parts.append(silence)
        parts.append(part)
    return torch.cat(parts, dim=-1)


class SFAuKLongSpeech:
    DESCRIPTION = (
        "AuK 长文本语音合成：长文本自动分句、逐段生成、拼接为一条音频（30 秒限制是单段预算，"
        "总时长不限）。参考音色 TTS 接 input_audio 克隆音色；后续段自动以上一段尾部音频作滚动"
        "参考续接语气（可切同一参考档）。不走提示词增强：直接套官方指令模板 + 本地时长估计。"
        "长文本建议用 AuK-Flash（4 步）；输出音频 + 分段报告"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "engine": (AUK_ENGINE, {
                    "tooltip": "SF AuK Models Loader 的输出",
                }),
                "text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "要合成的长文本；按句号/问号/感叹号/分号/换行分句后逐段生成",
                }),
                "mode": (MODE_OPTIONS, {
                    "default": MODE_REFERENCE,
                    "tooltip": "参考音色 TTS = 用 input_audio 克隆音色（必接音频）；"
                               "声音描述 TTS = 用 voice_description 描述音色（不接音频）",
                }),
                "max_chunk_seconds": ("FLOAT", {
                    "default": 24.0, "min": 2.0, "max": 30.0, "step": 0.5,
                    "tooltip": "单段生成时长上限（受 30s 预算与参考长度约束，实际按预算自动收窄）",
                }),
                "speech_rate": ("FLOAT", {
                    "default": 4.15, "min": 3.0, "max": 6.0, "step": 0.05,
                    "tooltip": "语速（中文字/秒；英文/数字/标点按字节权重折算）：数值越大越紧、越小越慢。"
                               "自然语速约 4.2-4.8；过大会语速过快或尾字被截断。"
                               "实际请求时长 = 加权字数 ÷ 语速 + 0.15s",
                }),
                "ref_tail_seconds": ("FLOAT", {
                    "default": 4.0, "min": 1.0, "max": 10.0, "step": 0.5,
                    "tooltip": "滚动参考：取上一段尾部多少秒作下一段参考（音色/语气续接；3-6s 常用）",
                }),
                "reference_seconds": ("FLOAT", {
                    "default": 10.0, "min": 0.0, "max": 20.0, "step": 0.5,
                    "tooltip": "首段参考裁剪上限（秒），0 = 用全长；参考越长，单段可生成长度越小",
                }),
                "pause_seconds": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "段间静音时长",
                }),
                "continuity": (CONTINUITY_OPTIONS, {
                    "default": CONTINUITY_ROLLING,
                    "tooltip": "滚动参考 = 用上一段尾音续接（推荐，语气最自然）；同一参考 = 每段都用首段参考",
                }),
                "seed": ("INT", {
                    "default": 42, "min": 0, "max": 0x7FFFFFFFFFFFFFFF,
                    "control_after_generate": "fixed",
                    "tooltip": "逐段用 seed + 段序号（同 seed 可复现）",
                }),
                "nfe_steps": ("INT", {
                    "default": 32, "min": 4, "max": 64, "step": 1, "advanced": True,
                    "tooltip": "AuK 采样步数；Flash 自动锁定为 4（忽略本值）",
                }),
                "cfg_strength": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 5.0, "step": 0.1, "advanced": True,
                    "tooltip": "CFG 强度；Flash 自动锁定为 0（忽略本值）",
                }),
                "sway_sampling_coef": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 1.0, "step": 0.1, "advanced": True,
                    "tooltip": "sway 采样系数；Flash 自动锁定为 -1（忽略本值）",
                }),
                "trim_trailing_silence": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "裁掉每段尾部静音（防段间静音累积；全静音/过短段不裁）",
                }),
            },
            "optional": {
                "input_audio": ("AUDIO", {
                    "tooltip": "参考音色 TTS 必接：音色克隆参考（按 reference_seconds 裁剪）；"
                               "声音描述 TTS 模式忽略此输入",
                }),
                "voice_description": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "声音描述 TTS 必填：音色/风格描述（官方模板 Generate speech based on ...）；"
                               "参考音色 TTS 模式忽略",
                }),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "report")
    OUTPUT_TOOLTIPS = (
        "拼接后的长语音（AUDIO）",
        "分段报告 JSON：mode/continuity/chunks/total_seconds/segments（每段文本/估计与应用时长/参考来源）",
    )
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    def execute(self, engine, text, mode, max_chunk_seconds=24.0, speech_rate=4.15, ref_tail_seconds=4.0,
                reference_seconds=10.0, pause_seconds=0.1, continuity=CONTINUITY_ROLLING,
                seed=42, nfe_steps=32, cfg_strength=2.0, sway_sampling_coef=-1.0,
                trim_trailing_silence=True, input_audio=None, voice_description=""):
        text = str(text or "").strip()
        if not text:
            raise ValueError("长文本为空，请输入要合成的文本")
        if mode not in MODE_OPTIONS:
            raise ValueError(f"未知模式：{mode!r}")
        if continuity not in CONTINUITY_OPTIONS:
            raise ValueError(f"未知连续性档：{continuity!r}")
        description = str(voice_description or "").strip()
        if mode == MODE_DESCRIPTION and not description:
            raise ValueError("声音描述 TTS 需要填写 voice_description")
        if mode == MODE_REFERENCE and input_audio is None:
            raise ValueError("参考音色 TTS 需要连接 input_audio")

        if engine.inference.is_flash:
            # Flash 配方锁定（与 SFAuKGenerateEdit 一致；引擎本就忽略这三个参数）
            if int(nfe_steps) != 4 or float(cfg_strength) != 0.0 or float(sway_sampling_coef) != -1.0:
                print(
                    f"[SFAuKLongSpeech] AuK-Flash 配方锁定：NFE 4 / CFG 0 / sway -1 "
                    f"（忽略 widget 值 NFE {int(nfe_steps)} / CFG {float(cfg_strength)} / sway {float(sway_sampling_coef)}）"
                )
            nfe_steps, cfg_strength, sway_sampling_coef = 4, 0.0, -1.0

        sample_rate = engine.inference.target_sample_rate
        ref_audio = normalize_audio(input_audio) if mode == MODE_REFERENCE else None
        if ref_audio is not None and float(reference_seconds) > 0:
            limit = max(1, int(float(reference_seconds) * ref_audio[1]))
            if ref_audio[0].shape[-1] > limit:
                ref_audio = (ref_audio[0][..., :limit].contiguous(), ref_audio[1])

        from .auk.infer.pe import DEFAULT_SPEECH_RATE, estimate_speech_units

        rate = max(1.0, min(10.0, float(speech_rate)))

        def estimate(value):
            return estimate_speech_units(value, None) / rate

        first_ref_seconds = ref_audio[0].shape[-1] / float(ref_audio[1]) if ref_audio is not None else 0.0
        next_ref_seconds = float(ref_tail_seconds) if continuity == CONTINUITY_ROLLING else first_ref_seconds
        budget_limit = MAX_SEQUENCE_SECONDS - max(first_ref_seconds, next_ref_seconds) - 0.3
        if budget_limit < 2.0:
            raise ValueError(
                f"参考音频过长（{first_ref_seconds:.1f}s），30 秒预算不足以生成；"
                f"请缩短参考或减小 reference_seconds"
            )
        chunk_limit = max(0.5, min(float(max_chunk_seconds), budget_limit))
        chunks = pack_chunks(text, estimate, chunk_limit)
        if not chunks:
            raise ValueError("长文本分句后为空，请输入要合成的文本")

        import comfy.model_management as mm
        import comfy.utils

        expected_steps = 4 if engine.inference.is_flash else int(nfe_steps)
        per_chunk_span = expected_steps + 3  # vae_encode + encode + sample + decode
        grand_total = per_chunk_span * len(chunks)
        pbar = comfy.utils.ProgressBar(grand_total)
        phase_offsets = {"vae_encode": 0, "encode": 1, "sample": 2, "decode": 2 + expected_steps}
        phase_spans = {"vae_encode": 1, "encode": 1, "sample": expected_steps, "decode": 1}

        def make_report(chunk_index):
            base = chunk_index * per_chunk_span

            def report(phase, done, total):
                mm.throw_exception_if_processing_interrupted()
                span = phase_spans.get(phase)
                if not span:
                    return
                value = base + phase_offsets[phase] + span * min(max(done, 0), total) / max(total, 1)
                pbar.update_absolute(min(round(value), grand_total), grand_total)

            return report

        generated_chunks = []
        segment_reports = []
        with engine.lock:
            if engine.inference.memory_mode != "standard":
                mm.unload_all_models()
                mm.soft_empty_cache()
            for index, chunk in enumerate(chunks):
                if index == 0 or continuity == CONTINUITY_FIXED:
                    reference = ref_audio
                    reference_source = "input_audio" if reference is not None else "none"
                else:
                    reference = (_rolling_reference(generated_chunks[-1], sample_rate, float(ref_tail_seconds)), sample_rate)
                    reference_source = "rolling"
                model_audio, qwen_audio = prepare_model_audio(reference, sample_rate)
                if index == 0 and mode == MODE_DESCRIPTION:
                    instruction = DESCRIPTION_TEMPLATE.format(description=description, text=chunk["text"])
                else:
                    instruction = REFERENCE_TEMPLATE.format(text=chunk["text"])
                target_seconds = _chunk_target_seconds(chunk["seconds"], reference, sample_rate)
                validate_sequence_duration(engine, model_audio, target_seconds)
                content = [{"type": "text", "text": instruction}]
                if qwen_audio is not None:
                    content.append({"type": "audio", "audio": qwen_audio})
                messages = [{"role": "user", "content": content}]
                torch.manual_seed(int(seed) + index)
                try:
                    waveform, chunk_rate = engine.inference.generate(
                        messages,
                        audio=model_audio,
                        gen_seconds=target_seconds,
                        nfe=int(nfe_steps),
                        cfg_strength=float(cfg_strength),
                        sway_sampling_coef=float(sway_sampling_coef),
                        seed=int(seed) + index,
                        progress_cb=make_report(index),
                    )
                except Exception as error:
                    raise ValueError(
                        f"长文本第 {index + 1}/{len(chunks)} 段生成失败：{type(error).__name__}: {error}"
                    ) from error
                waveform = waveform.detach().to(device="cpu", dtype=torch.float32)
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)
                if waveform.ndim != 2 or waveform.shape[-1] == 0:
                    raise RuntimeError(f"AuK 返回了非法音频形状：{tuple(waveform.shape)}")
                if not torch.isfinite(waveform).all():
                    raise RuntimeError(f"第 {index + 1} 段音频包含 NaN/Inf")
                if trim_trailing_silence:
                    waveform = _trim_trailing_silence(waveform, chunk_rate)
                generated_chunks.append(waveform)
                ref_seconds = reference[0].shape[-1] / float(reference[1]) if reference is not None else 0.0
                segment_reports.append({
                    "index": index + 1,
                    "text": chunk["text"],
                    "estimated_seconds": round(chunk["seconds"], 2),
                    "raw_estimated_seconds": round(chunk["seconds"] * rate / DEFAULT_SPEECH_RATE, 2),
                    "applied_seconds": round(target_seconds, 2),
                    "reference": reference_source,
                    "reference_seconds": round(ref_seconds, 2),
                    "audio_seconds": round(waveform.shape[-1] / float(chunk_rate), 2),
                })

        audio = _concat_chunks(generated_chunks, sample_rate, float(pause_seconds))
        report = json.dumps({
            "mode": mode,
            "continuity": continuity,
            "speech_rate": rate,
            "chunks": len(chunks),
            "total_seconds": round(audio.shape[-1] / float(sample_rate), 2),
            "pause_seconds": float(pause_seconds),
            "segments": segment_reports,
        }, ensure_ascii=False)
        return ({"waveform": audio.unsqueeze(0), "sample_rate": int(sample_rate)}, report)
