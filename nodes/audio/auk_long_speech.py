"""SFAuKLongSpeech：AuK 长文本语音合成 / 长音频处理（超过 30 秒的分块 + 拼接）。

30 秒是单次模型序列预算（参考/输入 + 生成长度），拼接后的总时长不受限。三种模式：

1. 参考音色 TTS：长文本分句 → 逐段生成（首段 input_audio 参考、后续滚动参考续接）→ 拼接；
2. 声音描述 TTS：同上，首段用声音描述模板；
3. 长音频处理（编辑/增强）：整段长音频按静音点切块，每块以自身为源（编辑语义）套用 instruction
   逐块处理 → 边缘淡化拼接。等长类任务（增强/修复/音量/音高/情绪/音色/口音/耳语/非语言声删除/
   音乐人声分离）与变速类共用；源块 + 目标块共享 30s 预算，单块上限按 `1+目标倍率` 自动收窄。

TTS 模式不走 Prompt Enhancer：节点直接套官方指令模板，时长用 pe.py 的本地 utf8 权重估计
（`estimate_speech_seconds` / `estimate_speech_units` + `speech_rate` 语速直控）。
处理模式的 instruction 可用前端预设下拉填入官方模板（sf_auk_presets_lib.js）。
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
MODE_PROCESS = "长音频处理（编辑/增强）"
MODE_OPTIONS = [MODE_REFERENCE, MODE_DESCRIPTION, MODE_PROCESS]

CONTINUITY_ROLLING = "滚动参考"
CONTINUITY_FIXED = "同一参考"
CONTINUITY_OPTIONS = [CONTINUITY_ROLLING, CONTINUITY_FIXED]

DURATION_EQUAL = "等长"
DURATION_SPEED = "变速"
DURATION_OPTIONS = [DURATION_EQUAL, DURATION_SPEED]

REFERENCE_TEMPLATE = 'Say the following with the same voice: "{text}"'
DESCRIPTION_TEMPLATE = (
    'Generate speech based on the following description: "{description}". '
    'The content to speak is: "{text}".'
)

TARGET_HEADROOM = 0.15  # TTS 段时长头寸（防估计略紧截字；语速由 speech_rate 直接控制）
PROCESS_RESERVE = 0.3  # 处理模式切块预算保留（源 + 目标 ≤ 30s）


def _chunk_target_seconds(estimated_seconds, reference, sample_rate):
    """TTS 单段生成时长：估计 + 头寸，并夹到「30s − 参考」预算内。"""
    ref_seconds = 0.0
    if reference is not None:
        ref_seconds = reference[0].shape[-1] / float(reference[1])
    allowed = MAX_SEQUENCE_SECONDS - ref_seconds - 0.2
    target = float(estimated_seconds) + TARGET_HEADROOM
    return max(0.5, min(target, allowed, MAX_SEQUENCE_SECONDS))


def _process_chunk_limit(duration_mode, speed_multiplier, max_chunk_seconds):
    """处理模式单块源音频上限：源块 + 目标块（源×倍率）≤ 30s，且不超过用户上限。"""
    speed = max(0.5, min(2.0, float(speed_multiplier)))
    factor = 1.0 if duration_mode == DURATION_EQUAL else 1.0 / speed
    budget = MAX_SEQUENCE_SECONDS / (1.0 + factor) - PROCESS_RESERVE
    return max(1.0, min(float(max_chunk_seconds), budget))


def _split_source_chunks(waveform, sample_rate, max_seconds):
    """长音频切块：目标切点前 1.2s 内找最低能量 20ms 窗口切（避开词中），返回 [(块, start, end)]。"""
    total = waveform.shape[-1]
    limit = max(1, int(float(max_seconds) * sample_rate))
    if total <= limit:
        return [(waveform, 0, total)]
    window = max(1, int(0.02 * sample_rate))
    search = max(window, int(1.2 * sample_rate))
    chunks = []
    start = 0
    while start < total:
        if total - start <= limit:
            chunks.append((waveform[..., start:total], start, total))
            break
        target = start + limit
        low = max(start + window, target - search)
        segment = waveform[..., low:target]
        usable = (segment.shape[-1] // window) * window
        cut = target
        if usable >= window:
            frames = segment[..., :usable].reshape(-1, window)
            rms = frames.square().mean(dim=1).sqrt()
            cut = low + int(rms.argmin().item()) * window
        if cut <= start:
            cut = target
        chunks.append((waveform[..., start:cut], start, cut))
        start = cut
    return chunks


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
        "AuK 长音频节点（30 秒限制是单次序列预算，总时长不限）："
        "参考音色 TTS / 声音描述 TTS = 长文本自动分句逐段合成（首段参考 + 后续滚动参考续接语气）；"
        "长音频处理（编辑/增强）= 整段长音频按静音点切块，每块以自身为源套用 instruction 处理"
        "（等长类增强/修复/音量/音高/情绪/音色/口音/耳语/非语言声删除/音乐人声分离，以及变速）。"
        "输出音频 + 分段报告；长文本/长音频建议用 AuK-Flash（4 步）"
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
                    "tooltip": "TTS 模式要合成的长文本（处理模式忽略）；按句号/问号/感叹号/分号/换行分句",
                }),
                "mode": (MODE_OPTIONS, {
                    "default": MODE_REFERENCE,
                    "tooltip": "参考音色 TTS = 用 input_audio 克隆音色合成 text；"
                               "声音描述 TTS = 用 voice_description 描述音色合成 text；"
                               "长音频处理 = 对整段 input_audio 逐块套用 instruction（编辑/增强/变速）",
                }),
                "max_chunk_seconds": ("FLOAT", {
                    "default": 24.0, "min": 2.0, "max": 30.0, "step": 0.5,
                    "tooltip": "单段上限：TTS 为生成时长上限；处理模式为单块源音频上限"
                               "（实际按 30s 预算自动收窄：处理模式源+目标共享预算，等长 ≈14.5s、变速按 1+1/倍率）",
                }),
                "speech_rate": ("FLOAT", {
                    "default": 4.15, "min": 3.0, "max": 6.0, "step": 0.05,
                    "tooltip": "TTS 语速（中文字/秒；英文/数字/标点按字节权重折算）：数值越大越紧。"
                               "自然语速约 4.2-4.8；实际请求时长 = 加权字数 ÷ 语速 + 0.15s（处理模式忽略）",
                }),
                "ref_tail_seconds": ("FLOAT", {
                    "default": 4.0, "min": 1.0, "max": 10.0, "step": 0.5,
                    "tooltip": "TTS 滚动参考：取上一段尾部多少秒作下一段参考（处理模式忽略）",
                }),
                "reference_seconds": ("FLOAT", {
                    "default": 10.0, "min": 0.0, "max": 20.0, "step": 0.5,
                    "tooltip": "TTS 首段参考裁剪上限（秒），0 = 用全长；处理模式忽略（每块以自身为源）",
                }),
                "pause_seconds": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "TTS 段间静音；处理模式忽略（不插静音，避免改变时间轴）",
                }),
                "continuity": (CONTINUITY_OPTIONS, {
                    "default": CONTINUITY_ROLLING,
                    "tooltip": "TTS 参考续接档：滚动参考 = 上一段尾音续接（推荐）；同一参考 = 每段都用首段参考",
                }),
                "seed": ("INT", {
                    "default": 42, "min": 0, "max": 0x7FFFFFFFFFFFFFFF,
                    "control_after_generate": "fixed",
                    "tooltip": "逐段/逐块用 seed + 序号（同 seed 可复现）",
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
                    "tooltip": "TTS 模式裁掉每段尾部静音（防段间静音累积）；处理模式忽略（源静音属于内容）",
                }),
            },
            "optional": {
                "input_audio": ("AUDIO", {
                    "tooltip": "TTS 参考音色模式必接（音色克隆参考，按 reference_seconds 裁剪）；"
                               "长音频处理模式必接（待处理的长音频，按静音点切块）；声音描述 TTS 忽略",
                }),
                "voice_description": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "声音描述 TTS 必填：音色/风格描述；其余模式忽略",
                }),
                "instruction": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "长音频处理模式必填：处理指令（可用上方预设下拉填入官方模板，"
                               "如去噪/去混响/音量/音高/语速/情绪/音色/去口音/耳语/分离等）；其余模式忽略",
                }),
                "duration_mode": (DURATION_OPTIONS, {
                    "default": DURATION_EQUAL,
                    "tooltip": "长音频处理模式的目标时长：等长 = 与源块同长（增强/编辑类）；"
                               "变速 = 源块 ÷ speed_multiplier",
                }),
                "speed_multiplier": ("FLOAT", {
                    "default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05,
                    "tooltip": "长音频处理「变速」档的目标倍率：2.0 = 快一倍（目标减半）、0.5 = 慢一倍",
                }),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "report")
    OUTPUT_TOOLTIPS = (
        "拼接后的长音频（AUDIO）",
        "分段报告 JSON：mode/chunks/total_seconds/segments（每段文本或时间区间、估计与应用时长、参考来源）",
    )
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    def execute(self, engine, text, mode, max_chunk_seconds=24.0, speech_rate=4.15, ref_tail_seconds=4.0,
                reference_seconds=10.0, pause_seconds=0.1, continuity=CONTINUITY_ROLLING,
                seed=42, nfe_steps=32, cfg_strength=2.0, sway_sampling_coef=-1.0,
                trim_trailing_silence=True, input_audio=None, voice_description="",
                instruction="", duration_mode=DURATION_EQUAL, speed_multiplier=1.0):
        if mode not in MODE_OPTIONS:
            raise ValueError(f"未知模式：{mode!r}")

        if engine.inference.is_flash:
            # Flash 配方锁定（与 SFAuKGenerateEdit 一致；引擎本就忽略这三个参数）
            if int(nfe_steps) != 4 or float(cfg_strength) != 0.0 or float(sway_sampling_coef) != -1.0:
                print(
                    f"[SFAuKLongSpeech] AuK-Flash 配方锁定：NFE 4 / CFG 0 / sway -1 "
                    f"（忽略 widget 值 NFE {int(nfe_steps)} / CFG {float(cfg_strength)} / sway {float(sway_sampling_coef)}）"
                )
            nfe_steps, cfg_strength, sway_sampling_coef = 4, 0.0, -1.0

        if mode == MODE_PROCESS:
            return self._execute_process(
                engine, input_audio, instruction, duration_mode, speed_multiplier,
                max_chunk_seconds, seed, nfe_steps, cfg_strength, sway_sampling_coef,
            )

        # ---- TTS 路径（参考音色 / 声音描述）----
        text = str(text or "").strip()
        if not text:
            raise ValueError("长文本为空，请输入要合成的文本")
        if continuity not in CONTINUITY_OPTIONS:
            raise ValueError(f"未知连续性档：{continuity!r}")
        description = str(voice_description or "").strip()
        if mode == MODE_DESCRIPTION and not description:
            raise ValueError("声音描述 TTS 需要填写 voice_description")
        if mode == MODE_REFERENCE and input_audio is None:
            raise ValueError("参考音色 TTS 需要连接 input_audio")

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
        make_report = _make_chunk_report(mm, pbar, grand_total, per_chunk_span, expected_steps)

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
                    chunk_instruction = DESCRIPTION_TEMPLATE.format(description=description, text=chunk["text"])
                else:
                    chunk_instruction = REFERENCE_TEMPLATE.format(text=chunk["text"])
                target_seconds = _chunk_target_seconds(chunk["seconds"], reference, sample_rate)
                validate_sequence_duration(engine, model_audio, target_seconds)
                content = [{"type": "text", "text": chunk_instruction}]
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
                waveform = _check_chunk_audio(waveform, index, len(chunks))
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

    def _execute_process(self, engine, input_audio, instruction, duration_mode, speed_multiplier,
                         max_chunk_seconds, seed, nfe_steps, cfg_strength, sway_sampling_coef):
        """长音频处理：静音点切块 → 每块以自身为源套用 instruction → 边缘淡化拼接。"""
        instruction = str(instruction or "").strip()
        if not instruction:
            raise ValueError("长音频处理模式需要填写 instruction（可用预设下拉填入官方模板）")
        if input_audio is None:
            raise ValueError("长音频处理模式需要连接 input_audio（待处理的长音频）")
        if duration_mode not in DURATION_OPTIONS:
            raise ValueError(f"未知时长模式：{duration_mode!r}")
        speed = max(0.5, min(2.0, float(speed_multiplier)))
        factor = 1.0 if duration_mode == DURATION_EQUAL else 1.0 / speed

        source = normalize_audio(input_audio)
        if source is None:
            raise ValueError("audio 输入为空")
        waveform, sample_rate = source
        chunk_limit = _process_chunk_limit(duration_mode, speed, max_chunk_seconds)
        chunks = _split_source_chunks(waveform, sample_rate, chunk_limit)

        import comfy.model_management as mm
        import comfy.utils

        output_rate = engine.inference.target_sample_rate
        expected_steps = 4 if engine.inference.is_flash else int(nfe_steps)
        per_chunk_span = expected_steps + 3  # vae_encode + encode + sample + decode
        grand_total = per_chunk_span * len(chunks)
        pbar = comfy.utils.ProgressBar(grand_total)
        make_report = _make_chunk_report(mm, pbar, grand_total, per_chunk_span, expected_steps)

        generated_chunks = []
        segment_reports = []
        with engine.lock:
            if engine.inference.memory_mode != "standard":
                mm.unload_all_models()
                mm.soft_empty_cache()
            for index, (chunk_wave, start, end) in enumerate(chunks):
                reference = (chunk_wave, sample_rate)
                model_audio, qwen_audio = prepare_model_audio(reference, output_rate)
                source_seconds = (end - start) / float(sample_rate)
                target_seconds = max(0.5, min(source_seconds * factor, MAX_SEQUENCE_SECONDS))
                validate_sequence_duration(engine, model_audio, target_seconds)
                content = [{"type": "text", "text": instruction}]
                if qwen_audio is not None:
                    content.append({"type": "audio", "audio": qwen_audio})
                messages = [{"role": "user", "content": content}]
                torch.manual_seed(int(seed) + index)
                try:
                    chunk_audio, chunk_rate = engine.inference.generate(
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
                        f"长音频第 {index + 1}/{len(chunks)} 块处理失败：{type(error).__name__}: {error}"
                    ) from error
                chunk_audio = _check_chunk_audio(chunk_audio, index, len(chunks))
                generated_chunks.append(chunk_audio)
                segment_reports.append({
                    "index": index + 1,
                    "start_seconds": round(start / float(sample_rate), 2),
                    "end_seconds": round(end / float(sample_rate), 2),
                    "source_seconds": round(source_seconds, 2),
                    "applied_seconds": round(target_seconds, 2),
                    "audio_seconds": round(chunk_audio.shape[-1] / float(chunk_rate), 2),
                })

        audio = _concat_chunks(generated_chunks, output_rate, 0.0)
        report = json.dumps({
            "mode": MODE_PROCESS,
            "instruction": instruction,
            "duration_mode": duration_mode,
            "speed_multiplier": speed,
            "chunks": len(chunks),
            "total_seconds": round(audio.shape[-1] / float(output_rate), 2),
            "segments": segment_reports,
        }, ensure_ascii=False)
        return ({"waveform": audio.unsqueeze(0), "sample_rate": int(output_rate)}, report)


def _make_chunk_report(mm, pbar, grand_total, per_chunk_span, expected_steps):
    """按段索引返回进度回调：段内 vae_encode/encode/sample/decode 映射到总进度，并做中断检查。"""
    phase_offsets = {"vae_encode": 0, "encode": 1, "sample": 2, "decode": 2 + expected_steps}
    phase_spans = {"vae_encode": 1, "encode": 1, "sample": expected_steps, "decode": 1}

    def make(chunk_index):
        base = chunk_index * per_chunk_span

        def report(phase, done, total):
            mm.throw_exception_if_processing_interrupted()
            span = phase_spans.get(phase)
            if not span:
                return
            value = base + phase_offsets[phase] + span * min(max(done, 0), total) / max(total, 1)
            pbar.update_absolute(min(round(value), grand_total), grand_total)

        return report

    return make


def _check_chunk_audio(waveform, index, total):
    """段/块返回音频校验与归一形状（[1, T]，有限值）。"""
    waveform = waveform.detach().to(device="cpu", dtype=torch.float32)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2 or waveform.shape[-1] == 0:
        raise RuntimeError(f"AuK 返回了非法音频形状：{tuple(waveform.shape)}（第 {index + 1}/{total} 段）")
    if not torch.isfinite(waveform).all():
        raise RuntimeError(f"第 {index + 1}/{total} 段音频包含 NaN/Inf")
    return waveform
