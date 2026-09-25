"""SFAuKGenerateEdit：AuK 语音生成/编辑（复刻 ComfyUI-AuK_Doc AuK Generate / Edit，MIT）。

上游：DocWorkBox/ComfyUI-AuK_Doc（基于 Tencent-Hunyuan/AuK，MIT，见 auk/LICENSE）。
V3→V1 适配（AUDIO 输入输出、SF_AUK_ENGINE/SF_AUK_LLM_CONFIG 槽类型、io.NodeOutput → tuple）；
torchaudio/SoundFile/PE 引擎均延迟到实际用到时导入（启动阶段只依赖 torch 与轻量模块）。

支持指令 TTS、零样本音色克隆与语音编辑；源/参考与生成目标共享 30 秒序列预算。
开启提示词增强（PE）时由 LLM 把自然语言请求转成标准模型指令并估计时长，ASR/VAD 链路
见 auk/infer/pe.py；关闭时 generation_seconds 必须大于 0。
"""

import math
import os
import tempfile

import torch

from .auk_config import AUK_LLM_CONFIG, LlamaSettings
from .auk_loader import AUK_ENGINE

_CATEGORY = "sfnodes/audio"

MAX_SEQUENCE_SECONDS = 30.0
QWEN_AUDIO_SAMPLE_RATE = 16_000


def normalize_audio(audio):
    if audio is None:
        return None
    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("input_audio must be a ComfyUI AUDIO value with waveform and sample_rate.")

    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]
    if not torch.is_tensor(waveform) or waveform.ndim != 3:
        shape = tuple(waveform.shape) if torch.is_tensor(waveform) else type(waveform).__name__
        raise ValueError(f"input_audio waveform must have shape [B, C, T], got {shape}.")
    if waveform.shape[0] != 1:
        raise ValueError(f"AuK accepts one audio sample per run, got batch size {waveform.shape[0]}.")
    if waveform.shape[1] < 1 or waveform.shape[2] < 1:
        raise ValueError(f"input_audio waveform is empty: {tuple(waveform.shape)}.")
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError(f"input_audio sample_rate must be a positive integer, got {sample_rate!r}.")

    waveform = waveform[0].detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(waveform).all():
        raise ValueError("input_audio waveform contains NaN or Inf.")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.contiguous(), sample_rate


def prepare_model_audio(audio, target_sample_rate):
    if audio is None:
        return None, None
    import torchaudio

    waveform, sample_rate = audio
    model_waveform = waveform
    if sample_rate != target_sample_rate:
        model_waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
    qwen_waveform = model_waveform
    if target_sample_rate != QWEN_AUDIO_SAMPLE_RATE:
        qwen_waveform = torchaudio.functional.resample(model_waveform, target_sample_rate, QWEN_AUDIO_SAMPLE_RATE)
    return (model_waveform.contiguous(), target_sample_rate), qwen_waveform.squeeze(0).contiguous().numpy()


def validate_sequence_duration(engine, source_audio, target_seconds):
    if not math.isfinite(target_seconds) or target_seconds <= 0:
        raise ValueError(f"Target duration must be a finite value greater than 0 seconds, got {target_seconds!r}.")
    sample_rate = engine.inference.target_sample_rate
    downsample_rate = engine.inference.downsample_rate
    source_frames = source_audio[0].shape[-1] // downsample_rate if source_audio is not None else 0
    target_frames = max(1, math.ceil(target_seconds * sample_rate / downsample_rate))
    max_frames = int(MAX_SEQUENCE_SECONDS * sample_rate / downsample_rate)
    if source_frames + target_frames > max_frames:
        source_seconds = source_frames * downsample_rate / sample_rate
        applied_target_seconds = target_frames * downsample_rate / sample_rate
        raise ValueError(
            f"AuK's source/reference plus generated target sequence is "
            f"{source_seconds + applied_target_seconds:.2f}s "
            f"({source_seconds:.2f}s + {applied_target_seconds:.2f}s after model-frame rounding), "
            f"exceeding the {MAX_SEQUENCE_SECONDS:.0f}s limit."
        )


class SFAuKGenerateEdit:
    DESCRIPTION = (
        "AuK 指令 TTS / 零样本音色克隆 / 语音编辑：输入自然语言 instruction 与生成秒数，"
        "可选 input_audio 作参考音色或待编辑音频（纯描述 TTS 不接），源/参考与生成目标合计"
        "最多 30 秒。开启提示词增强时 0 秒自动估计时长，llm_config 可接 SF AuK OpenAI Settings "
        "或 SF AuK Llama.cpp Adapter；Base 默认 32 步/CFG 2/sway -1，Flash 固定四步/CFG 0。"
        "输出音频、实际模型指令与增强信息"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "engine": (AUK_ENGINE, {
                    "tooltip": "SF AuK Models Loader 的输出",
                }),
                "instruction": ("STRING", {
                    "multiline": True,
                    "default": (
                        'Generate speech based on the following description: "a calm, warm female voice". '
                        'The content to speak is: "Hello, welcome to AuK.".'
                    ),
                    "tooltip": "AuK 自然语言请求；开启提示词增强时可由 PE 把自由描述转成标准模型指令",
                }),
                "generation_seconds": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": MAX_SEQUENCE_SECONDS, "step": 0.1,
                    "tooltip": "生成目标时长（秒）。0 表示由提示词增强估计；关闭增强时必须大于 0",
                }),
                "use_prompt_enhancer": ("BOOLEAN", {
                    "default": True,
                    "label_on": "enabled", "label_off": "disabled",
                    "tooltip": "使用与 AuK Gradio 演示一致的 PE 准备链路（分类/改写/时长估计/ASR）；"
                               "OpenAI 凭据来自 llm_config 或服务端环境变量",
                }),
                "seed": ("INT", {
                    "default": 42, "min": 0, "max": 0x7FFFFFFFFFFFFFFF,
                    "control_after_generate": "fixed",
                    "tooltip": "同时作为参考 VAE 编码与目标采样种子",
                }),
                "nfe_steps": ("INT", {
                    "default": 32, "min": 4, "max": 64, "step": 1, "advanced": True,
                    "tooltip": "AuK 采样步数；Flash 固定 4 步",
                }),
                "cfg_strength": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 5.0, "step": 0.1, "advanced": True,
                    "tooltip": "CFG 强度；Flash 固定 0",
                }),
                "sway_sampling_coef": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 1.0, "step": 0.1, "advanced": True,
                    "tooltip": "sway 采样系数；Flash 需保持默认 -1 占位",
                }),
            },
            "optional": {
                "input_audio": ("AUDIO", {
                    "tooltip": "零样本 TTS 的参考音频或语音编辑的源音频；纯描述 TTS 不接。"
                               "批大小必须为 1，多声道取均值转单声道",
                }),
                "llm_config": (AUK_LLM_CONFIG, {
                    "tooltip": "SF AuK OpenAI Settings 或 SF AuK Llama.cpp Adapter；不接则用服务端环境变量。"
                               "关闭提示词增强时忽略",
                }),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("generated_audio", "model_instruction", "prompt_enhancer_info")
    OUTPUT_TOOLTIPS = (
        "生成音频（AUDIO）",
        "实际送入 AuK 的模型指令（开启 PE 时为增强后的指令）",
        "提示词增强信息：任务/目标时长/ASR 内容；未开启时为 disabled",
    )
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    def execute(self, engine, instruction, generation_seconds, use_prompt_enhancer, seed,
                nfe_steps=32, cfg_strength=2.0, sway_sampling_coef=-1.0,
                input_audio=None, llm_config=None):
        instruction = instruction.strip()
        if not instruction:
            raise ValueError("AuK instruction is empty.")
        if not math.isfinite(float(generation_seconds)) or not 0.0 <= float(generation_seconds) <= MAX_SEQUENCE_SECONDS:
            raise ValueError(f"generation_seconds must be between 0 and {MAX_SEQUENCE_SECONDS:.0f}, got {generation_seconds!r}.")
        if engine.inference.is_flash and (int(nfe_steps) != 4 or float(cfg_strength) != 0.0 or float(sway_sampling_coef) != -1.0):
            raise ValueError(
                "AuK-Flash requires NFE steps=4, CFG strength=0, and sway=-1; these controls are fixed by its recipe."
            )

        audio = normalize_audio(input_audio)
        prepared = None
        bridge_path = None
        final_instruction = instruction
        prompt_enhancer_info = "Prompt Enhancer: disabled"
        target_seconds = float(generation_seconds)

        try:
            if use_prompt_enhancer:
                from .auk.infer.audio_io import read_audio, write_wav

                if audio is not None:
                    waveform, sample_rate = audio
                    descriptor, bridge_path = tempfile.mkstemp(prefix="auk_comfy_pe_", suffix=".wav")
                    os.close(descriptor)
                    write_wav(
                        bridge_path,
                        waveform,
                        sample_rate,
                    )

                from .auk.infer.pe import PromptEnhancer, PromptEnhancerError

                try:
                    prepared = PromptEnhancer(**(llm_config.enhancer_kwargs() if llm_config is not None else {})).prepare(
                        instruction,
                        bridge_path,
                        target_duration=target_seconds if target_seconds > 0 else None,
                    )
                except (PromptEnhancerError, ValueError, FileNotFoundError) as error:
                    raise ValueError(f"Prompt Enhancer failed: {type(error).__name__}: {error}") from error
                finally:
                    if isinstance(llm_config, LlamaSettings):
                        llm_config.cleanup()
                final_instruction = prepared.instruction
                target_seconds = target_seconds if target_seconds > 0 else prepared.gen_seconds
                if prepared.audio:
                    prepared_waveform, prepared_sample_rate = read_audio(prepared.audio)
                    audio = normalize_audio({"waveform": prepared_waveform.unsqueeze(0), "sample_rate": prepared_sample_rate})
                else:
                    audio = None

                task = prepared.task_type
                if prepared.operation_subtype:
                    task = f"{task} / {prepared.operation_subtype}"
                asr_text = prepared.asr.text if prepared.asr and prepared.asr.text else ""
                prompt_enhancer_info = f"Task: {task}\nTarget Duration: {target_seconds:.2f} s\nASR Content: {asr_text}"
            elif target_seconds <= 0:
                raise ValueError("Duration must be greater than 0 when Prompt Enhancer is disabled.")

            model_audio, qwen_audio = prepare_model_audio(audio, engine.inference.target_sample_rate)
            validate_sequence_duration(engine, model_audio, target_seconds)

            content = [{"type": "text", "text": final_instruction}]
            if qwen_audio is not None:
                content.append({"type": "audio", "audio": qwen_audio})
            messages = [{"role": "user", "content": content}]

            with engine.lock:
                if engine.inference.memory_mode != "standard":
                    import comfy.model_management as mm
                    mm.unload_all_models()
                    mm.soft_empty_cache()
                torch.manual_seed(int(seed))
                generated_waveform, sample_rate = engine.inference.generate(
                    messages,
                    audio=model_audio,
                    gen_seconds=target_seconds,
                    nfe=int(nfe_steps),
                    cfg_strength=float(cfg_strength),
                    sway_sampling_coef=float(sway_sampling_coef),
                    seed=int(seed),
                )
        finally:
            if prepared is not None:
                prepared.cleanup()
            if bridge_path is not None:
                try:
                    os.remove(bridge_path)
                except FileNotFoundError:
                    pass

        generated_waveform = generated_waveform.detach().to(device="cpu", dtype=torch.float32)
        if generated_waveform.ndim == 1:
            generated_waveform = generated_waveform.unsqueeze(0)
        if generated_waveform.ndim != 2 or generated_waveform.shape[-1] == 0:
            raise RuntimeError(f"AuK returned invalid audio shape: {tuple(generated_waveform.shape)}.")
        if not torch.isfinite(generated_waveform).all():
            raise RuntimeError("AuK returned audio containing NaN or Inf.")

        audio_output = {
            "waveform": generated_waveform.unsqueeze(0),
            "sample_rate": int(sample_rate),
        }
        return (audio_output, final_instruction, prompt_enhancer_info)
