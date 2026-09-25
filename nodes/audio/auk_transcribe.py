"""SFAuKAudioTranscribe：本地语音识别（复用 AuK 提示词增强的 SenseVoiceSmall 提供方）。

输入 ComfyUI AUDIO（批大小 1，多声道取均值转单声道），输出识别文字。识别在本机 CPU 完成
（funasr SenseVoiceSmall，首次使用经 ModelScope 下载并缓存），不联网上传音频。

复用：`auk_generate.normalize_audio`（AUDIO 校验/单声道化）、`auk.infer.audio_io.write_wav`
（临时 WAV 桥接）、`auk.infer.pe.SenseVoiceSmallASR`（引擎侧新增可选 language 参数）。
"""

import os
import tempfile

from .auk_generate import normalize_audio

_CATEGORY = "sfnodes/audio"

# 显示名 → SenseVoice 语言码（auto/zh/en/yue/ja/ko）
LANGUAGE_LABELS = {
    "自动": "auto",
    "中文": "zh",
    "英文": "en",
    "粤语": "yue",
    "日语": "ja",
    "韩语": "ko",
}


class SFAuKAudioTranscribe:
    DESCRIPTION = (
        "本地语音识别（SenseVoiceSmall，复用 AuK 引擎 ASR 栈）：输入 AUDIO 输出对应文字，"
        "目标语言默认自动（可指定中文/英文/粤语/日语/韩语）；识别在本机 CPU 完成，不上传音频；"
        "首次使用会自动下载模型（约 900MB，缓存后不再下载）"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "待识别音频；批大小必须为 1，多声道取均值转单声道",
                }),
                "language": (list(LANGUAGE_LABELS), {
                    "default": "自动",
                    "tooltip": "识别目标语言；自动 = 由模型判断（SenseVoice 支持中/英/粤/日/韩）",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_TOOLTIPS = ("识别文字（含标点与 ITN 规范化）",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    def execute(self, audio, language="自动"):
        from .auk.infer.audio_io import write_wav

        code = LANGUAGE_LABELS.get(language)
        if code is None:
            raise ValueError(f"未知目标语言：{language!r}")
        normalized = normalize_audio(audio)
        if normalized is None:
            raise ValueError("audio 输入为空")
        waveform, sample_rate = normalized

        descriptor, path = tempfile.mkstemp(prefix="sf_auk_asr_", suffix=".wav")
        os.close(descriptor)
        try:
            write_wav(path, waveform, sample_rate)
            from .auk.infer.pe import SenseVoiceSmallASR

            result = SenseVoiceSmallASR(language=code).transcribe(path)
        finally:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

        if result.error:
            raise ValueError(f"语音识别失败：{result.error}")
        text = (result.text or "").strip()
        if not text:
            raise ValueError("语音识别没有返回文本（音频可能无人声或过短）")
        return (text,)
