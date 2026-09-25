"""WAV bridge for ComfyUI and Prompt Enhancer without TorchCodec."""
import soundfile as sf
import torch


def read_audio(path):
    samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(samples.T.copy()), sample_rate


def write_wav(path, waveform, sample_rate, *, subtype="FLOAT"):
    waveform = waveform.detach().to(device="cpu", dtype=torch.float32)
    if waveform.ndim != 2:
        raise ValueError("Expected audio shaped [channels, samples].")
    sf.write(str(path), waveform.T.numpy(), sample_rate, format="WAV", subtype=subtype)


def audio_duration(path):
    info = sf.info(str(path))
    if info.samplerate <= 0:
        raise ValueError("Audio sample rate must be positive.")
    return info.frames / info.samplerate
