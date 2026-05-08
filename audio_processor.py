import numpy as np
import soundfile as sf
import librosa
import noisereduce as nr
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable
from scipy import signal as scipy_signal


@dataclass
class ProcessingParams:
    noise_intensity: float = 0.75   # 0.0–1.0 → prop_decrease in noisereduce
    stationary: bool = False        # True = ruído constante, False = adaptativo
    voice_boost_db: float = 0.0     # 0–12 dB, shelving em +3 kHz
    normalize: bool = True
    trim_silence: bool = False


def load_audio_info(path: str) -> dict:
    try:
        info = sf.info(path)
        return {
            "sample_rate": info.samplerate,
            "duration": info.duration,
            "channels": info.channels,
            "format": info.format,
            "size_mb": Path(path).stat().st_size / (1024 ** 2),
        }
    except Exception:
        # Fallback para MP3/M4A que soundfile não suporta diretamente
        y, sr = librosa.load(path, sr=None, mono=False, res_type="kaiser_fast")
        dur = y.shape[-1] / sr if y.ndim > 1 else len(y) / sr
        ch = y.shape[0] if y.ndim > 1 else 1
        return {
            "sample_rate": sr,
            "duration": dur,
            "channels": ch,
            "format": Path(path).suffix.upper().lstrip("."),
            "size_mb": Path(path).stat().st_size / (1024 ** 2),
        }


def load_waveform(path: str, max_points: int = 8000) -> tuple:
    """Carrega áudio como mono float32 com downsampling para exibição."""
    y, sr = librosa.load(path, sr=None, mono=True, res_type="kaiser_fast")
    y = y.astype(np.float32)
    # Computa envelope por chunks para visualização tipo audio editor
    chunk = max(1, len(y) // max_points)
    n = len(y) // chunk
    chunks = y[:n * chunk].reshape(n, chunk)
    envelope = np.max(np.abs(chunks), axis=1)
    return envelope, sr


def process_audio(
    input_path: str,
    output_path: str,
    params: ProcessingParams,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    stop_flag: Optional[list] = None,
) -> dict:
    def _prog(pct: float, msg: str):
        if progress_cb:
            progress_cb(pct, msg)

    def _check():
        if stop_flag and stop_flag[0]:
            raise InterruptedError("Processamento cancelado pelo usuário.")

    _prog(0.05, "Carregando áudio…")
    y, sr = librosa.load(input_path, sr=None, mono=False, res_type="kaiser_fast")
    y = y.astype(np.float32)
    _check()

    channels = [y] if y.ndim == 1 else [y[i] for i in range(y.shape[0])]
    n_ch = len(channels)

    _prog(0.15, f"Reduzindo ruído ({n_ch} canal{'is' if n_ch > 1 else ''})…")
    reduced = []
    for i, ch in enumerate(channels):
        _check()
        r = nr.reduce_noise(
            y=ch,
            sr=sr,
            prop_decrease=params.noise_intensity,
            stationary=params.stationary,
            n_jobs=1,
        )
        reduced.append(r.astype(np.float32))
        _prog(0.15 + 0.50 * (i + 1) / n_ch, f"Canal {i + 1}/{n_ch} processado")

    out = np.array(reduced) if n_ch > 1 else reduced[0]
    _check()

    if params.voice_boost_db > 0:
        _prog(0.68, "Aplicando boost de presença vocal…")
        out = _voice_boost(out, sr, params.voice_boost_db)
        _check()

    if params.normalize:
        _prog(0.80, "Normalizando volume…")
        out = _normalize(out)
        _check()

    if params.trim_silence and out.ndim == 1:
        _prog(0.88, "Cortando silêncio…")
        out, _ = librosa.effects.trim(out, top_db=30)

    _prog(0.93, "Salvando arquivo…")
    if out.ndim > 1:
        sf.write(output_path, out.T, sr, subtype="PCM_16")
    else:
        sf.write(output_path, out, sr, subtype="PCM_16")

    duration = (out.shape[-1] if out.ndim > 1 else len(out)) / sr
    _prog(1.0, "Concluído!")
    return {"sample_rate": sr, "duration": duration, "channels": n_ch}


def _voice_boost(audio: np.ndarray, sr: int, db: float) -> np.ndarray:
    gain = 10 ** (db / 20) - 1.0
    freq = min(3000.0, sr / 2 * 0.95)
    sos = scipy_signal.butter(2, freq / (sr / 2), btype="high", output="sos")
    if audio.ndim == 1:
        boost = scipy_signal.sosfilt(sos, audio).astype(np.float32)
        return audio + gain * boost
    result = np.empty_like(audio)
    for i in range(audio.shape[0]):
        boost = scipy_signal.sosfilt(sos, audio[i]).astype(np.float32)
        result[i] = audio[i] + gain * boost
    return result


def _normalize(audio: np.ndarray, target_db: float = -1.0) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak < 1e-9:
        return audio
    return audio * (10 ** (target_db / 20) / peak)
