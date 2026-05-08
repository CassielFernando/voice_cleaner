import numpy as np
import soundfile as sf
import librosa
import noisereduce as nr
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable
from scipy import signal as scipy_signal
from scipy.ndimage import uniform_filter1d


@dataclass
class ProcessingParams:
    noise_enabled: bool = True       # liga/desliga toda a etapa de ruído
    noise_intensity: float = 0.75    # 0.0–1.0 → prop_decrease em noisereduce
    # "adaptive"   = ruído variável (vento, ambiente)
    # "stationary" = ruído constante (AC, ventilador)
    # "percussive" = batidas/drums → HPSS remove percussão, depois noisereduce
    noise_type: str = "adaptive"
    voice_boost_db: float = 0.0      # 0–12 dB, shelving em +3 kHz
    normalize: bool = True
    trim_silence: bool = False
    compress: bool = False           # compressor de dinâmica
    compress_threshold_db: float = -20.0   # limiar em dBFS
    compress_ratio: float = 4.0      # proporção ex: 4 = 4:1


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
    """Retorna (envelope, audio_mono, sr) — envelope para display, audio para playback."""
    y, sr = librosa.load(path, sr=None, mono=True, res_type="kaiser_fast")
    y = y.astype(np.float32)
    chunk = max(1, len(y) // max_points)
    n = len(y) // chunk
    chunks = y[:n * chunk].reshape(n, chunk)
    envelope = np.max(np.abs(chunks), axis=1)
    return envelope, y, sr


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

    # ── Etapa 1: HPSS — remove percussão/batidas ─────────────────────────────
    if params.noise_enabled and params.noise_type == "percussive":
        _prog(0.12, "Removendo batidas/percussão (HPSS)…")
        isolated = []
        for i, ch in enumerate(channels):
            _check()
            isolated.append(_hpss_isolate(ch, margin=1.5))
            _prog(0.12 + 0.18 * (i + 1) / n_ch, f"Canal {i + 1}/{n_ch} — percussão removida")
        channels = isolated
        noise_start = 0.32
    else:
        noise_start = 0.12

    # ── Etapa 2: redução de ruído espectral ──────────────────────────────────
    if params.noise_enabled and params.noise_intensity > 0:
        stationary = (params.noise_type == "stationary")
        _prog(noise_start, f"Reduzindo ruído ({n_ch} canal{'is' if n_ch > 1 else ''})…")
        reduced = []
        for i, ch in enumerate(channels):
            _check()
            r = nr.reduce_noise(
                y=ch,
                sr=sr,
                prop_decrease=params.noise_intensity,
                stationary=stationary,
                n_jobs=1,
            )
            reduced.append(r.astype(np.float32))
            _prog(noise_start + 0.38 * (i + 1) / n_ch, f"Canal {i + 1}/{n_ch} processado")
        channels = reduced

    out = np.array(channels) if n_ch > 1 else channels[0]
    _check()

    # ── Etapa 3: boost de presença vocal ─────────────────────────────────────
    if params.voice_boost_db > 0:
        _prog(0.74, "Aplicando boost de presença vocal…")
        out = _voice_boost(out, sr, params.voice_boost_db)
        _check()

    # ── Etapa 4: compressão de dinâmica ──────────────────────────────────────
    if params.compress:
        _prog(0.82, "Comprimindo dinâmica…")
        if out.ndim == 1:
            out = _compress_audio(out, sr, params.compress_threshold_db, params.compress_ratio)
        else:
            result = np.empty_like(out)
            for i in range(out.shape[0]):
                result[i] = _compress_audio(out[i], sr, params.compress_threshold_db, params.compress_ratio)
            out = result
        _check()

    # ── Etapa 5: normalização ────────────────────────────────────────────────
    if params.normalize:
        _prog(0.90, "Normalizando volume…")
        out = _normalize(out)
        _check()

    if params.trim_silence and out.ndim == 1:
        _prog(0.94, "Cortando silêncio…")
        out, _ = librosa.effects.trim(out, top_db=30)

    _prog(0.97, "Salvando arquivo…")
    if out.ndim > 1:
        sf.write(output_path, out.T, sr, subtype="PCM_16")
    else:
        sf.write(output_path, out, sr, subtype="PCM_16")

    duration = (out.shape[-1] if out.ndim > 1 else len(out)) / sr
    _prog(1.0, "Concluído!")
    return {"sample_rate": sr, "duration": duration, "channels": n_ch}


def _compress_audio(audio: np.ndarray, sr: int, threshold_db: float, ratio: float) -> np.ndarray:
    """
    Compressor de dinâmica RMS.
    Nivela picos altos com partes mais baixas — voz fica com volume mais uniforme.
    Usa makeup gain automático para preservar o nível médio original.
    """
    audio = audio.astype(np.float64)
    threshold_lin = 10 ** (threshold_db / 20.0)

    # Envelope RMS com janela de 20ms (suaviza rápidas variações)
    window = max(64, int(sr * 0.020))
    rms = np.sqrt(uniform_filter1d(audio ** 2, size=window) + 1e-12)

    # Curva de ganho: sem compressão abaixo do limiar; acima → reduz pelo ratio
    gain = np.ones_like(rms)
    above = rms > threshold_lin
    if np.any(above):
        gain[above] = (threshold_lin * (rms[above] / threshold_lin) ** (1.0 / ratio)) / rms[above]

    # Suaviza a curva de ganho para evitar artefatos (ataque 5ms, release 120ms)
    attack_w  = max(1, int(sr * 0.005))
    release_w = max(1, int(sr * 0.120))
    gain_smooth = uniform_filter1d(gain, size=(attack_w + release_w) // 2)

    compressed = (audio * gain_smooth).astype(np.float32)

    # Makeup gain: restaura RMS médio original
    orig_rms = float(np.sqrt(np.mean(audio ** 2) + 1e-12))
    comp_rms = float(np.sqrt(np.mean(compressed.astype(np.float64) ** 2) + 1e-12))
    if comp_rms > 1e-9:
        compressed = (compressed * (orig_rms / comp_rms)).astype(np.float32)

    return compressed


def _hpss_isolate(audio: np.ndarray, margin: float) -> np.ndarray:
    D = librosa.stft(audio)
    H, _ = librosa.decompose.hpss(D, margin=margin)
    return librosa.istft(H, length=len(audio)).astype(np.float32)


def _voice_boost(audio: np.ndarray, sr: int, db: float) -> np.ndarray:
    gain = 10 ** (db / 20) - 1.0
    freq = min(3000.0, sr / 2 * 0.95)
    sos = scipy_signal.butter(2, freq / (sr / 2), btype="high", output="sos")
    if audio.ndim == 1:
        return audio + gain * scipy_signal.sosfilt(sos, audio).astype(np.float32)
    result = np.empty_like(audio)
    for i in range(audio.shape[0]):
        result[i] = audio[i] + gain * scipy_signal.sosfilt(sos, audio[i]).astype(np.float32)
    return result


def _normalize(audio: np.ndarray, target_db: float = -1.0) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak < 1e-9:
        return audio
    return audio * (10 ** (target_db / 20) / peak)
