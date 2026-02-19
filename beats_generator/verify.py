from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from beats_generator.generators.binaural import SAMPLE_RATE, linear_to_db
from beats_generator.presets import PRESET_OUTPUT_FILENAMES, PRESETS, get_preset_spec
from beats_generator.recommended import (
    get_recommended_session_by_output_file,
    get_recommended_session_by_preset_id,
    list_recommended_sessions,
)


@dataclass(frozen=True)
class VerificationResult:
    path: Path
    duration_sec: float
    rms_dbfs: float
    peak_dbfs: float
    dominant_beat_hz: float
    expected_low_hz: float
    expected_high_hz: float
    passed: bool


def verify_wav(path: Path, preset_id: str | None = None) -> VerificationResult:
    if preset_id is None:
        preset_id = _infer_preset_id(path)

    with sf.SoundFile(str(path), mode="r") as wav_file:
        sample_rate = wav_file.samplerate
        channels = wav_file.channels
        total_frames = wav_file.frames
        if channels != 2:
            raise ValueError("verify requires stereo WAV input")
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"expected sample rate {SAMPLE_RATE}, found {sample_rate}")

        sum_squares = 0.0
        sample_count = 0
        peak = 0.0
        analysis_left: list[np.ndarray] = []
        analysis_right: list[np.ndarray] = []
        max_analysis_frames = min(total_frames, SAMPLE_RATE * 600)
        collected_frames = 0
        analysis_decimation = max(1, sample_rate // 400)

        for block in wav_file.blocks(
            blocksize=262_144, dtype="float64", always_2d=True, fill_value=None
        ):
            sum_squares += float(np.sum(block * block, dtype=np.float64))
            sample_count += block.size
            peak = max(peak, float(np.max(np.abs(block))))

            if collected_frames < max_analysis_frames:
                remaining = max_analysis_frames - collected_frames
                window = block[:remaining]
                downsampled = window[::analysis_decimation]
                analysis_left.append(downsampled[:, 0].copy())
                analysis_right.append(downsampled[:, 1].copy())
                collected_frames += window.shape[0]

    duration_sec = float(total_frames / sample_rate)
    rms_linear = float(np.sqrt(sum_squares / max(sample_count, 1)))
    rms_dbfs = linear_to_db(rms_linear)
    peak_dbfs = linear_to_db(peak)

    left = np.concatenate(analysis_left, axis=0)
    right = np.concatenate(analysis_right, axis=0)
    analysis_sample_rate = sample_rate / analysis_decimation
    dominant_beat_hz = _dominant_beat_frequency(left, right, analysis_sample_rate)

    expected_low_hz, expected_high_hz = _expected_range(preset_id)
    passed = expected_low_hz <= dominant_beat_hz <= expected_high_hz
    return VerificationResult(
        path=path,
        duration_sec=duration_sec,
        rms_dbfs=rms_dbfs,
        peak_dbfs=peak_dbfs,
        dominant_beat_hz=dominant_beat_hz,
        expected_low_hz=expected_low_hz,
        expected_high_hz=expected_high_hz,
        passed=passed,
    )


def _dominant_beat_frequency(
    left: np.ndarray, right: np.ndarray, sample_rate: float
) -> float:
    if left.size < 32 or right.size < 32:
        return 0.0
    analytic_left = signal.hilbert(left)
    analytic_right = signal.hilbert(right)
    phase_diff = np.unwrap(np.angle(analytic_left) - np.angle(analytic_right))
    phase_signal = np.sin(phase_diff)
    phase_signal -= np.mean(phase_signal)
    windowed = phase_signal * np.hanning(phase_signal.size)
    spectrum = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(windowed.size, d=1.0 / sample_rate)
    band_mask = (freqs >= 0.5) & (freqs <= 50.0)
    if not np.any(band_mask):
        return 0.0
    band_magnitudes = np.abs(spectrum[band_mask])
    band_freqs = freqs[band_mask]
    peak_index = int(np.argmax(band_magnitudes))
    return float(band_freqs[peak_index])


def _expected_range(preset_id: str | None) -> tuple[float, float]:
    if preset_id is None:
        return 0.5, 50.0
    if preset_id in PRESETS:
        spec = get_preset_spec(preset_id)
    else:
        recommended = get_recommended_session_by_preset_id(preset_id)
        if recommended is None:
            return 0.5, 50.0
        spec = recommended.spec
    beat_values: list[float] = []
    for phase in spec.phases:
        beat_values.append(phase.beat_hz_start)
        beat_values.append(phase.beat_hz_end)
    low = max(0.5, min(beat_values) - 0.5)
    high = min(50.0, max(beat_values) + 0.5)
    return low, high


def _infer_preset_id(path: Path) -> str | None:
    name = path.name
    for preset_id, filename in PRESET_OUTPUT_FILENAMES.items():
        if name == filename:
            return preset_id
    recommended = get_recommended_session_by_output_file(name)
    if recommended is not None:
        return recommended.preset_id
    stem = path.stem
    for preset_id in PRESETS:
        if stem.startswith(preset_id):
            return preset_id
    for session in list_recommended_sessions():
        if stem.startswith(session.preset_id):
            return session.preset_id
    return None
