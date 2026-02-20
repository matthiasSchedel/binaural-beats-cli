from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from beats_generator.generators.binaural import linear_to_db
from beats_generator.presets import PRESET_OUTPUT_FILENAMES, PRESETS, get_preset_spec
from beats_generator.recommended import (
    get_recommended_session_by_output_file,
    get_recommended_session_by_preset_id,
    list_recommended_sessions,
)
from beats_generator.test_pack import (
    get_test_session_by_output_file,
    get_test_session_by_preset_id,
    list_test_sessions,
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

        sum_squares = 0.0
        sample_count = 0
        peak = 0.0
        analysis_left: list[np.ndarray] = []
        analysis_right: list[np.ndarray] = []
        max_analysis_frames = min(total_frames, sample_rate * 600)
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

    expected_low_hz, expected_high_hz = _expected_range(preset_id)
    left = np.concatenate(analysis_left, axis=0)
    right = np.concatenate(analysis_right, axis=0)
    analysis_sample_rate = sample_rate / analysis_decimation
    dominant_beat_hz = _dominant_beat_frequency(
        left=left,
        right=right,
        sample_rate=analysis_sample_rate,
        expected_low_hz=expected_low_hz,
        expected_high_hz=expected_high_hz,
    )
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
    left: np.ndarray,
    right: np.ndarray,
    sample_rate: float,
    expected_low_hz: float,
    expected_high_hz: float,
) -> float:
    if left.size < 32 or right.size < 32:
        return 0.0
    analytic_left = signal.hilbert(left)
    analytic_right = signal.hilbert(right)
    phase_diff = np.unwrap(np.angle(analytic_right) - np.angle(analytic_left))
    instantaneous = np.diff(phase_diff) * (sample_rate / (2.0 * np.pi))
    beat_hz = np.abs(instantaneous)
    amp = np.sqrt(np.abs(analytic_left[:-1] * analytic_right[:-1]))
    amp_floor = float(np.percentile(amp, 20.0))
    low = max(0.5, expected_low_hz - 0.75)
    high = min(50.0, expected_high_hz + 0.75)
    mask = (beat_hz >= low) & (beat_hz <= high) & (amp >= amp_floor)
    if not np.any(mask):
        return 0.0
    return float(np.median(beat_hz[mask]))


def _expected_range(preset_id: str | None) -> tuple[float, float]:
    if preset_id is None:
        return 0.5, 50.0
    if preset_id in PRESETS:
        spec = get_preset_spec(preset_id)
    else:
        recommended = get_recommended_session_by_preset_id(preset_id)
        if recommended is not None:
            spec = recommended.spec
        else:
            test_session = get_test_session_by_preset_id(preset_id)
            if test_session is None:
                return 0.5, 50.0
            spec = test_session.spec
    beat_values: list[float] = []
    for phase in spec.phases:
        beat_values.append(phase.beat_hz_start)
        beat_values.append(phase.beat_hz_end)
    low = max(0.5, min(beat_values) - 1.0)
    high = min(50.0, max(beat_values) + 1.0)
    return low, high


def _infer_preset_id(path: Path) -> str | None:
    name = path.name
    for preset_id, filename in PRESET_OUTPUT_FILENAMES.items():
        if name == filename:
            return preset_id
    recommended = get_recommended_session_by_output_file(name)
    if recommended is not None:
        return recommended.preset_id
    test_session = get_test_session_by_output_file(name)
    if test_session is not None:
        return test_session.preset_id
    stem = path.stem
    for preset_id in PRESETS:
        if stem.startswith(preset_id):
            return preset_id
    for session in list_recommended_sessions():
        if stem.startswith(session.preset_id):
            return session.preset_id
    for session in list_test_sessions():
        if stem.startswith(session.preset_id):
            return session.preset_id
    return None
