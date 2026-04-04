from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import soundfile as sf

from beats_generator.presets import PhaseSpec, SessionSpec

from .backgrounds import BackgroundGenerator, db_to_linear
from .filters import StereoSosFilter, design_high_shelf_sos, make_lowpass_sos

SAMPLE_RATE = 96_000
TARGET_RMS_DBFS = -18.0
TARGET_RMS_LINEAR = float(10.0 ** (TARGET_RMS_DBFS / 20.0))
TWO_PI = float(2.0 * np.pi)


@dataclass(frozen=True)
class RenderResult:
    output_path: Path
    duration_sec: float
    sample_count: int
    rms_dbfs: float
    peak_dbfs: float
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class PhaseSampleSpec:
    start_sample: int
    end_sample: int
    beat_hz_start: float
    beat_hz_end: float


@dataclass(frozen=True)
class PassStats:
    rms_linear: float
    peak_linear: float


@dataclass(frozen=True)
class RenderConfig:
    sample_rate: int = SAMPLE_RATE
    output_subtype: str = "FLOAT"
    target_rms_dbfs: float = TARGET_RMS_DBFS
    headphone_shelf_gain_db: float = 1.5
    headphone_shelf_hz: float = 8000.0
    headphone_shelf_q: float = 0.7
    post_lowpass_hz: float | None = None
    post_lowpass_order: int = 2
    background_gain_db_offset: float = 0.0
    background_profile: Literal["standard", "soft"] = "standard"
    clip_guard_threshold: float = 0.98


def render_session_to_wav(
    session: SessionSpec,
    output_path: Path,
    log: Callable[[str], None] = print,
    chunk_size: int = 1_048_576,
    seed: int = 7,
    config: RenderConfig | None = None,
) -> RenderResult:
    effective_config = config or RenderConfig()
    target_rms_linear = float(10.0 ** (effective_config.target_rms_dbfs / 20.0))
    total_samples = int(round(session.duration_sec * effective_config.sample_rate))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    phase_specs = _to_phase_sample_specs(
        phases=session.phases,
        sample_rate=effective_config.sample_rate,
    )
    for idx, phase in enumerate(phase_specs, start=1):
        duration_sec = (phase.end_sample - phase.start_sample) / effective_config.sample_rate
        log(
            "phase "
            f"{idx}: beat {phase.beat_hz_start:.2f}->{phase.beat_hz_end:.2f} Hz, "
            f"duration {duration_sec:.2f}s, samples {phase.end_sample - phase.start_sample}"
        )

    log("pass 1/2: measuring post-filter RMS")
    first_pass = _run_render_pass(
        session=session,
        phase_specs=phase_specs,
        total_samples=total_samples,
        chunk_size=chunk_size,
        seed=seed,
        gain=1.0,
        write_handle=None,
        config=effective_config,
    )

    gain = target_rms_linear / max(first_pass.rms_linear, 1e-18)
    log(
        f"normalization: input RMS {linear_to_db(first_pass.rms_linear):.2f} dBFS, "
        f"gain {gain:.6f}"
    )

    log(f"pass 2/2: writing {output_path}")
    with sf.SoundFile(
        str(output_path),
        mode="w",
        samplerate=effective_config.sample_rate,
        channels=2,
        subtype=effective_config.output_subtype,
        format="WAV",
    ) as wav_file:
        second_pass = _run_render_pass(
            session=session,
            phase_specs=phase_specs,
            total_samples=total_samples,
            chunk_size=chunk_size,
            seed=seed,
            gain=gain,
            write_handle=wav_file,
            config=effective_config,
        )

    warnings: list[str] = []
    if second_pass.peak_linear > effective_config.clip_guard_threshold:
        warning = (
            f"clip guard warning: post-clip peak exceeds {effective_config.clip_guard_threshold:.2f} "
            f"({second_pass.peak_linear:.6f})"
        )
        log(warning)
        warnings.append(warning)

    rms_dbfs = linear_to_db(second_pass.rms_linear)
    peak_dbfs = linear_to_db(second_pass.peak_linear)
    log(
        f"done: samples {total_samples}, RMS {rms_dbfs:.2f} dBFS, "
        f"peak {peak_dbfs:.2f} dBFS, output {output_path}"
    )
    return RenderResult(
        output_path=output_path,
        duration_sec=float(total_samples / effective_config.sample_rate),
        sample_count=total_samples,
        rms_dbfs=rms_dbfs,
        peak_dbfs=peak_dbfs,
        warnings=tuple(warnings),
    )


def linear_to_db(value: float) -> float:
    return float(20.0 * np.log10(max(value, 1e-18)))


def _run_render_pass(
    session: SessionSpec,
    phase_specs: tuple[PhaseSampleSpec, ...],
    total_samples: int,
    chunk_size: int,
    seed: int,
    gain: float,
    write_handle: sf.SoundFile | None,
    config: RenderConfig,
) -> PassStats:
    left_phase_state = 0.0
    right_phase_state = 0.0
    sum_squares = 0.0
    sample_value_count = 0
    peak_linear = 0.0

    background = BackgroundGenerator.create(
        mode=session.background,
        sample_rate=config.sample_rate,
        seed=seed,
        profile=config.background_profile,
    )
    bg_gain = background.mix_gain() * db_to_linear(config.background_gain_db_offset)
    headphone_filter = StereoSosFilter.create(
        design_high_shelf_sos(
            sample_rate=config.sample_rate,
            center_hz=config.headphone_shelf_hz,
            gain_db=config.headphone_shelf_gain_db,
            q=config.headphone_shelf_q,
        )
    )
    post_lowpass_filter: StereoSosFilter | None = None
    if (
        config.post_lowpass_hz is not None
        and config.post_lowpass_hz > 0.0
        and config.post_lowpass_hz < (config.sample_rate * 0.5)
    ):
        post_lowpass_filter = StereoSosFilter.create(
            make_lowpass_sos(
                sample_rate=config.sample_rate,
                cutoff_hz=config.post_lowpass_hz,
                order=config.post_lowpass_order,
            )
        )

    for start in range(0, total_samples, chunk_size):
        length = min(chunk_size, total_samples - start)
        left_freq = np.full(length, session.carrier_left_hz, dtype=np.float64)
        beat_hz = _beat_hz_for_chunk(
            start_sample=start, length=length, phase_specs=phase_specs
        )
        right_freq = left_freq + beat_hz

        left_phase, left_phase_state = _accumulate_phase(
            frequency_hz=left_freq,
            sample_rate=config.sample_rate,
            phase_state=left_phase_state,
        )
        right_phase, right_phase_state = _accumulate_phase(
            frequency_hz=right_freq,
            sample_rate=config.sample_rate,
            phase_state=right_phase_state,
        )

        carrier_stereo = np.column_stack(
            (
                np.sin(left_phase),
                np.sin(right_phase),
            )
        )
        envelope = _envelope_for_chunk(
            start_sample=start,
            length=length,
            total_samples=total_samples,
            sample_rate=config.sample_rate,
            fade_in_sec=session.fade_in_sec,
            fade_out_sec=session.fade_out_sec,
        )
        carrier_stereo *= envelope[:, None]

        background_mono = background.generate_mono(length) * bg_gain
        background_stereo = np.column_stack((background_mono, background_mono))
        mixed = carrier_stereo + background_stereo

        filtered = headphone_filter.process(mixed)
        if post_lowpass_filter is not None:
            filtered = post_lowpass_filter.process(filtered)
        if gain != 1.0:
            filtered *= gain
            np.clip(filtered, -1.0, 1.0, out=filtered)

        sum_squares += float(np.sum(filtered * filtered, dtype=np.float64))
        sample_value_count += filtered.size
        peak_linear = max(peak_linear, float(np.max(np.abs(filtered))))

        if write_handle is not None:
            write_handle.write(filtered.astype(np.float32, copy=False))

    return PassStats(
        rms_linear=float(np.sqrt(sum_squares / max(sample_value_count, 1))),
        peak_linear=peak_linear,
    )


def _to_phase_sample_specs(
    phases: tuple[PhaseSpec, ...], sample_rate: int
) -> tuple[PhaseSampleSpec, ...]:
    specs: list[PhaseSampleSpec] = []
    for phase in phases:
        specs.append(
            PhaseSampleSpec(
                start_sample=int(round(phase.start_sec * sample_rate)),
                end_sample=int(round(phase.end_sec * sample_rate)),
                beat_hz_start=phase.beat_hz_start,
                beat_hz_end=phase.beat_hz_end,
            )
        )
    return tuple(specs)


def _beat_hz_for_chunk(
    start_sample: int, length: int, phase_specs: tuple[PhaseSampleSpec, ...]
) -> np.ndarray:
    stop_sample = start_sample + length
    beat_hz = np.empty(length, dtype=np.float64)

    for phase in phase_specs:
        overlap_start = max(start_sample, phase.start_sample)
        overlap_end = min(stop_sample, phase.end_sample)
        if overlap_start >= overlap_end:
            continue
        write_start = overlap_start - start_sample
        write_end = overlap_end - start_sample
        local_indices = np.arange(overlap_start, overlap_end, dtype=np.float64)
        phase_span_samples = max(phase.end_sample - phase.start_sample, 1)

        if phase.beat_hz_start == phase.beat_hz_end or phase_span_samples <= 1:
            beat_hz[write_start:write_end] = phase.beat_hz_start
        else:
            progress = (local_indices - phase.start_sample) / float(phase_span_samples - 1)
            beat_hz[write_start:write_end] = (
                phase.beat_hz_start
                + progress * (phase.beat_hz_end - phase.beat_hz_start)
            )
    return beat_hz


def _accumulate_phase(
    frequency_hz: np.ndarray, sample_rate: int, phase_state: float
) -> tuple[np.ndarray, float]:
    phase_increments = TWO_PI * frequency_hz / sample_rate
    phase = np.cumsum(phase_increments, dtype=np.float64)
    phase += phase_state
    np.remainder(phase, TWO_PI, out=phase)
    return phase, float(phase[-1])


def _envelope_for_chunk(
    start_sample: int,
    length: int,
    total_samples: int,
    sample_rate: int,
    fade_in_sec: float,
    fade_out_sec: float,
) -> np.ndarray:
    envelope = np.ones(length, dtype=np.float64)
    indices = np.arange(start_sample, start_sample + length, dtype=np.float64)

    fade_in_samples = int(round(fade_in_sec * sample_rate))
    if fade_in_samples > 0:
        fade_in_mask = indices < fade_in_samples
        fade_in_t = indices[fade_in_mask] / sample_rate
        envelope[fade_in_mask] = 0.5 * (1.0 - np.cos(np.pi * fade_in_t / fade_in_sec))

    fade_out_samples = int(round(fade_out_sec * sample_rate))
    if fade_out_samples > 0:
        fade_out_start = total_samples - fade_out_samples
        fade_out_mask = indices >= fade_out_start
        fade_out_t = (indices[fade_out_mask] - fade_out_start) / sample_rate
        envelope[fade_out_mask] = 0.5 * (1.0 + np.cos(np.pi * fade_out_t / fade_out_sec))

    return envelope
