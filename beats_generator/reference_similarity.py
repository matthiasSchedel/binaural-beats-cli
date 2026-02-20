from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
from scipy import signal

from beats_generator.generators.backgrounds import BackgroundGenerator, db_to_linear
from beats_generator.generators.filters import StereoSosFilter, make_lowpass_sos

TWO_PI = float(2.0 * np.pi)
PROFILE_VERSION = 2
ANALYSIS_BANDS: tuple[tuple[int, int], ...] = (
    (0, 40),
    (40, 120),
    (120, 300),
    (300, 800),
    (800, 2000),
    (2000, 6000),
    (6000, 12000),
    (12000, 20000),
)

BackgroundMode = Literal["rain", "pink_noise", "brown_noise", "none"]


@dataclass(frozen=True)
class TonePeak:
    frequency_hz: float
    mean_level_db: float
    presence_ratio: float


@dataclass(frozen=True)
class ReferenceProfile:
    profile_version: int
    source_path: str
    sample_rate: int
    duration_sec: float
    rms_dbfs: float
    peak_dbfs: float
    crest_factor_db: float
    window_rms_range_db: float
    stereo_corr_median: float
    stereo_corr_p10: float
    stereo_corr_p90: float
    stereo_corr_std: float
    side_mid_db_median: float
    spectral_centroid_hz_median: float
    rolloff_85_hz_median: float
    tone_peaks: tuple[TonePeak, ...]
    modulation_hz: tuple[float, ...]
    modulation_env_hz: tuple[float, ...]
    band_energy_db: tuple[float, ...]
    inferred_background: BackgroundMode
    inferred_lowpass_hz: float

    def to_json(self) -> dict[str, object]:
        payload = asdict(self)
        payload["tone_peaks"] = [asdict(item) for item in self.tone_peaks]
        payload["modulation_hz"] = list(self.modulation_hz)
        payload["modulation_env_hz"] = list(self.modulation_env_hz)
        payload["band_energy_db"] = list(self.band_energy_db)
        payload["analysis_bands_hz"] = [list(item) for item in ANALYSIS_BANDS]
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> "ReferenceProfile":
        peaks_raw = payload.get("tone_peaks")
        if not isinstance(peaks_raw, list):
            raise ValueError("tone_peaks must be a list")
        peaks: list[TonePeak] = []
        for item in peaks_raw:
            if not isinstance(item, dict):
                raise ValueError("tone_peaks entries must be objects")
            peaks.append(
                TonePeak(
                    frequency_hz=float(item["frequency_hz"]),
                    mean_level_db=float(item["mean_level_db"]),
                    presence_ratio=float(item["presence_ratio"]),
                )
            )

        modulation_raw = payload.get("modulation_hz")
        if not isinstance(modulation_raw, list):
            raise ValueError("modulation_hz must be a list")
        modulation_env_raw = payload.get("modulation_env_hz", modulation_raw)
        if not isinstance(modulation_env_raw, list):
            raise ValueError("modulation_env_hz must be a list")

        crest_factor_db = float(
            payload.get(
                "crest_factor_db",
                float(payload["peak_dbfs"]) - float(payload["rms_dbfs"]),
            )
        )
        window_rms_range_db = float(payload.get("window_rms_range_db", 6.0))
        stereo_corr_std = float(
            payload.get(
                "stereo_corr_std",
                (float(payload["stereo_corr_p90"]) - float(payload["stereo_corr_p10"])) / 2.56,
            )
        )
        band_energy_raw = payload.get("band_energy_db")
        if isinstance(band_energy_raw, list):
            band_energy_db = tuple(float(value) for value in band_energy_raw)
        else:
            band_energy_db = _default_band_energy_from_legacy(
                centroid_hz=float(payload["spectral_centroid_hz_median"]),
                rolloff_hz=float(payload["rolloff_85_hz_median"]),
            )
        if len(band_energy_db) != len(ANALYSIS_BANDS):
            band_energy_db = _normalize_band_energy_length(band_energy_db)

        return cls(
            profile_version=int(payload.get("profile_version", 1)),
            source_path=str(payload["source_path"]),
            sample_rate=int(payload["sample_rate"]),
            duration_sec=float(payload["duration_sec"]),
            rms_dbfs=float(payload["rms_dbfs"]),
            peak_dbfs=float(payload["peak_dbfs"]),
            crest_factor_db=crest_factor_db,
            window_rms_range_db=window_rms_range_db,
            stereo_corr_median=float(payload["stereo_corr_median"]),
            stereo_corr_p10=float(payload["stereo_corr_p10"]),
            stereo_corr_p90=float(payload["stereo_corr_p90"]),
            stereo_corr_std=stereo_corr_std,
            side_mid_db_median=float(payload["side_mid_db_median"]),
            spectral_centroid_hz_median=float(payload["spectral_centroid_hz_median"]),
            rolloff_85_hz_median=float(payload["rolloff_85_hz_median"]),
            tone_peaks=tuple(peaks),
            modulation_hz=tuple(float(x) for x in modulation_raw),
            modulation_env_hz=tuple(float(x) for x in modulation_env_raw),
            band_energy_db=band_energy_db,
            inferred_background=_validate_background_mode(str(payload["inferred_background"])),
            inferred_lowpass_hz=float(payload["inferred_lowpass_hz"]),
        )


@dataclass(frozen=True)
class VariantSpec:
    name: str
    tone_shift_hz: float
    modulation_scale: float
    modulation_depth: float
    stereo_width: float
    background_gain_db_offset: float
    lowpass_scale: float
    channel_detune_hz: float
    seed_offset: int


@dataclass(frozen=True)
class VariantRenderResult:
    variant: str
    output_path: Path
    duration_sec: float
    sample_count: int
    rms_dbfs: float
    peak_dbfs: float


@dataclass
class _VariantState:
    left_phase: np.ndarray
    right_phase: np.ndarray
    mod_phase: np.ndarray


@dataclass(frozen=True)
class _PassStats:
    rms_linear: float
    peak_linear: float


def analyze_reference(
    input_path: Path,
    window_sec: float = 20.0,
    hop_sec: float = 20.0,
    max_tones: int = 8,
    max_modulations: int = 3,
) -> ReferenceProfile:
    with sf.SoundFile(str(input_path), mode="r") as handle:
        sample_rate = handle.samplerate
        total_samples = handle.frames
        if handle.channels != 2:
            raise ValueError("Reference analysis requires a stereo source")

    rms_linear, peak_linear = _scan_global_levels(path=input_path)

    window_samples = int(round(window_sec * sample_rate))
    hop_samples = int(round(hop_sec * sample_rate))

    metrics: list[tuple[float, float, float, float, float, float]] = []
    tone_stats: dict[float, list[float]] = {}
    modulation_bins: list[float] = []
    modulation_env_bins: list[float] = []

    with sf.SoundFile(str(input_path), mode="r") as handle:
        start_sample = 0
        while start_sample + window_samples <= total_samples:
            handle.seek(start_sample)
            block = handle.read(window_samples, dtype="float64", always_2d=True)
            left = block[:, 0]
            right = block[:, 1]
            mid = 0.5 * (left + right)
            side = 0.5 * (left - right)

            block_rms = _rms_linear(block)
            mid_rms = _rms_linear(mid)
            side_rms = _rms_linear(side)
            corr = float(np.corrcoef(left, right)[0, 1]) if left.size > 1 else 0.0

            freqs, psd = signal.welch(
                mid,
                fs=sample_rate,
                nperseg=min(32768, mid.size),
                noverlap=min(16384, max(mid.size - 1, 0)),
            )
            centroid = float(np.sum(freqs * psd) / (np.sum(psd) + 1e-18))
            cumulative = np.cumsum(psd)
            rolloff_index = int(np.searchsorted(cumulative, 0.85 * cumulative[-1]))
            rolloff_hz = float(freqs[min(rolloff_index, freqs.size - 1)])

            metrics.append(
                (
                    linear_to_db(block_rms),
                    linear_to_db(mid_rms),
                    linear_to_db(side_rms),
                    corr,
                    centroid,
                    rolloff_hz,
                )
            )

            peak_freqs, peak_levels = _extract_tone_peaks(freqs=freqs, psd=psd)
            for frequency_hz, level_db in zip(peak_freqs, peak_levels, strict=True):
                binned_hz = round(frequency_hz * 2.0) / 2.0
                tone_stats.setdefault(binned_hz, []).append(level_db)

            if peak_freqs.size >= 2:
                differences = np.diff(np.sort(peak_freqs))
                valid = differences[(differences >= 1.0) & (differences <= 40.0)]
                for diff_hz in valid:
                    modulation_bins.append(round(diff_hz * 2.0) / 2.0)

            for mod_hz in _extract_envelope_modulation_peaks(mid=mid, sample_rate=sample_rate):
                modulation_env_bins.append(round(mod_hz * 2.0) / 2.0)

            start_sample += hop_samples

    if not metrics:
        raise ValueError("Reference audio is too short for analysis window")

    metric_values = np.asarray(metrics, dtype=np.float64)
    tone_peaks = _summarize_tone_peaks(
        tone_stats=tone_stats,
        window_count=len(metrics),
        max_tones=max_tones,
    )
    modulation_hz = _summarize_modulation(
        modulation_bins=modulation_bins,
        window_count=len(metrics),
        max_modulations=max_modulations,
    )
    modulation_env_hz = _summarize_modulation(
        modulation_bins=modulation_env_bins,
        window_count=len(metrics),
        max_modulations=max_modulations,
    )

    centroid_median = float(np.median(metric_values[:, 4]))
    rolloff_median = float(np.median(metric_values[:, 5]))
    band_energy_db = _compute_band_energy_db(input_path=input_path)
    crest_factor_db = linear_to_db(peak_linear) - linear_to_db(rms_linear)
    window_rms_range_db = float(np.percentile(metric_values[:, 0], 95) - np.percentile(metric_values[:, 0], 5))

    inferred_background = _infer_background(
        centroid_hz=centroid_median,
        rolloff_hz=rolloff_median,
    )
    inferred_lowpass_hz = float(
        np.clip(
            max(
                np.percentile(metric_values[:, 5], 90) * 1.35,
                centroid_median * 2.7,
            ),
            1200.0,
            5200.0,
        )
    )

    return ReferenceProfile(
        profile_version=PROFILE_VERSION,
        source_path=str(input_path),
        sample_rate=sample_rate,
        duration_sec=float(total_samples / sample_rate),
        rms_dbfs=linear_to_db(rms_linear),
        peak_dbfs=linear_to_db(peak_linear),
        crest_factor_db=crest_factor_db,
        window_rms_range_db=window_rms_range_db,
        stereo_corr_median=float(np.median(metric_values[:, 3])),
        stereo_corr_p10=float(np.percentile(metric_values[:, 3], 10)),
        stereo_corr_p90=float(np.percentile(metric_values[:, 3], 90)),
        stereo_corr_std=float(np.std(metric_values[:, 3])),
        side_mid_db_median=float(np.median(metric_values[:, 2] - metric_values[:, 1])),
        spectral_centroid_hz_median=centroid_median,
        rolloff_85_hz_median=rolloff_median,
        tone_peaks=tone_peaks,
        modulation_hz=modulation_hz,
        modulation_env_hz=modulation_env_hz,
        band_energy_db=band_energy_db,
        inferred_background=inferred_background,
        inferred_lowpass_hz=inferred_lowpass_hz,
    )


def save_reference_profile(profile: ReferenceProfile, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profile.to_json(), indent=2), encoding="utf-8")


def load_reference_profile(input_path: Path) -> ReferenceProfile:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Profile JSON must be an object")
    return ReferenceProfile.from_json(payload)


def generate_variants_from_profile(
    profile: ReferenceProfile,
    output_dir: Path,
    duration_sec: int | None = None,
    sample_rate: int = 48_000,
    chunk_size: int = 131_072,
) -> tuple[VariantRenderResult, ...]:
    effective_duration = duration_sec if duration_sec is not None else int(round(profile.duration_sec))
    variant_specs = _default_variant_specs()
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[VariantRenderResult] = []
    for variant in variant_specs:
        output_path = output_dir / f"dive_style_{variant.name}.flac"
        result = _render_variant(
            profile=profile,
            variant=variant,
            output_path=output_path,
            duration_sec=effective_duration,
            sample_rate=sample_rate,
            chunk_size=chunk_size,
        )
        results.append(result)
    return tuple(results)


def linear_to_db(value: float) -> float:
    return float(20.0 * np.log10(max(value, 1e-18)))


def _scan_global_levels(path: Path) -> tuple[float, float]:
    sum_squares = 0.0
    sample_count = 0
    peak = 0.0
    with sf.SoundFile(str(path), mode="r") as handle:
        for block in handle.blocks(blocksize=262_144, dtype="float64", always_2d=True):
            sum_squares += float(np.sum(block * block, dtype=np.float64))
            sample_count += block.size
            peak = max(peak, float(np.max(np.abs(block))))

    rms = float(np.sqrt(sum_squares / max(sample_count, 1)))
    return rms, peak


def _rms_linear(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values * values, dtype=np.float64) + 1e-18))


def _extract_tone_peaks(freqs: np.ndarray, psd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    band = (freqs >= 80.0) & (freqs <= 700.0)
    selected_freqs = freqs[band]
    selected_levels_db = 10.0 * np.log10(np.maximum(psd[band], 1e-30))
    peaks, _ = signal.find_peaks(selected_levels_db, prominence=6.0, distance=6)
    if peaks.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    top = np.argsort(selected_levels_db[peaks])[-12:]
    chosen = peaks[top]
    chosen_freqs = selected_freqs[chosen]
    chosen_levels = selected_levels_db[chosen]

    order = np.argsort(chosen_freqs)
    return chosen_freqs[order], chosen_levels[order]


def _summarize_tone_peaks(
    tone_stats: dict[float, list[float]],
    window_count: int,
    max_tones: int,
) -> tuple[TonePeak, ...]:
    if not tone_stats:
        return tuple()

    min_presence = max(2, int(round(window_count * 0.15)))
    scored: list[tuple[int, float, float]] = []
    for frequency_hz, levels_db in tone_stats.items():
        count = len(levels_db)
        if count < min_presence:
            continue
        mean_level = float(np.mean(levels_db))
        scored.append((count, mean_level, frequency_hz))

    if not scored:
        for frequency_hz, levels_db in tone_stats.items():
            scored.append((len(levels_db), float(np.mean(levels_db)), frequency_hz))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = scored[:max_tones]
    selected.sort(key=lambda item: item[2])

    peaks = [
        TonePeak(
            frequency_hz=float(frequency_hz),
            mean_level_db=float(mean_level),
            presence_ratio=float(count / max(window_count, 1)),
        )
        for count, mean_level, frequency_hz in selected
    ]
    return tuple(peaks)


def _summarize_modulation(
    modulation_bins: list[float],
    window_count: int,
    max_modulations: int,
) -> tuple[float, ...]:
    if not modulation_bins:
        return (8.0, 16.0)

    counts: dict[float, int] = {}
    for item in modulation_bins:
        counts[item] = counts.get(item, 0) + 1

    min_presence = max(2, int(round(window_count * 0.10)))
    scored = [
        (count, freq_hz)
        for freq_hz, count in counts.items()
        if count >= min_presence
    ]

    if not scored:
        scored = [(count, freq_hz) for freq_hz, count in counts.items()]

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [freq_hz for _, freq_hz in scored[:max_modulations]]
    selected.sort()
    return tuple(float(freq_hz) for freq_hz in selected)


def _extract_envelope_modulation_peaks(mid: np.ndarray, sample_rate: int) -> tuple[float, ...]:
    if mid.size < 4096:
        return tuple()
    band = signal.sosfiltfilt(
        signal.butter(4, [80.0, 500.0], btype="bandpass", fs=sample_rate, output="sos"),
        mid,
    )
    envelope = np.abs(signal.hilbert(band))
    envelope -= np.mean(envelope)
    envelope = signal.sosfiltfilt(
        signal.butter(4, 40.0, btype="lowpass", fs=sample_rate, output="sos"),
        envelope,
    )
    freqs, psd = signal.welch(
        envelope,
        fs=sample_rate,
        nperseg=min(32768, envelope.size),
        noverlap=min(16384, max(envelope.size - 1, 0)),
    )
    mask = (freqs >= 0.5) & (freqs <= 40.0)
    selected_freqs = freqs[mask]
    selected_psd = psd[mask]
    if selected_freqs.size == 0:
        return tuple()

    peaks, _ = signal.find_peaks(selected_psd, distance=2)
    if peaks.size == 0:
        return (float(selected_freqs[np.argmax(selected_psd)]),)
    top = np.argsort(selected_psd[peaks])[-3:]
    values = selected_freqs[peaks][top]
    values.sort()
    return tuple(float(value) for value in values)


def _compute_band_energy_db(input_path: Path) -> tuple[float, ...]:
    with sf.SoundFile(str(input_path), mode="r") as handle:
        sample_rate = handle.samplerate
        total_samples = handle.frames
        segment_samples = min(total_samples, sample_rate * 600)
        segment_start = max(0, (total_samples - segment_samples) // 2)
        handle.seek(segment_start)
        segment = handle.read(segment_samples, dtype="float64", always_2d=True)

    mono = 0.5 * (segment[:, 0] + segment[:, 1])
    freqs, psd = signal.welch(
        mono,
        fs=sample_rate,
        nperseg=min(65536, mono.size),
        noverlap=min(32768, max(mono.size - 1, 0)),
    )
    values: list[float] = []
    for low_hz, high_hz in ANALYSIS_BANDS:
        mask = (freqs >= low_hz) & (freqs < high_hz)
        energy = float(np.trapezoid(psd[mask], freqs[mask])) if np.any(mask) else 1e-30
        values.append(float(10.0 * np.log10(max(energy, 1e-30))))
    return tuple(values)


def _default_band_energy_from_legacy(centroid_hz: float, rolloff_hz: float) -> tuple[float, ...]:
    base = np.array([-72.0, -45.0, -40.0, -38.0, -40.0, -44.0, -48.0, -53.0], dtype=np.float64)
    brightness = float(np.clip((centroid_hz - 350.0) / 1200.0, 0.0, 1.0))
    air = float(np.clip((rolloff_hz - 500.0) / 1800.0, 0.0, 1.0))
    base[2] -= 8.0 * brightness
    base[3] -= 7.0 * brightness
    base[4] += 4.0 * brightness
    base[5] += 6.0 * brightness
    base[6] += 8.0 * air
    base[7] += 6.0 * air
    return tuple(float(value) for value in base)


def _normalize_band_energy_length(values: tuple[float, ...]) -> tuple[float, ...]:
    if not values:
        return _default_band_energy_from_legacy(centroid_hz=500.0, rolloff_hz=700.0)
    if len(values) >= len(ANALYSIS_BANDS):
        return tuple(float(value) for value in values[: len(ANALYSIS_BANDS)])
    padded = list(values)
    while len(padded) < len(ANALYSIS_BANDS):
        padded.append(float(padded[-1]))
    return tuple(float(value) for value in padded)


def _band_group_levels(band_energy_db: tuple[float, ...]) -> tuple[float, float, float]:
    normalized = _normalize_band_energy_length(band_energy_db)
    low_mid = float(np.mean(np.array([normalized[2], normalized[3]], dtype=np.float64)))
    presence = float(np.mean(np.array([normalized[4], normalized[5]], dtype=np.float64)))
    air = float(np.mean(np.array([normalized[6], normalized[7]], dtype=np.float64)))
    return low_mid, presence, air


def _infer_background(centroid_hz: float, rolloff_hz: float) -> BackgroundMode:
    if rolloff_hz >= 850.0 and centroid_hz >= 420.0:
        return "rain"
    if centroid_hz >= 300.0:
        return "pink_noise"
    return "brown_noise"


def _validate_background_mode(value: str) -> BackgroundMode:
    if value in {"rain", "pink_noise", "brown_noise", "none"}:
        return value
    raise ValueError(f"Unsupported background mode in profile: {value}")


def _default_variant_specs() -> tuple[VariantSpec, ...]:
    return (
        VariantSpec(
            name="close_match",
            tone_shift_hz=0.0,
            modulation_scale=1.0,
            modulation_depth=0.36,
            stereo_width=0.78,
            background_gain_db_offset=0.0,
            lowpass_scale=1.0,
            channel_detune_hz=0.25,
            seed_offset=0,
        ),
        VariantSpec(
            name="deeper_drift",
            tone_shift_hz=-18.0,
            modulation_scale=0.78,
            modulation_depth=0.32,
            stereo_width=0.75,
            background_gain_db_offset=2.0,
            lowpass_scale=0.82,
            channel_detune_hz=0.18,
            seed_offset=11,
        ),
        VariantSpec(
            name="lighter_shimmer",
            tone_shift_hz=14.0,
            modulation_scale=1.22,
            modulation_depth=0.40,
            stereo_width=1.05,
            background_gain_db_offset=-1.5,
            lowpass_scale=1.25,
            channel_detune_hz=0.32,
            seed_offset=23,
        ),
        VariantSpec(
            name="sleepier_pulse",
            tone_shift_hz=-34.0,
            modulation_scale=0.55,
            modulation_depth=0.22,
            stereo_width=0.65,
            background_gain_db_offset=3.0,
            lowpass_scale=0.70,
            channel_detune_hz=0.10,
            seed_offset=37,
        ),
    )


def _render_variant(
    profile: ReferenceProfile,
    variant: VariantSpec,
    output_path: Path,
    duration_sec: int,
    sample_rate: int,
    chunk_size: int,
) -> VariantRenderResult:
    if not profile.tone_peaks:
        raise ValueError("Profile has no tone peaks; cannot synthesize variant")

    total_samples = int(round(duration_sec * sample_rate))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tone_frequencies = np.array([peak.frequency_hz for peak in profile.tone_peaks], dtype=np.float64)
    tone_frequencies = tone_frequencies + variant.tone_shift_hz
    tone_frequencies = np.clip(tone_frequencies, 60.0, 900.0)

    base_levels = np.array([peak.mean_level_db for peak in profile.tone_peaks], dtype=np.float64)
    relative = base_levels - np.max(base_levels)
    tone_amplitudes = np.power(10.0, relative / 20.0)
    tone_amplitudes /= np.sum(tone_amplitudes)
    low_mid_db, presence_db, air_db = _band_group_levels(profile.band_energy_db)
    low_vs_presence_db = low_mid_db - presence_db
    tone_weight = float(
        np.clip(
            0.34
            + 0.012 * low_vs_presence_db
            - 0.00008 * (profile.spectral_centroid_hz_median - 600.0),
            0.12,
            0.75,
        )
    )
    tone_amplitudes *= tone_weight

    modulation_source = (
        profile.modulation_env_hz
        if len(profile.modulation_env_hz) > 0
        else profile.modulation_hz
    )
    modulation = np.array(modulation_source, dtype=np.float64)
    if modulation.size == 0:
        modulation = np.array([8.0, 16.0], dtype=np.float64)
    modulation = np.clip(modulation * variant.modulation_scale, 0.3, 42.0)

    seed = 17 + variant.seed_offset
    rng = np.random.default_rng(seed)

    tone_count = tone_frequencies.size
    pan = rng.uniform(-1.0, 1.0, size=tone_count)
    phase_offsets = rng.uniform(-np.pi, np.pi, size=tone_count)
    detune = rng.normal(loc=0.0, scale=variant.channel_detune_hz, size=tone_count)

    mod_assignments = np.arange(tone_count) % modulation.size
    modulation_hz = modulation[mod_assignments]

    initial_state = _VariantState(
        left_phase=rng.uniform(0.0, TWO_PI, size=tone_count),
        right_phase=rng.uniform(0.0, TWO_PI, size=tone_count),
        mod_phase=rng.uniform(0.0, TWO_PI, size=tone_count),
    )

    lowpass_hz = float(np.clip(profile.inferred_lowpass_hz * variant.lowpass_scale, 900.0, 5200.0))
    if profile.rolloff_85_hz_median > 1200.0 and profile.spectral_centroid_hz_median > 900.0:
        lowpass_hz = float(np.clip(lowpass_hz * 0.78, 900.0, 5200.0))

    target_rms_linear = float(10.0 ** (profile.rms_dbfs / 20.0))

    first_state = _VariantState(
        left_phase=initial_state.left_phase.copy(),
        right_phase=initial_state.right_phase.copy(),
        mod_phase=initial_state.mod_phase.copy(),
    )
    first_filter = StereoSosFilter.create(
        make_lowpass_sos(sample_rate=sample_rate, cutoff_hz=lowpass_hz, order=2)
    )
    probe_pass = _run_variant_pass(
        total_samples=total_samples,
        sample_rate=sample_rate,
        chunk_size=chunk_size,
        state=first_state,
        tone_frequencies=tone_frequencies,
        tone_amplitudes=tone_amplitudes,
        modulation_hz=modulation_hz,
        phase_offsets=phase_offsets,
        detune=detune,
        pan=pan,
        stereo_width=variant.stereo_width,
        modulation_depth=variant.modulation_depth,
        background_mode=profile.inferred_background,
        background_gain_db_offset=variant.background_gain_db_offset,
        target_crest_factor_db=profile.crest_factor_db,
        target_window_rms_range_db=profile.window_rms_range_db,
        target_stereo_side_mid_db=profile.side_mid_db_median,
        target_stereo_corr_median=profile.stereo_corr_median,
        target_stereo_corr_std=profile.stereo_corr_std,
        target_centroid_hz=profile.spectral_centroid_hz_median,
        target_rolloff_hz=profile.rolloff_85_hz_median,
        target_band_energy_db=profile.band_energy_db,
        seed=seed,
        lowpass_filter=first_filter,
        crest_boost=1.0,
        gain=1.0,
        write_handle=None,
    )

    probe_crest = linear_to_db(probe_pass.peak_linear) - linear_to_db(probe_pass.rms_linear)
    crest_boost = float(
        np.clip(
            10.0 ** ((profile.crest_factor_db - probe_crest) / 20.0),
            1.0,
            3.2,
        )
    )

    second_state = _VariantState(
        left_phase=initial_state.left_phase.copy(),
        right_phase=initial_state.right_phase.copy(),
        mod_phase=initial_state.mod_phase.copy(),
    )
    second_filter = StereoSosFilter.create(
        make_lowpass_sos(sample_rate=sample_rate, cutoff_hz=lowpass_hz, order=2)
    )
    measure_pass = _run_variant_pass(
        total_samples=total_samples,
        sample_rate=sample_rate,
        chunk_size=chunk_size,
        state=second_state,
        tone_frequencies=tone_frequencies,
        tone_amplitudes=tone_amplitudes,
        modulation_hz=modulation_hz,
        phase_offsets=phase_offsets,
        detune=detune,
        pan=pan,
        stereo_width=variant.stereo_width,
        modulation_depth=variant.modulation_depth,
        background_mode=profile.inferred_background,
        background_gain_db_offset=variant.background_gain_db_offset,
        target_crest_factor_db=profile.crest_factor_db,
        target_window_rms_range_db=profile.window_rms_range_db,
        target_stereo_side_mid_db=profile.side_mid_db_median,
        target_stereo_corr_median=profile.stereo_corr_median,
        target_stereo_corr_std=profile.stereo_corr_std,
        target_centroid_hz=profile.spectral_centroid_hz_median,
        target_rolloff_hz=profile.rolloff_85_hz_median,
        target_band_energy_db=profile.band_energy_db,
        seed=seed,
        lowpass_filter=second_filter,
        crest_boost=crest_boost,
        gain=1.0,
        write_handle=None,
    )
    gain = target_rms_linear / max(measure_pass.rms_linear, 1e-18)

    with sf.SoundFile(
        str(output_path),
        mode="w",
        samplerate=sample_rate,
        channels=2,
        subtype="PCM_24",
        format=_format_for_path(output_path),
    ) as handle:
        third_state = _VariantState(
            left_phase=initial_state.left_phase.copy(),
            right_phase=initial_state.right_phase.copy(),
            mod_phase=initial_state.mod_phase.copy(),
        )
        third_filter = StereoSosFilter.create(
            make_lowpass_sos(sample_rate=sample_rate, cutoff_hz=lowpass_hz, order=2)
        )
        second_pass = _run_variant_pass(
            total_samples=total_samples,
            sample_rate=sample_rate,
            chunk_size=chunk_size,
            state=third_state,
            tone_frequencies=tone_frequencies,
            tone_amplitudes=tone_amplitudes,
            modulation_hz=modulation_hz,
            phase_offsets=phase_offsets,
            detune=detune,
            pan=pan,
            stereo_width=variant.stereo_width,
            modulation_depth=variant.modulation_depth,
            background_mode=profile.inferred_background,
            background_gain_db_offset=variant.background_gain_db_offset,
            target_crest_factor_db=profile.crest_factor_db,
            target_window_rms_range_db=profile.window_rms_range_db,
            target_stereo_side_mid_db=profile.side_mid_db_median,
            target_stereo_corr_median=profile.stereo_corr_median,
            target_stereo_corr_std=profile.stereo_corr_std,
            target_centroid_hz=profile.spectral_centroid_hz_median,
            target_rolloff_hz=profile.rolloff_85_hz_median,
            target_band_energy_db=profile.band_energy_db,
            seed=seed,
            lowpass_filter=third_filter,
            crest_boost=crest_boost,
            gain=gain,
            write_handle=handle,
        )

    return VariantRenderResult(
        variant=variant.name,
        output_path=output_path,
        duration_sec=float(total_samples / sample_rate),
        sample_count=total_samples,
        rms_dbfs=linear_to_db(second_pass.rms_linear),
        peak_dbfs=linear_to_db(second_pass.peak_linear),
    )


def _run_variant_pass(
    total_samples: int,
    sample_rate: int,
    chunk_size: int,
    state: _VariantState,
    tone_frequencies: np.ndarray,
    tone_amplitudes: np.ndarray,
    modulation_hz: np.ndarray,
    phase_offsets: np.ndarray,
    detune: np.ndarray,
    pan: np.ndarray,
    stereo_width: float,
    modulation_depth: float,
    background_mode: BackgroundMode,
    background_gain_db_offset: float,
    target_crest_factor_db: float,
    target_window_rms_range_db: float,
    target_stereo_side_mid_db: float,
    target_stereo_corr_median: float,
    target_stereo_corr_std: float,
    target_centroid_hz: float,
    target_rolloff_hz: float,
    target_band_energy_db: tuple[float, ...],
    seed: int,
    lowpass_filter: StereoSosFilter,
    crest_boost: float,
    gain: float,
    write_handle: sf.SoundFile | None,
) -> _PassStats:
    tone_count = tone_frequencies.size
    if tone_count == 0:
        raise ValueError("At least one tone is required")

    background = BackgroundGenerator.create(
        mode=background_mode,
        sample_rate=sample_rate,
        seed=seed,
        profile="standard",
    )
    background_gain = background.mix_gain() * db_to_linear(background_gain_db_offset)
    noise_rng = np.random.default_rng(seed + 10_001)
    side_sos = signal.butter(
        N=2,
        Wn=[450.0, 6400.0],
        btype="bandpass",
        fs=sample_rate,
        output="sos",
    )
    air_sos = signal.butter(
        N=2,
        Wn=[1800.0, 9000.0],
        btype="bandpass",
        fs=sample_rate,
        output="sos",
    )
    transient_sos = signal.butter(
        N=2,
        Wn=[900.0, 4600.0],
        btype="bandpass",
        fs=sample_rate,
        output="sos",
    )
    click_sos = signal.butter(
        N=2,
        Wn=[700.0, 5200.0],
        btype="bandpass",
        fs=sample_rate,
        output="sos",
    )
    low_mid_db, presence_db, air_db = _band_group_levels(target_band_energy_db)
    air_rel_db = air_db - low_mid_db
    presence_rel_db = presence_db - low_mid_db
    air_gain = float(
        np.clip(
            0.05
            + 0.00006 * target_centroid_hz
            + 0.004 * (air_rel_db + 12.0),
            0.015,
            0.22,
        )
    )
    negative_corr_push = float(np.clip(-target_stereo_corr_median, 0.0, 0.4))
    side_gain = float(np.clip(0.05 + target_stereo_corr_std * 0.32 + 0.08 * negative_corr_push, 0.02, 0.2))
    crest_push = float(np.clip((target_crest_factor_db - 14.0) / 8.0, 0.0, 1.5))
    transient_gain = float(np.clip((0.08 + 0.45 * crest_push) * crest_boost, 0.08, 1.1))
    transient_probability = float(np.clip(1.0 / (sample_rate * (2.6 - 1.2 * crest_push)), 1e-6, 0.03))
    click_probability = float(
        np.clip(1.0 / (sample_rate * (30.0 - 12.0 * min(crest_boost, 2.2))), 1e-7, 0.01)
    )
    click_gain = float(np.clip(0.10 * crest_boost, 0.03, 0.5))
    post_click_probability = float(
        np.clip(1.0 / (sample_rate * (24.0 - 8.0 * min(crest_boost, 2.8))), 1e-7, 0.01)
    )
    post_click_gain = float(np.clip(0.16 * crest_boost, 0.03, 0.7))
    dynamic_wobble = float(np.clip(target_window_rms_range_db / 10.0, 0.05, 1.2))
    rain_drop_gain = 0.0
    if background_mode == "rain":
        rain_drop_gain = float(np.clip(0.05 + 0.22 * (presence_rel_db + 8.0) / 16.0, 0.0, 0.35))

    sum_squares = 0.0
    sample_value_count = 0
    peak_linear = 0.0

    fade_in_samples = int(round(12.0 * sample_rate))
    fade_out_samples = int(round(20.0 * sample_rate))

    for start_sample in range(0, total_samples, chunk_size):
        length = min(chunk_size, total_samples - start_sample)
        sample_index = np.arange(1, length + 1, dtype=np.float64)

        left_increment = (TWO_PI * tone_frequencies / sample_rate)[:, None]
        right_increment = (TWO_PI * (tone_frequencies + detune) / sample_rate)[:, None]
        modulation_increment = (TWO_PI * modulation_hz / sample_rate)[:, None]

        left_phase = state.left_phase[:, None] + left_increment * sample_index[None, :]
        right_phase = state.right_phase[:, None] + right_increment * sample_index[None, :]
        mod_phase = state.mod_phase[:, None] + modulation_increment * sample_index[None, :]

        modulation_env = 1.0 + modulation_depth * np.sin(mod_phase)
        modulation_env *= 0.5 + 0.07 * dynamic_wobble * np.sin(0.27 * mod_phase + 0.7)

        left_matrix = tone_amplitudes[:, None] * modulation_env * np.sin(left_phase)
        right_matrix = tone_amplitudes[:, None] * modulation_env * np.sin(right_phase + phase_offsets[:, None])

        width = np.clip(stereo_width, 0.0, 1.3)
        left_pan_gain = 1.0 - width * 0.35 * pan
        right_pan_gain = 1.0 + width * 0.35 * pan

        left_signal = np.sum(left_matrix * left_pan_gain[:, None], axis=0)
        right_signal = np.sum(right_matrix * right_pan_gain[:, None], axis=0)

        state.left_phase = np.remainder(left_phase[:, -1], TWO_PI)
        state.right_phase = np.remainder(right_phase[:, -1], TWO_PI)
        state.mod_phase = np.remainder(mod_phase[:, -1], TWO_PI)

        background_left = background.generate_mono(length) * background_gain
        background_right = background.generate_mono(length) * background_gain
        side_noise = noise_rng.standard_normal(length)
        side_noise = signal.sosfilt(side_sos, side_noise)
        side_noise = side_noise / (_rms_linear(side_noise) + 1e-18)

        air_left = noise_rng.standard_normal(length)
        air_left = signal.sosfilt(air_sos, air_left)
        air_left = air_left / (_rms_linear(air_left) + 1e-18)

        air_right = noise_rng.standard_normal(length)
        air_right = signal.sosfilt(air_sos, air_right)
        air_right = air_right / (_rms_linear(air_right) + 1e-18)

        rain_drops = np.zeros(length, dtype=np.float64)
        if rain_drop_gain > 0.0:
            drop_probability = 1.0 / (sample_rate * 0.8)
            impulses = (noise_rng.random(length) < drop_probability).astype(np.float64)
            impulses *= noise_rng.uniform(0.7, 2.0, size=length)
            decay = float(np.exp(-1.0 / (0.012 * sample_rate)))
            rain_drops = signal.lfilter(
                b=[1.0 - decay],
                a=[1.0, -decay],
                x=impulses,
            )
            rain_drops *= rain_drop_gain
        decor_noise = noise_rng.standard_normal(length)
        decor_noise = signal.sosfilt(side_sos, decor_noise)
        decor_noise = decor_noise / (_rms_linear(decor_noise) + 1e-18)
        decor_gain = float(np.clip(0.02 + 0.9 * negative_corr_push, 0.01, 0.2))
        transients = (noise_rng.random(length) < transient_probability).astype(np.float64)
        transients *= noise_rng.uniform(0.9, 3.8, size=length)
        transients = signal.sosfilt(transient_sos, transients)
        transients *= transient_gain
        clicks = (noise_rng.random(length) < click_probability).astype(np.float64)
        clicks *= noise_rng.uniform(1.5, 5.0, size=length)
        clicks = signal.sosfilt(click_sos, clicks)
        clicks *= click_gain

        stereo = np.column_stack(
            (
                left_signal
                + background_left
                + side_gain * side_noise
                + air_gain * air_left
                + rain_drops
                + transients
                + clicks
                + decor_gain * decor_noise,
                right_signal
                + background_right
                - side_gain * side_noise
                + air_gain * air_right
                + rain_drops
                + transients
                + clicks
                - decor_gain * decor_noise,
            )
        )

        envelope = np.ones(length, dtype=np.float64)
        indices = np.arange(start_sample, start_sample + length, dtype=np.float64)
        if fade_in_samples > 0:
            mask = indices < fade_in_samples
            if np.any(mask):
                t = indices[mask] / fade_in_samples
                envelope[mask] = 0.5 * (1.0 - np.cos(np.pi * t))
        if fade_out_samples > 0:
            fade_out_start = total_samples - fade_out_samples
            mask = indices >= fade_out_start
            if np.any(mask):
                t = (indices[mask] - fade_out_start) / fade_out_samples
                envelope[mask] = np.minimum(envelope[mask], 0.5 * (1.0 + np.cos(np.pi * t)))

        stereo *= envelope[:, None]

        filtered = lowpass_filter.process(stereo)
        filtered_mid = 0.5 * (filtered[:, 0] + filtered[:, 1])
        filtered_side = 0.5 * (filtered[:, 0] - filtered[:, 1])
        side_ratio = _rms_linear(filtered_side) / max(_rms_linear(filtered_mid), 1e-18)
        target_ratio = float(10.0 ** (target_stereo_side_mid_db / 20.0))
        side_scale = float(np.clip(target_ratio / max(side_ratio, 1e-6), 0.55, 2.2))
        filtered_side *= side_scale
        filtered[:, 0] = filtered_mid + filtered_side
        filtered[:, 1] = filtered_mid - filtered_side
        post_clicks = (noise_rng.random(length) < post_click_probability).astype(np.float64)
        post_clicks *= noise_rng.uniform(2.0, 10.0, size=length)
        post_decay = float(np.exp(-1.0 / (0.0018 * sample_rate)))
        post_clicks = signal.lfilter(
            b=[1.0 - post_decay],
            a=[1.0, -post_decay],
            x=post_clicks,
        )
        post_clicks *= post_click_gain
        filtered[:, 0] += post_clicks
        filtered[:, 1] += post_clicks
        filtered *= gain
        np.clip(filtered, -1.0, 1.0, out=filtered)

        sum_squares += float(np.sum(filtered * filtered, dtype=np.float64))
        sample_value_count += filtered.size
        peak_linear = max(peak_linear, float(np.max(np.abs(filtered))))

        if write_handle is not None:
            write_handle.write(filtered.astype(np.float32, copy=False))

    return _PassStats(
        rms_linear=float(np.sqrt(sum_squares / max(sample_value_count, 1))),
        peak_linear=peak_linear,
    )


def _format_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".flac":
        return "FLAC"
    return "WAV"
