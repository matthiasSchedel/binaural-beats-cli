from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BackgroundType = Literal["rain", "pink_noise", "brown_noise", "none"]


PRESETS: dict[str, dict[str, object]] = {
    "sleep_onset": {
        "duration_sec": 1800,
        "carrier_left_hz": 230.0,
        "carrier_right_hz": 236.0,
        "phases": [
            (0, 600, 10.0, 6.0),
            (600, 1500, 6.0, 3.0),
            (1500, 1800, 3.0, 3.0),
        ],
        "fade_in_sec": 8,
        "fade_out_sec": 15,
        "background": "rain",
    },
    "deep_focus": {
        "duration_sec": 1500,
        "carrier_left_hz": 340.0,
        "carrier_right_hz": 380.0,
        "phases": [(0, 1500, 40.0, 40.0)],
        "fade_in_sec": 5,
        "fade_out_sec": 10,
        "background": "pink_noise",
    },
    "anxiety_unwind": {
        "duration_sec": 1200,
        "carrier_left_hz": 260.0,
        "carrier_right_hz": 280.0,
        "phases": [
            (0, 600, 20.0, 10.0),
            (600, 1200, 10.0, 10.0),
        ],
        "fade_in_sec": 8,
        "fade_out_sec": 15,
        "background": "brown_noise",
    },
    "theta_meditation": {
        "duration_sec": 1200,
        "carrier_left_hz": 230.0,
        "carrier_right_hz": 236.0,
        "phases": [(0, 1200, 6.0, 6.0)],
        "fade_in_sec": 10,
        "fade_out_sec": 20,
        "background": "none",
    },
    "creative_flow": {
        "duration_sec": 1200,
        "carrier_left_hz": 250.0,
        "carrier_right_hz": 260.0,
        "phases": [(0, 1200, 10.0, 10.0)],
        "fade_in_sec": 8,
        "fade_out_sec": 10,
        "background": "pink_noise",
    },
    "chronic_pain": {
        "duration_sec": 1500,
        "carrier_left_hz": 250.0,
        "carrier_right_hz": 255.0,
        "phases": [(0, 1500, 5.0, 5.0)],
        "fade_in_sec": 10,
        "fade_out_sec": 20,
        "background": "rain",
    },
    "morning_activation": {
        "duration_sec": 900,
        "carrier_left_hz": 300.0,
        "carrier_right_hz": 315.0,
        "phases": [(0, 900, 15.0, 15.0)],
        "fade_in_sec": 5,
        "fade_out_sec": 8,
        "background": "none",
    },
    "deep_meditation": {
        "duration_sec": 1800,
        "carrier_left_hz": 220.0,
        "carrier_right_hz": 227.0,
        "phases": [
            (0, 300, 10.0, 7.0),
            (300, 1800, 7.0, 7.0),
        ],
        "fade_in_sec": 15,
        "fade_out_sec": 30,
        "background": "none",
    },
}

PRESET_OUTPUT_FILENAMES: dict[str, str] = {
    "sleep_onset": "sleep_onset_30m.wav",
    "deep_focus": "deep_focus_25m.wav",
    "anxiety_unwind": "anxiety_unwind_20m.wav",
    "theta_meditation": "theta_meditation_20m.wav",
    "creative_flow": "creative_flow_20m.wav",
    "chronic_pain": "chronic_pain_25m.wav",
    "morning_activation": "morning_activation_15m.wav",
    "deep_meditation": "deep_meditation_30m.wav",
}


@dataclass(frozen=True)
class PhaseSpec:
    start_sec: float
    end_sec: float
    beat_hz_start: float
    beat_hz_end: float


@dataclass(frozen=True)
class SessionSpec:
    preset_id: str
    duration_sec: int
    carrier_left_hz: float
    carrier_right_hz: float
    phases: tuple[PhaseSpec, ...]
    fade_in_sec: float
    fade_out_sec: float
    background: BackgroundType


def parse_session_spec(preset_id: str, payload: dict[str, object]) -> SessionSpec:
    duration_raw = payload.get("duration_sec")
    carrier_left_raw = payload.get("carrier_left_hz")
    carrier_right_raw = payload.get("carrier_right_hz")
    fade_in_raw = payload.get("fade_in_sec")
    fade_out_raw = payload.get("fade_out_sec")
    background_raw = payload.get("background")
    phases_raw = payload.get("phases")

    if (
        duration_raw is None
        or carrier_left_raw is None
        or carrier_right_raw is None
        or fade_in_raw is None
        or fade_out_raw is None
        or background_raw is None
        or phases_raw is None
    ):
        raise ValueError("session payload is missing required keys")
    if not isinstance(phases_raw, list):
        raise ValueError("phases must be a list of phase tuples")

    phases: list[PhaseSpec] = []
    for item in phases_raw:
        if not isinstance(item, tuple) or len(item) != 4:
            raise ValueError("phase entries must be 4-tuples: start,end,beat_start,beat_end")
        phases.append(
            PhaseSpec(
                start_sec=float(item[0]),
                end_sec=float(item[1]),
                beat_hz_start=float(item[2]),
                beat_hz_end=float(item[3]),
            )
        )

    background = str(background_raw)
    if background not in {"rain", "pink_noise", "brown_noise", "none"}:
        raise ValueError(f"invalid background mode: {background}")

    spec = SessionSpec(
        preset_id=preset_id,
        duration_sec=int(duration_raw),
        carrier_left_hz=float(carrier_left_raw),
        carrier_right_hz=float(carrier_right_raw),
        phases=tuple(phases),
        fade_in_sec=float(fade_in_raw),
        fade_out_sec=float(fade_out_raw),
        background=background,
    )
    _validate_spec(spec)
    return spec


def get_preset_spec(preset_id: str) -> SessionSpec:
    if preset_id not in PRESETS:
        raise KeyError(f"Unknown preset: {preset_id}")
    return parse_session_spec(preset_id=preset_id, payload=PRESETS[preset_id])


def _validate_spec(spec: SessionSpec) -> None:
    if spec.duration_sec <= 0:
        raise ValueError("duration_sec must be positive")
    if not spec.phases:
        raise ValueError("at least one phase is required")
    if spec.fade_in_sec < 0 or spec.fade_out_sec < 0:
        raise ValueError("fade durations must be non-negative")
    if spec.fade_in_sec + spec.fade_out_sec > spec.duration_sec:
        raise ValueError("fade_in_sec + fade_out_sec must not exceed duration_sec")

    expected_start = 0.0
    for index, phase in enumerate(spec.phases):
        if phase.start_sec != expected_start:
            raise ValueError(
                f"phase {index} starts at {phase.start_sec}s, expected {expected_start}s"
            )
        if phase.end_sec <= phase.start_sec:
            raise ValueError(f"phase {index} has non-positive duration")
        expected_start = phase.end_sec
    if abs(expected_start - float(spec.duration_sec)) > 1e-9:
        raise ValueError(
            f"phases end at {expected_start}s, expected duration {spec.duration_sec}s"
        )
