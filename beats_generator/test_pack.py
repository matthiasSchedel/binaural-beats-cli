from __future__ import annotations

from dataclasses import dataclass

from beats_generator.presets import SessionSpec, parse_session_spec


@dataclass(frozen=True)
class TestSession:
    preset_id: str
    title: str
    purpose: str
    output_file: str
    spec: SessionSpec


def list_test_sessions() -> tuple[TestSession, ...]:
    sessions: list[TestSession] = []
    for payload in _TEST_SESSION_PAYLOADS:
        spec_payload = payload.get("spec_payload")
        if not isinstance(spec_payload, dict):
            raise ValueError("invalid test session payload")
        spec = parse_session_spec(
            preset_id=str(payload["preset_id"]),
            payload=spec_payload,
        )
        sessions.append(
            TestSession(
                preset_id=str(payload["preset_id"]),
                title=str(payload["title"]),
                purpose=str(payload["purpose"]),
                output_file=str(payload["output_file"]),
                spec=spec,
            )
        )
    return tuple(sessions)


def get_test_session_by_output_file(output_file: str) -> TestSession | None:
    for session in list_test_sessions():
        if session.output_file == output_file:
            return session
    return None


def get_test_session_by_preset_id(preset_id: str) -> TestSession | None:
    for session in list_test_sessions():
        if session.preset_id == preset_id:
            return session
    return None


_TEST_SESSION_PAYLOADS: tuple[dict[str, object], ...] = (
    {
        "preset_id": "test_theta_glide_soft",
        "title": "Theta Glide Soft",
        "purpose": "meditation",
        "output_file": "test_theta_glide_soft_12m.wav",
        "spec_payload": {
            "duration_sec": 720,
            "carrier_left_hz": 196.0,
            "carrier_right_hz": 204.0,
            "phases": [
                (0, 240, 8.0, 6.0),
                (240, 600, 6.0, 4.5),
                (600, 720, 4.5, 4.5),
            ],
            "fade_in_sec": 14.0,
            "fade_out_sec": 22.0,
            "background": "brown_noise",
        },
    },
    {
        "preset_id": "test_sleep_delta_drift",
        "title": "Sleep Delta Drift",
        "purpose": "sleep",
        "output_file": "test_sleep_delta_drift_15m.wav",
        "spec_payload": {
            "duration_sec": 900,
            "carrier_left_hz": 184.0,
            "carrier_right_hz": 190.0,
            "phases": [
                (0, 300, 6.0, 4.0),
                (300, 900, 4.0, 3.0),
            ],
            "fade_in_sec": 18.0,
            "fade_out_sec": 30.0,
            "background": "brown_noise",
        },
    },
    {
        "preset_id": "test_focus_calm_beta",
        "title": "Focus Calm Beta",
        "purpose": "focus",
        "output_file": "test_focus_calm_beta_10m.wav",
        "spec_payload": {
            "duration_sec": 600,
            "carrier_left_hz": 220.0,
            "carrier_right_hz": 234.0,
            "phases": [
                (0, 180, 14.0, 13.0),
                (180, 480, 13.0, 12.0),
                (480, 600, 12.0, 12.0),
            ],
            "fade_in_sec": 10.0,
            "fade_out_sec": 14.0,
            "background": "pink_noise",
        },
    },
)
