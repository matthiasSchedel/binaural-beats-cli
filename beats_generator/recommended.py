from __future__ import annotations

from dataclasses import dataclass

from beats_generator.presets import SessionSpec, parse_session_spec


@dataclass(frozen=True)
class RecommendedSession:
    collection_id: str
    collection_title: str
    preset_id: str
    title: str
    purpose: str
    level: str
    version: str
    output_file: str
    spec: SessionSpec


def list_recommended_sessions() -> tuple[RecommendedSession, ...]:
    sessions: list[RecommendedSession] = []
    for item in _RECOMMENDED_PAYLOADS:
        spec = parse_session_spec(preset_id=item["preset_id"], payload=item["spec_payload"])
        sessions.append(
            RecommendedSession(
                collection_id=item["collection_id"],
                collection_title=item["collection_title"],
                preset_id=item["preset_id"],
                title=item["title"],
                purpose=item["purpose"],
                level=item["level"],
                version=item["version"],
                output_file=item["output_file"],
                spec=spec,
            )
        )
    return tuple(sessions)


def get_recommended_session_by_output_file(output_file: str) -> RecommendedSession | None:
    for session in list_recommended_sessions():
        if session.output_file == output_file:
            return session
    return None


def get_recommended_session_by_preset_id(preset_id: str) -> RecommendedSession | None:
    for session in list_recommended_sessions():
        if session.preset_id == preset_id:
            return session
    return None


_RECOMMENDED_PAYLOADS: tuple[dict[str, object], ...] = (
    {
        "collection_id": "focus_ladder",
        "collection_title": "Focus Ladder",
        "preset_id": "focus_ladder_quick",
        "title": "Focus Ladder Quick",
        "purpose": "focus",
        "level": "beginner",
        "version": "quick_10m",
        "output_file": "focus_ladder_quick_10m.wav",
        "spec_payload": {
            "duration_sec": 600,
            "carrier_left_hz": 310.0,
            "carrier_right_hz": 324.0,
            "phases": [
                (0, 180, 14.0, 18.0),
                (180, 420, 18.0, 16.0),
                (420, 600, 16.0, 14.0),
            ],
            "fade_in_sec": 5,
            "fade_out_sec": 8,
            "background": "pink_noise",
        },
    },
    {
        "collection_id": "focus_ladder",
        "collection_title": "Focus Ladder",
        "preset_id": "focus_ladder_standard",
        "title": "Focus Ladder Standard",
        "purpose": "focus",
        "level": "intermediate",
        "version": "standard_20m",
        "output_file": "focus_ladder_standard_20m.wav",
        "spec_payload": {
            "duration_sec": 1200,
            "carrier_left_hz": 310.0,
            "carrier_right_hz": 326.0,
            "phases": [
                (0, 300, 14.0, 18.0),
                (300, 960, 18.0, 15.0),
                (960, 1200, 15.0, 14.0),
            ],
            "fade_in_sec": 6,
            "fade_out_sec": 10,
            "background": "pink_noise",
        },
    },
    {
        "collection_id": "focus_ladder",
        "collection_title": "Focus Ladder",
        "preset_id": "focus_ladder_deep",
        "title": "Focus Ladder Deep",
        "purpose": "focus",
        "level": "advanced",
        "version": "deep_30m",
        "output_file": "focus_ladder_deep_30m.wav",
        "spec_payload": {
            "duration_sec": 1800,
            "carrier_left_hz": 305.0,
            "carrier_right_hz": 320.0,
            "phases": [
                (0, 360, 12.0, 16.0),
                (360, 1320, 16.0, 14.0),
                (1320, 1800, 14.0, 12.0),
            ],
            "fade_in_sec": 8,
            "fade_out_sec": 12,
            "background": "pink_noise",
        },
    },
    {
        "collection_id": "nervous_system_reset",
        "collection_title": "Nervous System Reset",
        "preset_id": "reset_quick",
        "title": "Reset Quick",
        "purpose": "anxiety",
        "level": "beginner",
        "version": "quick_10m",
        "output_file": "reset_quick_10m.wav",
        "spec_payload": {
            "duration_sec": 600,
            "carrier_left_hz": 245.0,
            "carrier_right_hz": 257.0,
            "phases": [
                (0, 180, 12.0, 8.0),
                (180, 420, 8.0, 6.0),
                (420, 600, 6.0, 6.0),
            ],
            "fade_in_sec": 6,
            "fade_out_sec": 10,
            "background": "brown_noise",
        },
    },
    {
        "collection_id": "nervous_system_reset",
        "collection_title": "Nervous System Reset",
        "preset_id": "reset_standard",
        "title": "Reset Standard",
        "purpose": "anxiety",
        "level": "intermediate",
        "version": "standard_20m",
        "output_file": "reset_standard_20m.wav",
        "spec_payload": {
            "duration_sec": 1200,
            "carrier_left_hz": 242.0,
            "carrier_right_hz": 256.0,
            "phases": [
                (0, 240, 14.0, 10.0),
                (240, 840, 10.0, 5.0),
                (840, 1200, 5.0, 5.0),
            ],
            "fade_in_sec": 8,
            "fade_out_sec": 14,
            "background": "brown_noise",
        },
    },
    {
        "collection_id": "nervous_system_reset",
        "collection_title": "Nervous System Reset",
        "preset_id": "reset_deep",
        "title": "Reset Deep",
        "purpose": "anxiety",
        "level": "advanced",
        "version": "deep_30m",
        "output_file": "reset_deep_30m.wav",
        "spec_payload": {
            "duration_sec": 1800,
            "carrier_left_hz": 238.0,
            "carrier_right_hz": 250.0,
            "phases": [
                (0, 360, 12.0, 8.0),
                (360, 1320, 8.0, 4.0),
                (1320, 1800, 4.0, 4.0),
            ],
            "fade_in_sec": 10,
            "fade_out_sec": 16,
            "background": "brown_noise",
        },
    },
    {
        "collection_id": "pain_release",
        "collection_title": "Pain Release",
        "preset_id": "pain_release_quick",
        "title": "Pain Release Quick",
        "purpose": "pain",
        "level": "beginner",
        "version": "quick_10m",
        "output_file": "pain_release_quick_10m.wav",
        "spec_payload": {
            "duration_sec": 600,
            "carrier_left_hz": 248.0,
            "carrier_right_hz": 256.0,
            "phases": [
                (0, 180, 8.0, 6.0),
                (180, 420, 6.0, 5.0),
                (420, 600, 5.0, 5.0),
            ],
            "fade_in_sec": 7,
            "fade_out_sec": 12,
            "background": "rain",
        },
    },
    {
        "collection_id": "pain_release",
        "collection_title": "Pain Release",
        "preset_id": "pain_release_standard",
        "title": "Pain Release Standard",
        "purpose": "pain",
        "level": "intermediate",
        "version": "standard_20m",
        "output_file": "pain_release_standard_20m.wav",
        "spec_payload": {
            "duration_sec": 1200,
            "carrier_left_hz": 246.0,
            "carrier_right_hz": 253.0,
            "phases": [
                (0, 300, 7.0, 5.0),
                (300, 960, 5.0, 4.0),
                (960, 1200, 4.0, 4.0),
            ],
            "fade_in_sec": 8,
            "fade_out_sec": 16,
            "background": "rain",
        },
    },
    {
        "collection_id": "pain_release",
        "collection_title": "Pain Release",
        "preset_id": "pain_release_deep",
        "title": "Pain Release Deep",
        "purpose": "pain",
        "level": "advanced",
        "version": "deep_30m",
        "output_file": "pain_release_deep_30m.wav",
        "spec_payload": {
            "duration_sec": 1800,
            "carrier_left_hz": 244.0,
            "carrier_right_hz": 251.0,
            "phases": [
                (0, 360, 7.0, 5.0),
                (360, 1380, 5.0, 3.5),
                (1380, 1800, 3.5, 3.5),
            ],
            "fade_in_sec": 10,
            "fade_out_sec": 20,
            "background": "rain",
        },
    },
)
