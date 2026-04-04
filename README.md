# Binaural Beats CLI

General-purpose CLI for generating ambient/binaural sessions and reference-matched variants.

Outputs supported by the simple flow: `wav`, `flac`, `mp3`.

## 2-Minute Demo Samples

Dive foundation use cases:
- Focus sprint: [`dive_focus_sprint_2m.mp3`](assets/audio/demos/dive_focus_sprint_2m.mp3)
- Creative flow: [`dive_creative_flow_2m.mp3`](assets/audio/demos/dive_creative_flow_2m.mp3)
- Downshift: [`dive_downshift_2m.mp3`](assets/audio/demos/dive_downshift_2m.mp3)

Immersion foundation use cases:
- Breath meditation: [`immersion_breath_meditation_2m.mp3`](assets/audio/demos/immersion_breath_meditation_2m.mp3)
- Sleep transition: [`immersion_sleep_transition_2m.mp3`](assets/audio/demos/immersion_sleep_transition_2m.mp3)
- Anxiety reset: [`immersion_anxiety_reset_2m.mp3`](assets/audio/demos/immersion_anxiety_reset_2m.mp3)

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

If you want `flac` or `mp3` output, install `ffmpeg`.

## Quick Start (Simple CLI)

1) Generate from a built-in preset (multi-format in one command):

```bash
beats-gen create \
  --preset theta_meditation \
  --name theta_20m \
  --output-dir ./exports \
  --formats wav,flac,mp3
```

2) Analyze a reference track:

```bash
beats-gen analyze \
  --input ./reference/The\ Dive\ CD\ 1-4.flac
```

3) Generate a similar track from a saved profile:

```bash
beats-gen create \
  --profile ./assets/audio/reference/the_dive_cd_1-4_profile.json \
  --variant close_match \
  --duration-min 30 \
  --name dive_focus_30m \
  --output-dir ./exports \
  --formats flac,mp3
```

4) Generate a similar track directly from reference audio (and save profile):

```bash
beats-gen create \
  --reference ./reference/Immersion\ CD\ 0.5.flac \
  --variant deeper_drift \
  --duration-min 25 \
  --name immersion_sleep_25m \
  --output-dir ./exports \
  --formats wav,mp3 \
  --save-profile ./assets/audio/reference/immersion_profile.json
```

## `create` Command Model

Choose exactly one source:
- `--preset <preset-id>`
- `--profile <profile.json>`
- `--reference <audio-file>`

Main options:
- `--name`: output basename
- `--output-dir`: destination folder
- `--formats`: comma-separated `wav,flac,mp3`
- `--duration-min`: optional override (for profile/reference defaults to 20m when omitted)
- `--variant`: `close_match`, `deeper_drift`, `lighter_shimmer`, `sleepier_pulse`

## Advanced Commands

These remain available for deeper control:

```bash
# Legacy/explicit analysis
beats-gen analyze-reference --input <audio> --output <profile.json>

# Generate all similarity variants from profile/reference
beats-gen generate-similar-variants --profile <profile.json> --output-dir <dir>

# Classic preset/custom render paths
beats-gen generate --preset <preset-id> --output <file.wav>
beats-gen generate-custom ...

# Validation
beats-gen verify <audio.wav>
```

## Presets

- `sleep_onset`
- `deep_focus`
- `anxiety_unwind`
- `theta_meditation`
- `creative_flow`
- `chronic_pain`
- `morning_activation`
- `deep_meditation`

## Notes

- WAV is rendered first for deterministic generation; FLAC/MP3 are derived from it.
- `flac`/`mp3` require `ffmpeg` in PATH.
- 30-minute large renders in `assets/audio/reference/*_30m/` are intentionally ignored in git.
