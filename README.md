# Binaural Beats CLI

Generates 96kHz / 32-bit float stereo WAV binaural beat sessions with phase-continuous synthesis and procedural background layers.

## 2-Minute Demo Samples

- Deep work focus (steady concentration): [`deep_work_focus_2m.mp3`](assets/audio/demos/deep_work_focus_2m.mp3)
- Wind-down recovery (post-work decompression): [`wind_down_recovery_2m.mp3`](assets/audio/demos/wind_down_recovery_2m.mp3)
- Rain meditation (breathwork / mindfulness): [`rain_meditation_2m.mp3`](assets/audio/demos/rain_meditation_2m.mp3)
- Sleep transition (pre-sleep downshift): [`sleep_transition_2m.mp3`](assets/audio/demos/sleep_transition_2m.mp3)

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Usage

```bash
# Generate all Flutter preset assets
beats-gen generate-all --output-dir ./assets/audio/

# Generate 3 recommended collections (quick/standard/deep versions)
beats-gen generate-recommended --output-dir ./assets/audio/recommended/

# Generate 3 calmer test files + FLAC + MP3 comparison encodes
beats-gen generate-tests --output-dir ./assets/audio/tests/

# Generate a single preset
beats-gen generate --preset theta_meditation --output ./assets/audio/theta_meditation_20m.wav

# Generate custom session
beats-gen generate-custom \
  --carrier-left 230 \
  --carrier-right 236 \
  --duration 600 \
  --phases "0:600:6:6" \
  --background rain \
  --output ./custom.wav

# Verify generated file
beats-gen verify ./assets/audio/theta_meditation_20m.wav

# Analyze a reference track into a profile JSON
beats-gen analyze-reference \
  --input ./reference/The\ Dive\ CD\ 1-4.flac \
  --output ./assets/audio/reference/dive_profile.json

# Generate similar variants from a profile
beats-gen generate-similar-variants \
  --profile ./assets/audio/reference/dive_profile.json \
  --output-dir ./assets/audio/reference/variants \
  --duration-sec 120
```

## Preset Output Files

- `sleep_onset_30m.wav`
- `deep_focus_25m.wav`
- `anxiety_unwind_20m.wav`
- `theta_meditation_20m.wav`
- `creative_flow_20m.wav`
- `chronic_pain_25m.wav`
- `morning_activation_15m.wav`
- `deep_meditation_30m.wav`

## Recommended Collections

1. `Focus Ladder` (focus): quick 10m, standard 20m, deep 30m
2. `Nervous System Reset` (anxiety): quick 10m, standard 20m, deep 30m
3. `Pain Release` (pain): quick 10m, standard 20m, deep 30m
