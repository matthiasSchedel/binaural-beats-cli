# Binaural Beats CLI

Generates 96kHz / 32-bit float stereo WAV binaural beat sessions with phase-continuous synthesis and procedural background layers.

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
