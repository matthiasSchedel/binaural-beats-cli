# Go Migration Plan: Faster Rendering with Output Quality Parity

## Goal

Rebuild the Python rendering + analysis pipeline in Go to reduce wall-clock render time and memory pressure while preserving audible quality and profile-matching behavior.

Success criteria:
- Output quality: no audible regressions on reference A/B checks.
- Numeric parity: key metrics remain within tolerance vs current Python implementation.
- Throughput target: at least 3x faster end-to-end generation on the same machine.

## Scope

In scope:
- Core synthesis engine (carrier/drone layers, modulation, background layers, filtering, transient shaping).
- Reference analysis (profile extraction fields used by `v2`: crest, modulation_env, band-energy vector, stereo stats).
- CLI equivalents for `create`, `analyze`, and variant generation.
- Multi-format export (`wav`, `flac`, `mp3`) with optional ffmpeg bridge.

Out of scope for v1 Go port:
- UI or web service layer.
- Feature expansion beyond current Python behavior.

## Constraints

- Preserve current output semantics and default presets/variants.
- Deterministic renders for the same seed/options.
- Keep streaming/block rendering model to avoid large RAM spikes.

## Architecture (Proposed)

## Packages

1. `cmd/beats-gen`
- CLI entrypoint.
- Subcommands: `create`, `analyze`, `verify`, `generate-similar-variants`.

2. `internal/profile`
- Profile structs + JSON I/O (versioned schema).
- Backward-compatible loader for older profile versions.

3. `internal/analyze`
- Windowed DSP analysis.
- Welch PSD, band energies, envelope modulation extraction.
- Stereo metrics and crest stats.

4. `internal/synth`
- Sample-accurate render engine.
- Tone layers, modulation, background generators.
- Transient/crest shaping and stereo control.

5. `internal/dsp`
- Reusable DSP primitives (biquads, SOS, envelope, RMS/peak, FFT wrappers).

6. `internal/encode`
- WAV writer (native Go).
- FLAC/MP3 strategy:
  - Option A: shell out to ffmpeg for parity and low complexity.
  - Option B: native encoders later if operationally required.

## Data model

- Keep `ReferenceProfile` JSON compatible with current Python field names.
- Add explicit `profile_version` in all generated Go profiles.
- Add strict validation at load boundaries.

## Performance Strategy

1. Block-parallel processing
- Process independent blocks in worker pools where state does not require strict serial order.
- Keep stateful filters deterministic by channel/stream ownership.

2. SIMD-friendly memory layout
- Prefer contiguous `[]float64` slices and avoid repeated allocations.
- Reuse buffers via sync.Pool in hot paths.

3. FFT optimization
- Benchmark Go FFT libraries vs cgo-backed FFTW route.
- Choose based on real profiling, not assumptions.

4. I/O streaming
- Stream reads/writes for long sessions.
- Avoid loading full tracks for analysis/render unless explicitly required.

5. Profiling-first loop
- Use `pprof` for CPU + heap at each milestone.
- Do not micro-optimize before flamegraph confirmation.

## Quality Parity Plan

Reference set (fixed):
- `The Dive CD 1-4.flac`
- `Immersion CD 0.5.flac`
- Existing generated demo outputs.

Parity metrics per artifact:
- RMS dBFS (abs error <= 0.2 dB)
- Peak dBFS (abs error <= 1.0 dB)
- Crest factor (abs error <= 1.5 dB)
- Stereo corr median/std (tight tolerance)
- Side-mid dB median
- Spectral centroid and rolloff85
- Band-energy vector per analysis band
- Envelope modulation peaks

Subjective check:
- AB listening checklist for focus, meditation, sleep use cases.

## Migration Phases

1. Baseline + fixtures
- Freeze Python outputs for known seeds/configs.
- Store compact golden metadata (not giant audio blobs).

2. Go analyzer first
- Implement `analyze` parity pipeline.
- Match profile JSON schema + tolerances.

3. Go renderer core
- Port deterministic block renderer without format conversion complexity.
- Emit WAV only initially.

4. Variant synthesis parity
- Port all variant behaviors and profile-conditioned controls.
- Validate against Python metric harness.

5. Encoding + CLI polish
- Add FLAC/MP3 output path.
- Match UX of simplified CLI.

6. Benchmark + optimize
- Profile bottlenecks; optimize in descending impact order.
- Lock final perf targets and publish benchmark table.

7. Cutover
- Run Python and Go side-by-side for a release window.
- Promote Go as default once parity gates pass.

## Benchmark Plan

Test matrix:
- 2m, 20m, 30m render durations.
- Preset generation and reference-based variant generation.
- Single-thread and default-thread modes.

Report fields:
- Wall time
- CPU time
- Peak RSS
- Output size
- Metric delta vs Python baseline

## Risk Register

1. DSP numerical drift
- Risk: tiny float differences compound into audible differences.
- Mitigation: deterministic seed handling + tolerance-gated tests.

2. Filter behavior mismatch
- Risk: Python SciPy filter behavior differs from Go implementation.
- Mitigation: impulse-response parity tests and coefficient snapshot tests.

3. Encoding inconsistencies
- Risk: native encoders produce materially different artifacts.
- Mitigation: start with ffmpeg bridge; defer native encoder swap.

4. Over-optimization early
- Risk: complexity explosion without measured gains.
- Mitigation: pprof gates before optimization changes.

## Deliverables

1. `cmd/beats-gen` Go CLI with `create` and `analyze` parity.
2. Parity test harness + golden fixtures.
3. Benchmark report (`docs/benchmarks/go-vs-python.md`).
4. Migration notes + cutover checklist.

## Immediate Next Steps

1. Create Go module skeleton and package layout.
2. Implement profile structs + JSON compatibility tests.
3. Port analyzer first and validate against current `dive_profile.json` and `immersion_profile.json`.
