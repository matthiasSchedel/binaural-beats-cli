from __future__ import annotations

from pathlib import Path

import click

from beats_generator.generators.binaural import render_session_to_wav
from beats_generator.presets import (
    PRESETS,
    PRESET_OUTPUT_FILENAMES,
    parse_session_spec,
    get_preset_spec,
)
from beats_generator.recommended import list_recommended_sessions
from beats_generator.verify import verify_wav


@click.group()
def main() -> None:
    """Generate and verify phase-accurate binaural beat WAV files."""


@main.command(name="generate-all")
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    required=True,
)
@click.option("--chunk-size", type=int, default=1_048_576, show_default=True)
def generate_all(output_dir: Path, chunk_size: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for preset_id in PRESETS:
        output_file = output_dir / PRESET_OUTPUT_FILENAMES[preset_id]
        spec = get_preset_spec(preset_id)
        click.echo(f"\n== {preset_id} -> {output_file} ==")
        render_session_to_wav(
            session=spec,
            output_path=output_file,
            log=click.echo,
            chunk_size=chunk_size,
            seed=7,
        )


@main.command(name="generate-recommended")
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    required=True,
)
@click.option("--chunk-size", type=int, default=1_048_576, show_default=True)
def generate_recommended(output_dir: Path, chunk_size: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sessions = list_recommended_sessions()
    click.echo(f"generating {len(sessions)} recommended sessions")
    for session in sessions:
        output_file = output_dir / session.output_file
        click.echo(
            f"\n== [{session.collection_title}] {session.title} "
            f"({session.version}) -> {output_file} =="
        )
        render_session_to_wav(
            session=session.spec,
            output_path=output_file,
            log=click.echo,
            chunk_size=chunk_size,
            seed=11,
        )


@main.command(name="generate")
@click.option("--preset", type=click.Choice(list(PRESETS.keys())), required=True)
@click.option(
    "--output",
    type=click.Path(path_type=Path, file_okay=True, dir_okay=False),
    required=True,
)
@click.option("--chunk-size", type=int, default=1_048_576, show_default=True)
def generate(preset: str, output: Path, chunk_size: int) -> None:
    spec = get_preset_spec(preset)
    render_session_to_wav(
        session=spec,
        output_path=output,
        log=click.echo,
        chunk_size=chunk_size,
        seed=7,
    )


@main.command(name="generate-custom")
@click.option("--carrier-left", type=float, required=True)
@click.option("--carrier-right", type=float, required=True)
@click.option("--duration", type=int, required=True)
@click.option(
    "--phases",
    type=str,
    required=True,
    help="Comma-separated start:end:beat_start:beat_end tuples.",
)
@click.option(
    "--background",
    type=click.Choice(["rain", "pink_noise", "brown_noise", "none"]),
    required=True,
)
@click.option("--fade-in", type=float, default=8.0, show_default=True)
@click.option("--fade-out", type=float, default=12.0, show_default=True)
@click.option(
    "--output",
    type=click.Path(path_type=Path, file_okay=True, dir_okay=False),
    required=True,
)
@click.option("--chunk-size", type=int, default=1_048_576, show_default=True)
def generate_custom(
    carrier_left: float,
    carrier_right: float,
    duration: int,
    phases: str,
    background: str,
    fade_in: float,
    fade_out: float,
    output: Path,
    chunk_size: int,
) -> None:
    phase_entries = _parse_phase_entries(phases)
    spec_payload: dict[str, object] = {
        "duration_sec": duration,
        "carrier_left_hz": carrier_left,
        "carrier_right_hz": carrier_right,
        "phases": phase_entries,
        "fade_in_sec": fade_in,
        "fade_out_sec": fade_out,
        "background": background,
    }
    spec = parse_session_spec(preset_id="custom", payload=spec_payload)
    render_session_to_wav(
        session=spec,
        output_path=output,
        log=click.echo,
        chunk_size=chunk_size,
        seed=7,
    )


@main.command(name="verify")
@click.argument("wav_path", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.option(
    "--preset",
    type=str,
    required=False,
    default=None,
    help="Optional explicit preset id to validate beat range against.",
)
def verify_command(wav_path: Path, preset: str | None) -> None:
    result = verify_wav(path=wav_path, preset_id=preset)
    click.echo(f"path: {result.path}")
    click.echo(f"duration_sec: {result.duration_sec:.3f}")
    click.echo(f"rms_dbfs: {result.rms_dbfs:.3f}")
    click.echo(f"peak_dbfs: {result.peak_dbfs:.3f}")
    click.echo(f"dominant_beat_hz: {result.dominant_beat_hz:.3f}")
    click.echo(
        f"expected_beat_range_hz: [{result.expected_low_hz:.3f}, {result.expected_high_hz:.3f}]"
    )
    click.echo(f"RESULT: {'PASS' if result.passed else 'FAIL'}")
    if not result.passed:
        raise click.ClickException("Beat verification failed")


def _parse_phase_entries(raw: str) -> list[tuple[float, float, float, float]]:
    entries = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        start, end, beat_start, beat_end = item.split(":")
        entries.append(
            (float(start), float(end), float(beat_start), float(beat_end))
        )
    if not entries:
        raise click.ClickException("No phases provided")
    return entries


if __name__ == "__main__":
    main()
