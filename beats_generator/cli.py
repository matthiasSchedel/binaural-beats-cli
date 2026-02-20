from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import click

from beats_generator.generators.binaural import RenderConfig, render_session_to_wav
from beats_generator.presets import (
    PRESETS,
    PRESET_OUTPUT_FILENAMES,
    parse_session_spec,
    get_preset_spec,
)
from beats_generator.reference_similarity import (
    analyze_reference,
    generate_variants_from_profile,
    load_reference_profile,
    save_reference_profile,
)
from beats_generator.recommended import list_recommended_sessions
from beats_generator.test_pack import list_test_sessions
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


@main.command(name="generate-tests")
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    required=True,
)
@click.option("--chunk-size", type=int, default=1_048_576, show_default=True)
@click.option("--mp3-bitrate-kbps", type=int, default=320, show_default=True)
@click.option("--with-mp3/--no-mp3", default=True, show_default=True)
@click.option("--with-flac/--no-flac", default=True, show_default=True)
def generate_tests(
    output_dir: Path,
    chunk_size: int,
    mp3_bitrate_kbps: int,
    with_mp3: bool,
    with_flac: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_path = shutil.which("ffmpeg")
    if (with_mp3 or with_flac) and ffmpeg_path is None:
        raise click.ClickException("ffmpeg not found but encoding was requested")

    render_config = RenderConfig(
        sample_rate=48_000,
        output_subtype="PCM_24",
        target_rms_dbfs=-20.0,
        headphone_shelf_gain_db=-0.75,
        post_lowpass_hz=2200.0,
        post_lowpass_order=2,
        background_gain_db_offset=-4.0,
    )

    sessions = list_test_sessions()
    click.echo(f"generating {len(sessions)} relaxation test sessions")
    for index, session in enumerate(sessions, start=1):
        wav_path = output_dir / session.output_file
        click.echo(
            f"\n== test {index}/{len(sessions)}: {session.title} "
            f"({session.purpose}) -> {wav_path} =="
        )
        result = render_session_to_wav(
            session=session.spec,
            output_path=wav_path,
            log=click.echo,
            chunk_size=chunk_size,
            seed=30 + index,
            config=render_config,
        )
        verify_result = verify_wav(path=wav_path, preset_id=session.preset_id)
        click.echo(
            f"verify: {'PASS' if verify_result.passed else 'FAIL'} "
            f"beat={verify_result.dominant_beat_hz:.3f}Hz "
            f"rms={result.rms_dbfs:.2f}dBFS peak={result.peak_dbfs:.2f}dBFS"
        )
        if not verify_result.passed:
            raise click.ClickException(f"verification failed for {wav_path}")

        if with_flac and ffmpeg_path is not None:
            flac_path = wav_path.with_suffix(".flac")
            _encode_with_ffmpeg(
                ffmpeg_path=ffmpeg_path,
                args=[
                    "-c:a",
                    "flac",
                    "-compression_level",
                    "8",
                ],
                input_path=wav_path,
                output_path=flac_path,
            )
            click.echo(f"encoded lossless: {flac_path}")

        if with_mp3 and ffmpeg_path is not None:
            mp3_path = wav_path.with_suffix(".mp3")
            _encode_with_ffmpeg(
                ffmpeg_path=ffmpeg_path,
                args=[
                    "-c:a",
                    "libmp3lame",
                    "-b:a",
                    f"{mp3_bitrate_kbps}k",
                ],
                input_path=wav_path,
                output_path=mp3_path,
            )
            click.echo(f"encoded mp3: {mp3_path}")


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


@main.command(name="analyze-reference")
@click.option(
    "--input",
    "input_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path, file_okay=True, dir_okay=False),
    required=True,
)
@click.option("--window-sec", type=float, default=20.0, show_default=True)
@click.option("--hop-sec", type=float, default=20.0, show_default=True)
def analyze_reference_command(
    input_path: Path,
    output_path: Path,
    window_sec: float,
    hop_sec: float,
) -> None:
    profile = analyze_reference(
        input_path=input_path,
        window_sec=window_sec,
        hop_sec=hop_sec,
    )
    save_reference_profile(profile=profile, output_path=output_path)
    click.echo(f"profile saved: {output_path}")
    click.echo(
        f"duration={profile.duration_sec:.2f}s "
        f"rms={profile.rms_dbfs:.2f}dBFS peak={profile.peak_dbfs:.2f}dBFS"
    )
    click.echo(
        f"crest={profile.crest_factor_db:.2f}dB "
        f"window-rms-range={profile.window_rms_range_db:.2f}dB"
    )
    click.echo(
        f"stereo corr median={profile.stereo_corr_median:.3f} "
        f"std={profile.stereo_corr_std:.3f} side-mid={profile.side_mid_db_median:.2f}dB"
    )
    click.echo(f"inferred background={profile.inferred_background}")
    click.echo(f"inferred lowpass={profile.inferred_lowpass_hz:.1f}Hz")
    click.echo(
        "tone peaks="
        + ", ".join(f"{peak.frequency_hz:.1f}Hz" for peak in profile.tone_peaks)
    )
    click.echo(
        "modulation="
        + ", ".join(f"{value:.1f}Hz" for value in profile.modulation_hz)
    )
    click.echo(
        "modulation_env="
        + ", ".join(f"{value:.1f}Hz" for value in profile.modulation_env_hz)
    )


@main.command(name="generate-similar-variants")
@click.option(
    "--reference",
    "reference_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=False,
    default=None,
    help="Optional source audio for on-the-fly analysis.",
)
@click.option(
    "--profile",
    "profile_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=False,
    default=None,
    help="Optional precomputed profile JSON from analyze-reference.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    required=True,
)
@click.option("--duration-sec", type=int, default=None)
@click.option("--sample-rate", type=int, default=48_000, show_default=True)
@click.option("--chunk-size", type=int, default=131_072, show_default=True)
@click.option("--window-sec", type=float, default=20.0, show_default=True)
@click.option("--hop-sec", type=float, default=20.0, show_default=True)
def generate_similar_variants_command(
    reference_path: Path | None,
    profile_path: Path | None,
    output_dir: Path,
    duration_sec: int | None,
    sample_rate: int,
    chunk_size: int,
    window_sec: float,
    hop_sec: float,
) -> None:
    if reference_path is None and profile_path is None:
        raise click.ClickException("Provide --reference or --profile")
    if reference_path is not None and profile_path is not None:
        raise click.ClickException("Use either --reference or --profile, not both")

    if profile_path is not None:
        profile = load_reference_profile(input_path=profile_path)
    else:
        assert reference_path is not None
        profile = analyze_reference(
            input_path=reference_path,
            window_sec=window_sec,
            hop_sec=hop_sec,
        )

    results = generate_variants_from_profile(
        profile=profile,
        output_dir=output_dir,
        duration_sec=duration_sec,
        sample_rate=sample_rate,
        chunk_size=chunk_size,
    )
    for result in results:
        click.echo(
            f"{result.variant}: {result.output_path} "
            f"dur={result.duration_sec:.2f}s rms={result.rms_dbfs:.2f}dBFS "
            f"peak={result.peak_dbfs:.2f}dBFS"
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


def _encode_with_ffmpeg(
    ffmpeg_path: str, args: list[str], input_path: Path, output_path: Path
) -> None:
    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        *args,
        str(output_path),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "ffmpeg failed"
        raise click.ClickException(message)


if __name__ == "__main__":
    main()
