from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import signal

from .filters import make_lowpass_sos

BackgroundType = Literal["rain", "pink_noise", "brown_noise", "none"]

BACKGROUND_LEVEL_DB: dict[BackgroundType, float] = {
    "rain": -14.0,
    "pink_noise": -16.0,
    "brown_noise": -18.0,
    "none": -120.0,
}


def db_to_linear(db: float) -> float:
    return float(10.0 ** (db / 20.0))


@dataclass
class BackgroundGenerator:
    mode: BackgroundType
    sample_rate: int
    rng: np.random.Generator
    rain_sos: np.ndarray | None = None
    rain_zi: np.ndarray | None = None
    rain_drop_zi: np.ndarray | None = None
    brown_state: float = 0.0

    @classmethod
    def create(
        cls, mode: BackgroundType, sample_rate: int, seed: int
    ) -> "BackgroundGenerator":
        rain_sos: np.ndarray | None = None
        rain_zi: np.ndarray | None = None
        rain_drop_zi: np.ndarray | None = None
        if mode == "rain":
            rain_sos = make_lowpass_sos(sample_rate=sample_rate, cutoff_hz=2000.0, order=4)
            rain_zi = np.zeros_like(signal.sosfilt_zi(rain_sos))
            rain_drop_zi = np.zeros(1, dtype=np.float64)
        return cls(
            mode=mode,
            sample_rate=sample_rate,
            rng=np.random.default_rng(seed),
            rain_sos=rain_sos,
            rain_zi=rain_zi,
            rain_drop_zi=rain_drop_zi,
            brown_state=0.0,
        )

    def mix_gain(self) -> float:
        return db_to_linear(BACKGROUND_LEVEL_DB[self.mode])

    def generate_mono(self, num_samples: int) -> np.ndarray:
        if self.mode == "none":
            return np.zeros(num_samples, dtype=np.float64)
        if self.mode == "pink_noise":
            return _normalize_rms(self._generate_pink(num_samples))
        if self.mode == "brown_noise":
            return _normalize_rms(self._generate_brown(num_samples))
        if self.mode == "rain":
            return _normalize_rms(self._generate_rain(num_samples))
        raise ValueError(f"Unsupported background mode: {self.mode}")

    def _generate_pink(self, num_samples: int) -> np.ndarray:
        white = self.rng.standard_normal(num_samples, dtype=np.float64)
        spectrum = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(num_samples, d=1.0 / self.sample_rate)
        shaping = np.ones_like(freqs, dtype=np.float64)
        shaping[1:] = 1.0 / np.sqrt(freqs[1:])
        pink = np.fft.irfft(spectrum * shaping, n=num_samples)
        return pink.astype(np.float64, copy=False)

    def _generate_brown(self, num_samples: int) -> np.ndarray:
        white = self.rng.standard_normal(num_samples, dtype=np.float64)
        brown = np.cumsum(white, dtype=np.float64)
        brown += self.brown_state
        self.brown_state = float(brown[-1])
        brown -= np.mean(brown)
        return brown

    def _generate_rain(self, num_samples: int) -> np.ndarray:
        assert self.rain_sos is not None
        assert self.rain_zi is not None
        assert self.rain_drop_zi is not None

        white = self.rng.standard_normal(num_samples, dtype=np.float64)
        rain_noise, self.rain_zi = signal.sosfilt(self.rain_sos, white, zi=self.rain_zi)

        drop_probability = 1.0 / (self.sample_rate * 0.18)
        impulses = (
            self.rng.random(num_samples) < drop_probability
        ).astype(np.float64, copy=False)
        impulses *= self.rng.uniform(0.35, 1.0, size=num_samples)

        decay = float(np.exp(-1.0 / (0.035 * self.sample_rate)))
        drops, self.rain_drop_zi = signal.lfilter(
            b=[1.0 - decay],
            a=[1.0, -decay],
            x=impulses,
            zi=self.rain_drop_zi,
        )
        return rain_noise + 0.45 * drops


def _normalize_rms(signal_mono: np.ndarray) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(signal_mono)) + 1e-18))
    return signal_mono / rms
