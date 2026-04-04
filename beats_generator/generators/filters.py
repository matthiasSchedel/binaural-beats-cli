from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


def design_high_shelf_sos(
    sample_rate: int, center_hz: float = 8000.0, gain_db: float = 1.5, q: float = 0.7
) -> np.ndarray:
    a_gain = float(10.0 ** (gain_db / 40.0))
    omega = float(2.0 * np.pi * center_hz / sample_rate)
    sin_omega = float(np.sin(omega))
    cos_omega = float(np.cos(omega))
    alpha = sin_omega / (2.0 * q)
    sqrt_a = float(np.sqrt(a_gain))

    b0 = a_gain * ((a_gain + 1.0) + (a_gain - 1.0) * cos_omega + 2.0 * sqrt_a * alpha)
    b1 = -2.0 * a_gain * ((a_gain - 1.0) + (a_gain + 1.0) * cos_omega)
    b2 = a_gain * ((a_gain + 1.0) + (a_gain - 1.0) * cos_omega - 2.0 * sqrt_a * alpha)
    a0 = (a_gain + 1.0) - (a_gain - 1.0) * cos_omega + 2.0 * sqrt_a * alpha
    a1 = 2.0 * ((a_gain - 1.0) - (a_gain + 1.0) * cos_omega)
    a2 = (a_gain + 1.0) - (a_gain - 1.0) * cos_omega - 2.0 * sqrt_a * alpha

    b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
    a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    return signal.tf2sos(b, a).astype(np.float64, copy=False)


@dataclass
class StereoSosFilter:
    sos: np.ndarray
    zi_left: np.ndarray
    zi_right: np.ndarray

    @classmethod
    def create(cls, sos: np.ndarray) -> "StereoSosFilter":
        zi = signal.sosfilt_zi(sos).astype(np.float64, copy=False)
        return cls(
            sos=sos.astype(np.float64, copy=False),
            zi_left=np.zeros_like(zi),
            zi_right=np.zeros_like(zi),
        )

    def process(self, stereo_block: np.ndarray) -> np.ndarray:
        left, self.zi_left = signal.sosfilt(
            self.sos, stereo_block[:, 0], zi=self.zi_left
        )
        right, self.zi_right = signal.sosfilt(
            self.sos, stereo_block[:, 1], zi=self.zi_right
        )
        return np.column_stack((left, right)).astype(np.float64, copy=False)


def make_lowpass_sos(
    sample_rate: int, cutoff_hz: float = 2000.0, order: int = 4
) -> np.ndarray:
    return signal.butter(
        N=order,
        Wn=cutoff_hz,
        btype="lowpass",
        fs=sample_rate,
        output="sos",
    ).astype(np.float64, copy=False)
