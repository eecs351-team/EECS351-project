"""Label-independent transforms and a fixed, compact feature schema."""

import numpy as np
import pywt
from scipy.fft import dctn

WAVELET = "db4"
DWT_LEVEL = 2
DWT_MODE = "symmetric"
ENERGY_EPS = 1e-24
FEATURE_COLUMNS = (
    "fft_low_energy_ratio", "fft_mid_energy_ratio", "fft_high_energy_ratio",
    "dct_low_energy_ratio", "dct_high_energy_ratio", "dct_ac_concentration",
    "dwt_horizontal_energy_ratio", "dwt_vertical_energy_ratio",
    "dwt_diagonal_energy_ratio", "dwt_level1_energy_ratio", "dwt_level2_energy_ratio",
)


def validate_image(image: np.ndarray) -> np.ndarray:
    """Require a finite 2D grayscale signal in [0, 1]; never resize it."""
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2 or min(image.shape) < 2:
        raise ValueError("Expected a 2D grayscale image with both dimensions >= 2.")
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("Image must contain finite values in [0, 1].")
    return image


def fft_transform(image: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(validate_image(image)))


def dct_transform(image: np.ndarray) -> np.ndarray:
    return dctn(validate_image(image), type=2, norm="ortho")


def dwt_transform(image: np.ndarray) -> list:
    image = validate_image(image)
    maximum = pywt.dwtn_max_level(image.shape, WAVELET)
    if DWT_LEVEL > maximum:
        raise ValueError(
            f"{WAVELET} level {DWT_LEVEL} requires larger input; "
            f"shape {image.shape} supports at most level {maximum}."
        )
    return pywt.wavedec2(image, WAVELET, mode=DWT_MODE, level=DWT_LEVEL)


def _ratios(energies: np.ndarray) -> np.ndarray:
    total = float(np.sum(energies))
    return energies / total if total > ENERGY_EPS else np.zeros_like(energies)


def extract_features(image: np.ndarray) -> dict[str, float]:
    """Extract 11 fixed features; energy <= 1e-24 yields zero ratios."""
    image = validate_image(image)
    height, width = image.shape

    # FFT radius in cycles/pixel. DC is removed before normalization.
    power = np.abs(fft_transform(image)) ** 2
    power[height // 2, width // 2] = 0
    fy = np.fft.fftshift(np.fft.fftfreq(height))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(width))[None, :]
    radius = np.hypot(fy, fx)
    fft_ratios = _ratios(np.array([
        power[radius < 0.15].sum(),
        power[(radius >= 0.15) & (radius < 0.35)].sum(),
        power[radius >= 0.35].sum(),
    ]))

    # DCT-II basis frequency: k/(2*N) cycles/pixel on each axis.
    power = dct_transform(image) ** 2
    power[0, 0] = 0
    radius = np.hypot(np.arange(height)[:, None] / (2 * height),
                      np.arange(width)[None, :] / (2 * width))
    dct_ratios = _ratios(np.array([
        power[radius < 0.15].sum(), power[radius >= 0.15].sum(),
    ]))
    total = power.sum()
    # Inverse participation ratio of the AC energy distribution.
    concentration = float(np.sum((power / total) ** 2)) if total > ENERGY_EPS else 0.0

    # wavedec2 gives [cA2, details2, details1]; rows here run fine -> coarse.
    details = np.array([
        [np.sum(band ** 2) for band in level]
        for level in reversed(dwt_transform(image)[1:])
    ])
    orientation_ratios = _ratios(details.sum(axis=0))
    level_ratios = _ratios(details.sum(axis=1))
    values = np.concatenate((fft_ratios, dct_ratios, [concentration],
                             orientation_ratios, level_ratios))
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite DSP feature encountered.")
    return dict(zip(FEATURE_COLUMNS, map(float, values), strict=True))
