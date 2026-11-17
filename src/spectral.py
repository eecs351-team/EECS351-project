"""Fine-grained spectral features for larger images.

Task 1's band ratios sum energy over wide radial bands, which averages away
narrow spectral peaks left by generator upsampling. These features keep the
spectrum's shape instead (Durall et al. 2020; Frank et al. 2020).
"""

import numpy as np

from .dsp_features import dct_transform, extract_features, validate_image

LOG_EPS = 1e-12
DCT_BLOCK = 8
GRID = 4
GRID_FEATURES = ("fft_high_energy_ratio", "dwt_level1_energy_ratio")


def hann_2d(shape: tuple[int, int]) -> np.ndarray:
    """Separable 2D Hann window, scaled to unit mean power so spectra stay comparable."""
    window = np.outer(np.hanning(shape[0]), np.hanning(shape[1]))
    return window / np.sqrt(np.mean(window ** 2))


def power_spectrum(image: np.ndarray, window: bool = False) -> np.ndarray:
    """Centered FFT power of the mean-removed image, optionally Hann-windowed.

    The DFT treats the image as periodic, so a mismatch between opposite edges acts
    like a sharp step and leaks power along the frequency axes (a bright "+" in the
    spectrum). Tapering the edges to zero with a Hann window removes that leakage
    at the cost of slightly wider spectral peaks.
    """
    image = validate_image(image)
    signal = image - image.mean()
    if window:
        signal = signal * hann_2d(signal.shape)
    return np.abs(np.fft.fftshift(np.fft.fft2(signal))) ** 2


def log_power_spectrum(image: np.ndarray, window: bool = False) -> np.ndarray:
    """Centered log10 power of the mean-removed image (DC power is then zero)."""
    return np.log10(power_spectrum(image, window) + LOG_EPS)


def radial_spectrum(image: np.ndarray, window: bool = False) -> np.ndarray:
    """Azimuthal average of FFT power in 1-cycle-per-image rings, log10 scale.

    Ring k contains frequencies with k <= N * radius < k + 1 (radius in
    cycles/pixel), for k = 1 .. N/2 - 1; DC and the corners beyond Nyquist are
    excluded. Returns N/2 - 1 values for an N x N image. Removing the mean only
    changes DC, so the unwindowed result equals the raw-image spectrum.
    """
    image = validate_image(image)
    height, width = image.shape
    if height != width:
        raise ValueError("radial_spectrum expects a square image.")
    power = power_spectrum(image, window)
    fy = np.fft.fftshift(np.fft.fftfreq(height))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(width))[None, :]
    ring = np.floor(np.hypot(fy, fx) * height).astype(int).ravel()
    sums = np.bincount(ring, power.ravel())
    counts = np.bincount(ring)
    rings = np.arange(1, height // 2)
    return np.log10(sums[rings] / counts[rings] + LOG_EPS)


def dct_log_blocks(image: np.ndarray, block: int = DCT_BLOCK) -> np.ndarray:
    """log10 |DCT-II| averaged over block x block groups of coefficients.

    A 256x256 image gives a 32x32 map (1024 values), ordered low -> high
    frequency along each axis. Averaging keeps the dimension practical for a
    linear classifier while preserving where in the spectrum energy sits.
    """
    image = validate_image(image)
    height, width = image.shape
    if height % block or width % block:
        raise ValueError(f"Image shape {image.shape} must be divisible by {block}.")
    magnitude = np.log10(np.abs(dct_transform(image)) + LOG_EPS)
    return magnitude.reshape(height // block, block, width // block, block).mean(axis=(1, 3)).ravel()


def grid_features(image: np.ndarray, grid: int = GRID) -> np.ndarray:
    """Task 1 high-frequency features computed on each tile of a grid x grid split.

    Faces are aligned, so each tile covers roughly the same content (background,
    hair, skin, eyes) in every image. Order: tile row, tile column, feature.
    """
    image = validate_image(image)
    height, width = image.shape
    if height % grid or width % grid:
        raise ValueError(f"Image shape {image.shape} must be divisible by {grid}.")
    tiles = image.reshape(grid, height // grid, grid, width // grid).swapaxes(1, 2)
    return np.array([[extract_features(tile)[name] for name in GRID_FEATURES]
                     for row in tiles for tile in row]).ravel()


def grid_columns(grid: int = GRID) -> list[str]:
    return [f"tile_r{r}_c{c}_{name}" for r in range(grid) for c in range(grid)
            for name in GRID_FEATURES]
