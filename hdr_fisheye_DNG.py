# -*- coding: utf-8 -*-
"""
HDR Merging and Luminance Extraction of 180° Fisheye Panoramas in RAW (DNG) Format

Pipeline:
1. Import bracketed DNG exposures
2. Correct vignetting and fisheye-to-hemispherical projection
3. Merge exposures into an HDR image (Debevec method)
4. Extract calibrated luminance values (photopic, melanopic, CCT)
5. Display false-colour luminance / CCT maps
6. Compute planar illuminance
7. Tonemap for a viewable preview image

Requirements: opencv-python, rawpy==0.21, exifread, numpy, matplotlib, Pillow
"""

import os
from typing import Tuple

import cv2
import numpy as np
import rawpy
import exifread
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------------------------------------------------------
# 1. Import DNG files
# ---------------------------------------------------------------------------

# Path to the folder containing one bracketed exposure series (.DNG files)
PATH_TO_BRACKET = "your_path"


def read_dng_to_bayer_image(path_to_dng_image: str) -> np.ndarray:
    """Read a DNG file and return its raw (Bayer) image matrix."""
    return rawpy.imread(path_to_dng_image).raw_image


def bayer_to_rgb(bayer_img) -> np.ndarray:
    """Convert a rawpy RAW object (Bayer BGGR/RGGB) into an RGB image."""
    return cv2.cvtColor(bayer_img.raw_image, cv2.COLOR_BAYER_RGGB2RGB)


def load_bracket_series(path_to_bracket: str) -> Tuple[list, list]:
    """Load every DNG file in a folder and return (rgb_images, filenames)."""
    dng_images_list = sorted(
        f for f in os.listdir(path_to_bracket) if f.endswith(".DNG")
    )

    images_rgb = []
    for filename in dng_images_list:
        path_to_dng_image = os.path.join(path_to_bracket, filename)
        raw_image = rawpy.imread(path_to_dng_image)
        images_rgb.append(bayer_to_rgb(raw_image))

    return images_rgb, dng_images_list


def read_exposure_times(path_to_bracket: str, dng_images_list: list) -> np.ndarray:
    """Read the EXIF exposure time (seconds) for each DNG file, in order."""
    exposure_times = []
    for filename in dng_images_list:
        path_to_dng_image = os.path.join(path_to_bracket, filename)
        with open(path_to_dng_image, "rb") as file:
            tags = exifread.process_file(file)
            exposure_times.append(tags["EXIF ExposureTime"].values[0].decimal())
    return np.array(exposure_times, dtype=np.float32)


# ---------------------------------------------------------------------------
# 2. Correction: vignetting + fisheye-to-hemispherical projection
# ---------------------------------------------------------------------------

def split_left_right(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split a dual-fisheye panorama into its left and right hemispheres."""
    middle_index = image.shape[1] // 2
    return image[:, :middle_index], image[:, middle_index:]


def merge_left_right(images: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Merge a (left, right) pair of hemispheres back into one panorama."""
    return np.hstack((images[0], images[1]))


def create_circular_mask(h: int, w: int, d: int) -> np.ndarray:
    """Create a circular mask matching a fisheye lens' circular image area."""
    center = (int(w / 2), int(h / 2))
    radius = min(center[0], center[1], w - center[0], h - center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= radius
    return mask if d == 1 else np.dstack([mask] * d)


def split_and_mask(image: np.ndarray) -> np.ndarray:
    """Apply the circular fisheye mask to each hemisphere (outside = 0)."""
    left_image, right_image = split_left_right(image)
    h, w = left_image.shape[:2]
    d = left_image.shape[2] if left_image.ndim == 3 else 1
    mask = create_circular_mask(h, w, d)
    return merge_left_right((left_image * mask, right_image * mask))


def split_and_mask_na(image: np.ndarray) -> np.ndarray:
    """Same as split_and_mask, but sets values outside the circle to NaN."""
    left_image, right_image = split_left_right(image)
    h, w = left_image.shape[:2]
    d = left_image.shape[2] if left_image.ndim == 3 else 1
    mask = create_circular_mask(h, w, d)
    left_masked = np.where(mask, left_image, np.nan)
    right_masked = np.where(mask, right_image, np.nan)
    return merge_left_right((left_masked, right_masked))


def apply_radial_falloff(image: np.ndarray) -> np.ndarray:
    """
    Correct lens vignetting with a radial fall-off polynomial.
    Coefficients from: https://journals.sagepub.com/doi/abs/10.1177/14771535221101557
    """
    size = image.shape
    center = np.array(size[:2]) // 2
    x, y = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
    distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    distance = distance / distance[0, 0]

    def vignetting_curve(d):
        return -0.1225 * d ** 3 + 0.0354 * d ** 2 - 0.0098 * d + 0.9995

    falloff = 1 / vignetting_curve(distance)
    return image * np.dstack((falloff, falloff, falloff))


def adjust_fisheye_to_hemispherical(
    image: np.ndarray, K: np.ndarray, D: np.ndarray, balance: float = 1.0
) -> np.ndarray:
    """
    Undistort a fisheye image into a hemispherical projection.

    K: camera intrinsic matrix
    D: fisheye distortion coefficients
    balance: 0-1, how much of the original fisheye field of view to retain
    """
    h, w = image.shape[:2]
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=balance
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
    )
    return cv2.remap(
        image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )


# Camera calibration (intrinsic matrix and fisheye distortion coefficients)
CAMERA_MATRIX = np.array(
    [
        [1972.38274, 0.0, 1822.0],
        [0.0, 1972.29672, 1822.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
DIST_COEFFS = np.array(
    [[-1.49788381e-03, -2.81638299e-03, 2.71872527e-05, 6.09923614e-05]],
    dtype=np.float32,
)


def correct_bracket_series(
    images_rgb: list, K: np.ndarray = CAMERA_MATRIX, D: np.ndarray = DIST_COEFFS
) -> list:
    """Apply vignetting + hemispherical correction to every exposure."""
    corrected = []
    for image in images_rgb:
        left_image, right_image = split_left_right(image)
        left_corrected = adjust_fisheye_to_hemispherical(
            apply_radial_falloff(left_image), K, D
        )
        right_corrected = adjust_fisheye_to_hemispherical(
            apply_radial_falloff(right_image), K, D
        )
        image_corrected = merge_left_right((left_corrected, right_corrected))
        image_rgb_corrected = split_and_mask(image_corrected)

        # 16-bit -> 8-bit range reduction for HDR merging input
        corrected.append(np.clip(image_rgb_corrected / 64, 0, 255).astype(np.uint8))
    return corrected


# ---------------------------------------------------------------------------
# 3. HDR merging
# ---------------------------------------------------------------------------

def merge_hdr(corrected_images: list, exposure_times: np.ndarray) -> np.ndarray:
    """Merge a bracketed, corrected exposure series into one HDR image."""
    merge_debevec = cv2.createMergeDebevec()
    return split_and_mask(
        merge_debevec.process(corrected_images, times=exposure_times)
    )


# ---------------------------------------------------------------------------
# 4. Extraction of real luminance values (photopic, melanopic, CCT)
# ---------------------------------------------------------------------------

RGB_TO_XYZ_MATRIX = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)


def rgb_to_xyz(rgb_image: np.ndarray) -> np.ndarray:
    return rgb_image @ RGB_TO_XYZ_MATRIX.T


def xyz_to_xy(xyz_image: np.ndarray) -> np.ndarray:
    """Convert CIE XYZ to chromaticity coordinates (x, y)."""
    X, Y, Z = xyz_image[:, :, 0], xyz_image[:, :, 1], xyz_image[:, :, 2]
    sum_xyz = X + Y + Z + 1e-8  # avoid division by zero
    return np.dstack((X / sum_xyz, Y / sum_xyz))


def xy_to_cct(xy_image: np.ndarray) -> np.ndarray:
    """Estimate correlated colour temperature (CCT) using McCamy's formula."""
    x, y = xy_image[:, :, 0], xy_image[:, :, 1]
    n = (x - 0.3320) / (0.1858 - y)
    return (437 * n ** 3) + (3601 * n ** 2) + (6861 * n) + 5517


def compute_cct(rgb_image: np.ndarray) -> np.ndarray:
    return xy_to_cct(xyz_to_xy(rgb_to_xyz(rgb_image)))


# Radiometric-to-photometric calibration factor for this camera/exposure setup
LUMINANCE_CALIBRATION_FACTOR = 7.0  # change according to your calibration


def compute_luminance_maps(hdr_image: np.ndarray, calibration_factor: float):
    """Return (photopic, melanopic) luminance maps in cd/m^2."""
    R, G, B = [hdr_image[:, :, i] for i in range(3)]
    photopic = (0.2126729 * R + 0.7151522 * G + 0.0721750 * B) / calibration_factor
    melanopic = (0.0013 * R + 0.3812 * G + 0.6175 * B) / calibration_factor
    return photopic, melanopic


# ---------------------------------------------------------------------------
# 5. Display luminance / CCT maps
# ---------------------------------------------------------------------------

def display_luminance_map(matrix: np.ndarray, luminance_type: str = "Photopic", log10: bool = True):
    """Display a false-colour luminance map (expects log10-scaled input)."""
    minimum_value = np.nanmin(split_and_mask_na(matrix))
    plt.figure(figsize=(16, 7))
    cax = plt.imshow(matrix, cmap="jet", vmin=minimum_value, vmax=np.log10(3000.0))
    cbar = plt.colorbar(cax)
    cbar.set_label(f"{luminance_type} luminance")

    ticks = np.linspace(minimum_value, np.log10(3000.0), num=10)
    tick_labels = (
        [f"{10 ** t:.2f}" for t in ticks] if log10 else [f"{t:.2f}" for t in ticks]
    )
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(tick_labels)
    plt.show()


def display_cct(matrix: np.ndarray, cct_min: float = 2000, cct_max: float = 50000):
    """Display a false-colour CCT map; values outside [cct_min, cct_max] are masked."""
    masked_matrix = np.where((matrix < cct_min) | (matrix > cct_max), np.nan, matrix)
    plt.figure(figsize=(16, 7))
    im = plt.imshow(masked_matrix, cmap="bwr_r", vmin=2000, vmax=10000)
    cbar = plt.colorbar(im)
    cbar.set_label("CCT value")
    plt.show()


# ---------------------------------------------------------------------------
# 6. Planar illuminance
# ---------------------------------------------------------------------------

def calculate_illuminance(luminance_map: np.ndarray, fov: float = 180) -> float:
    """
    Compute planar illuminance from a fisheye luminance map by integrating
    luminance over the hemisphere's solid angle, weighted by cos(theta).

    luminance_map: 2D luminance values for one hemisphere
    fov: field of view of the fisheye lens in degrees
    """
    height, width = luminance_map.shape
    y, x = np.indices((height, width))
    cx, cy = width / 2, height / 2

    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_norm = r / min(cx, cy)
    theta_rad = np.deg2rad(r_norm * (fov / 2))

    solid_angle = (2 * np.pi * (1 - np.cos(np.deg2rad(fov / 2)))) / (height * width)
    cos_theta = np.cos(theta_rad)

    return np.sum(luminance_map * solid_angle * cos_theta)


def calculate_total_illuminance(luminance_map: np.ndarray, fov: float = 180) -> dict:
    """Compute left/right/total illuminance for a full dual-fisheye map."""
    left, right = split_left_right(luminance_map)
    left_ill = calculate_illuminance(left, fov)
    right_ill = calculate_illuminance(right, fov)
    return {"left": left_ill, "right": right_ill, "total": left_ill + right_ill}


# ---------------------------------------------------------------------------
# 7. Tonemapping
# ---------------------------------------------------------------------------

def tonemap_hdr(hdr_image: np.ndarray) -> np.ndarray:
    """Tonemap an HDR image (Reinhard operator) to an 8-bit preview image."""
    tonemap = cv2.createTonemapReinhard(
        gamma=1.0, intensity=0.0, color_adapt=0.0, light_adapt=0.0
    )
    tonemapped = tonemap.process(hdr_image.copy())
    return np.clip(tonemapped * 255, 0, 255).astype(np.uint8)


def balance_white(image_array: np.ndarray, p: float = 0.5) -> np.ndarray:
    """
    Simple grey-world white balance.
    p: balancing strength, 0 = no correction, 1 = full correction to green channel.
    """
    avg_r = np.mean(image_array[:, :, 0])
    avg_g = np.mean(image_array[:, :, 1])
    avg_b = np.mean(image_array[:, :, 2])

    scale_r = avg_g / avg_r * p + (1 - p)
    scale_b = avg_g / avg_b * p + (1 - p)

    balanced = image_array.copy()
    balanced[:, :, 0] = np.clip(image_array[:, :, 0] * scale_r, 0, 255)
    balanced[:, :, 2] = np.clip(image_array[:, :, 2] * scale_b, 0, 255)
    return balanced.astype(np.uint8)


# ---------------------------------------------------------------------------
# Example end-to-end usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    images_rgb, dng_images_list = load_bracket_series(PATH_TO_BRACKET)
    exposure_times = read_exposure_times(PATH_TO_BRACKET, dng_images_list)

    corrected_images = correct_bracket_series(images_rgb)
    hdr_image = merge_hdr(corrected_images, exposure_times)

    photopic_luminance, melanopic_luminance = compute_luminance_maps(
        hdr_image, LUMINANCE_CALIBRATION_FACTOR
    )
    cct = compute_cct(hdr_image)

    display_luminance_map(np.log10(photopic_luminance), "Photopic")
    display_luminance_map(np.log10(melanopic_luminance), "Melanopic")
    display_cct(split_and_mask_na(cct))

    photopic_illuminance = calculate_total_illuminance(photopic_luminance)
    melanopic_illuminance = calculate_total_illuminance(melanopic_luminance)
    print("Photopic illuminance:", photopic_illuminance)
    print("Melanopic illuminance:", melanopic_illuminance)

    preview_image = tonemap_hdr(hdr_image)
    preview_image = balance_white(preview_image, p=0.5)
    Image.fromarray(preview_image).resize((1250, 625)).show()
