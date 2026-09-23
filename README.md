# HDR Fisheye Luminance Extraction

Processes a bracketed series of 180° fisheye panorama photos (in RAW/DNG
format) into a calibrated HDR luminance map, used to evaluate photopic and
melanopic light exposure and correlated colour temperature (CCT) in indoor
spaces.

## What it does

1. Reads a bracketed exposure series of DNG files.
2. Corrects each exposure for lens vignetting and reprojects the fisheye
   image into a hemispherical projection.
3. Merges the corrected exposures into a single HDR image (Debevec method).
4. Converts the HDR image into calibrated luminance values:
   - Photopic luminance (cd/m²)
   - Melanopic luminance (cd/m²)
   - Correlated colour temperature (CCT)
5. Displays false-colour luminance and CCT maps.
6. Computes planar illuminance (left/right hemisphere and total) from the
   luminance maps.
7. Tonemaps the HDR image into a white-balanced, viewable preview image.

## Input

A folder containing one bracketed exposure series: multiple `.DNG` files of
the same 180° fisheye scene taken at different exposure times, with the
exposure time readable from each file's EXIF data.

Set the folder path in `PATH_TO_BRACKET` at the top of
`hdr_fisheye_luminance.py`, or pass it to `load_bracket_series()` /
`read_exposure_times()` directly if importing the functions.

The camera intrinsic matrix (`CAMERA_MATRIX`), distortion coefficients
(`DIST_COEFFS`), and luminance calibration factor
(`LUMINANCE_CALIBRATION_FACTOR`) are set for the camera/lens this pipeline
was calibrated for — update these if using different equipment.

## How to run

```bash
pip install -r requirements.txt
python hdr_fisheye_luminance.py
```

Or import individual functions:

```python
from hdr_fisheye_luminance import (
    load_bracket_series, read_exposure_times,
    correct_bracket_series, merge_hdr, compute_luminance_maps
)

images_rgb, filenames = load_bracket_series("path/to/bracket")
exposure_times = read_exposure_times("path/to/bracket", filenames)
corrected = correct_bracket_series(images_rgb)
hdr_image = merge_hdr(corrected, exposure_times)
photopic, melanopic = compute_luminance_maps(hdr_image, calibration_factor=7.0)
```

## Output

- False-colour photopic and melanopic luminance maps (log10 scale)
- False-colour CCT map
- Photopic and melanopic planar illuminance values (left/right/total)
- A tonemapped, white-balanced preview image of the HDR panorama

## Reference

Vignetting fall-off correction coefficients from:
https://journals.sagepub.com/doi/abs/10.1177/14771535221101557
