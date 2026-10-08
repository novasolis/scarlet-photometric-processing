#!/usr/bin/env python3
"""
Diagnostic comparison of two background-removal methods for HD 209458 TESSCut.

Methods compared
----------------
1. Global sigma-clipped median
   - Robust single background value for the complete 50 x 50 cutout.
   - This represents the current TESS background approach.

2. Photutils Background2D
   - Spatially varying background model.
   - Default test configuration:
       box_size=(10, 10)
       filter_size=(3, 3)
       sigma_clip=SigmaClip(sigma=3.0, maxiters=10)
       bkg_estimator=MedianBackground()

The script also reports:
- TESSCut provenance (BACKAPP, FLUX_BKG)
- input FLUX statistics
- estimated background statistics for both methods
- residual-image statistics
- difference between the two background models
- values at the brightest pixel
- central/representative pixel values
- a simple quantitative comparison of how similar the two corrected
  images are

This is diagnostic only. It does not modify the production pipeline.

Target:
    HD 209458
    RA     = 330.79504167 deg
    Dec    = 18.88424167 deg
    Sector = 56
    Cutout = 50 x 50
"""

from __future__ import annotations

import sys

import numpy as np
import lightkurve as lk

from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
from lightkurve import TessTargetPixelFile


TARGET_NAME = "HD 209458"
RA_DEG = 330.79504167
DEC_DEG = 18.88424167
SECTOR = 56
CUTOUT_SIZE = 50
CADENCE_INDEX = 0

# Background2D test configuration.
BACKGROUND2D_BOX_SIZE = (10, 10)
BACKGROUND2D_FILTER_SIZE = (3, 3)
BACKGROUND2D_SIGMA = 3.0
BACKGROUND2D_MAX_ITERS = 10


def finite_values(array):
    """Return finite values as float."""
    array = np.asarray(array, dtype=float)
    return array[np.isfinite(array)]


def stats(label, array):
    """Print robust statistics for an image/array."""
    array = np.asarray(array, dtype=float)
    finite = finite_values(array)

    print(f"\n{label}")
    print("-" * len(label))

    if finite.size == 0:
        print("No finite values.")
        return

    print(f"shape          : {array.shape}")
    print(f"finite pixels  : {finite.size:,}")
    print(f"min            : {np.min(finite):.9f}")
    print(f"max            : {np.max(finite):.9f}")
    print(f"mean           : {np.mean(finite):.9f}")
    print(f"median         : {np.median(finite):.9f}")
    print(f"std            : {np.std(finite):.9f}")
    print(f"p01            : {np.percentile(finite, 1):.9f}")
    print(f"p05            : {np.percentile(finite, 5):.9f}")
    print(f"p95            : {np.percentile(finite, 95):.9f}")
    print(f"p99            : {np.percentile(finite, 99):.9f}")


def header_value(header, key):
    return header[key] if key in header else "<not present>"


def global_sigma_clipped_background(
    image,
    sigma=3.0,
    maxiters=10,
):
    """
    Estimate a single global background using sigma clipping + median.

    This intentionally mirrors the conceptual approach currently being
    evaluated for TESS: reject strong positive/negative outliers and use
    the median of the remaining image pixels.
    """
    image = np.asarray(image, dtype=float)

    finite = image[np.isfinite(image)]

    if finite.size == 0:
        raise ValueError("Image contains no finite pixels.")

    sigma_clip = SigmaClip(
        sigma=sigma,
        maxiters=maxiters,
    )

    clipped = sigma_clip(finite)

    background_value = float(np.ma.median(clipped))

    return background_value


def print_pixel_comparison(
    image,
    global_background,
    background2d_model,
):
    """Compare both methods at representative pixels."""
    rows, cols = image.shape

    brightest_index = np.nanargmax(image)
    brightest_row, brightest_col = np.unravel_index(
        brightest_index,
        image.shape,
    )

    sample_points = [
        ("top-left", 0, 0),
        ("top-right", 0, cols - 1),
        ("centre", rows // 2, cols // 2),
        ("bottom-left", rows - 1, 0),
        ("bottom-right", rows - 1, cols - 1),
        (
            "brightest pixel",
            brightest_row,
            brightest_col,
        ),
    ]

    print("\nRepresentative pixel comparison")
    print("-" * 80)
    print(
        f"{'Location':<20}"
        f"{'Input FLUX':>16}"
        f"{'Global BKG':>16}"
        f"{'B2D BKG':>16}"
        f"{'Global corr.':>16}"
        f"{'B2D corr.':>16}"
    )

    for label, row, col in sample_points:
        value = image[row, col]
        b2d = background2d_model[row, col]

        print(
            f"{label:<20}"
            f"{value:>16.6f}"
            f"{global_background:>16.6f}"
            f"{b2d:>16.6f}"
            f"{value - global_background:>16.6f}"
            f"{value - b2d:>16.6f}"
        )

    print(
        f"\nBrightest pixel location: "
        f"row={brightest_row}, col={brightest_col}"
    )


def main():
    print("=" * 80)
    print("TESS BACKGROUND METHOD COMPARISON")
    print("=" * 80)
    print(f"Target              : {TARGET_NAME}")
    print(f"RA                  : {RA_DEG}")
    print(f"Dec                 : {DEC_DEG}")
    print(f"Sector              : {SECTOR}")
    print(f"Cutout              : {CUTOUT_SIZE} x {CUTOUT_SIZE}")
    print(f"Cadence             : {CADENCE_INDEX}")
    print()
    print("Global background:")
    print("  estimator          : sigma-clipped median")
    print("  sigma              : 3.0")
    print("  max iterations     : 10")
    print()
    print("Background2D:")
    print(f"  box_size           : {BACKGROUND2D_BOX_SIZE}")
    print(f"  filter_size        : {BACKGROUND2D_FILTER_SIZE}")
    print(f"  sigma              : {BACKGROUND2D_SIGMA}")
    print(f"  max iterations     : {BACKGROUND2D_MAX_ITERS}")
    print("  estimator          : MedianBackground")

    # ------------------------------------------------------------------
    # Acquire TESSCut
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TESSCUT ACQUISITION")
    print("=" * 80)

    print("\nSearching TESSCut...")

    search = lk.search_tesscut(
        f"{RA_DEG} {DEC_DEG}",
        sector=SECTOR,
    )

    if len(search) == 0:
        raise RuntimeError("No TESSCut products were returned.")

    print(f"Search results: {len(search)}")
    print(search)

    print("\nDownloading first TESSCut result...")

    tpf = search[0].download(
        cutout_size=CUTOUT_SIZE,
    )

    if tpf is None:
        raise RuntimeError("TESSCut download returned None.")

    if not isinstance(tpf, TessTargetPixelFile):
        raise RuntimeError(
            f"Unexpected product type: {type(tpf).__name__}"
        )

    print(f"Product type : {type(tpf).__name__}")
    print(f"Sector       : {getattr(tpf, 'sector', '<unknown>')}")
    print(f"FLUX shape   : {tpf.flux.shape}")
    print(f"Cadences     : {len(tpf.time)}")

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FITS BACKGROUND PROVENANCE")
    print("=" * 80)

    found_backapp = False

    for i, hdu in enumerate(tpf.hdu):
        if "BACKAPP" in hdu.header:
            found_backapp = True
            print(
                f"BACKAPP in HDU {i}: "
                f"{hdu.header['BACKAPP']!r}"
            )

    if not found_backapp:
        print("BACKAPP was not found in the inspected headers.")

    header = tpf.hdu[1].header

    print("\nRelevant HDU 1 header values:")
    for key in (
        "BACKAPP",
        "SECTOR",
        "CAMERA",
        "CCD",
        "BJDREFI",
        "BJDREFF",
        "TSTART",
        "TSTOP",
        "DATA_REL",
    ):
        print(
            f"  {key:10s}: "
            f"{header_value(header, key)!r}"
        )

    # ------------------------------------------------------------------
    # Input image
    # ------------------------------------------------------------------
    flux = np.asarray(
        tpf.flux.value
        if hasattr(tpf.flux, "value")
        else tpf.flux,
        dtype=float,
    )

    if flux.ndim != 3:
        raise RuntimeError(
            f"Unexpected FLUX shape: {flux.shape}"
        )

    image = flux[CADENCE_INDEX]

    stats("INPUT TPF FLUX", image)

    # ------------------------------------------------------------------
    # FLUX_BKG
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ARCHIVE FLUX_BKG")
    print("=" * 80)

    try:
        value = tpf.flux_bkg

        flux_bkg = np.asarray(
            value.value
            if hasattr(value, "value")
            else value,
            dtype=float,
        )

        stats(
            "TPF FLUX_BKG",
            flux_bkg[CADENCE_INDEX],
        )

    except Exception as exc:
        print(f"tpf.flux_bkg unavailable: {exc}")

    # ------------------------------------------------------------------
    # Method 1: global sigma-clipped median
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("METHOD 1 — GLOBAL SIGMA-CLIPPED MEDIAN")
    print("=" * 80)

    global_background = global_sigma_clipped_background(
        image,
        sigma=3.0,
        maxiters=10,
    )

    global_corrected = image - global_background

    print(
        f"\nEstimated global background = "
        f"{global_background:.9f} e-/s"
    )

    stats(
        "GLOBAL-BACKGROUND-CORRECTED IMAGE",
        global_corrected,
    )

    # ------------------------------------------------------------------
    # Method 2: Background2D
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("METHOD 2 — PHOTUTILS BACKGROUND2D")
    print("=" * 80)

    sigma_clip = SigmaClip(
        sigma=BACKGROUND2D_SIGMA,
        maxiters=BACKGROUND2D_MAX_ITERS,
    )

    bkg_estimator = MedianBackground()

    print(
        f"\nRunning Background2D with "
        f"box_size={BACKGROUND2D_BOX_SIZE}, "
        f"filter_size={BACKGROUND2D_FILTER_SIZE}..."
    )

    background2d = Background2D(
        image,
        box_size=BACKGROUND2D_BOX_SIZE,
        filter_size=BACKGROUND2D_FILTER_SIZE,
        sigma_clip=sigma_clip,
        bkg_estimator=bkg_estimator,
    )

    background2d_model = np.asarray(
        background2d.background,
        dtype=float,
    )

    background2d_corrected = image - background2d_model

    stats(
        "BACKGROUND2D MODEL",
        background2d_model,
    )

    stats(
        "BACKGROUND2D-CORRECTED IMAGE",
        background2d_corrected,
    )

    # ------------------------------------------------------------------
    # Compare background models
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("BACKGROUND MODEL COMPARISON")
    print("=" * 80)

    finite = (
        np.isfinite(image)
        & np.isfinite(background2d_model)
    )

    global_model = np.full(
        image.shape,
        global_background,
        dtype=float,
    )

    difference = background2d_model - global_model

    stats(
        "Background2D - Global background",
        difference,
    )

    b2d_finite = background2d_model[finite]
    difference_finite = difference[finite]

    print("\nKey comparison values:")
    print(
        f"  Global background median : "
        f"{global_background:.9f} e-/s"
    )
    print(
        f"  Background2D median      : "
        f"{np.median(b2d_finite):.9f} e-/s"
    )
    print(
        f"  Background2D mean        : "
        f"{np.mean(b2d_finite):.9f} e-/s"
    )
    print(
        f"  B2D - Global median      : "
        f"{np.median(difference_finite):.9f} e-/s"
    )
    print(
        f"  B2D - Global mean        : "
        f"{np.mean(difference_finite):.9f} e-/s"
    )
    print(
        f"  B2D - Global min         : "
        f"{np.min(difference_finite):.9f} e-/s"
    )
    print(
        f"  B2D - Global max         : "
        f"{np.max(difference_finite):.9f} e-/s"
    )
    print(
        f"  B2D spatial std          : "
        f"{np.std(b2d_finite):.9f} e-/s"
    )

    # ------------------------------------------------------------------
    # Compare corrected images
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CORRECTED IMAGE COMPARISON")
    print("=" * 80)

    corrected_difference = (
        background2d_corrected
        - global_corrected
    )

    stats(
        "Background2D corrected - Global corrected",
        corrected_difference,
    )

    finite_corrected = corrected_difference[
        np.isfinite(corrected_difference)
    ]

    print("\nCorrected-image agreement:")
    print(
        f"  median difference : "
        f"{np.median(finite_corrected):.9f} e-/s"
    )
    print(
        f"  mean difference   : "
        f"{np.mean(finite_corrected):.9f} e-/s"
    )
    print(
        f"  RMS difference    : "
        f"{np.sqrt(np.mean(finite_corrected ** 2)):.9f} e-/s"
    )
    print(
        f"  max abs difference: "
        f"{np.max(np.abs(finite_corrected)):.9f} e-/s"
    )

    # ------------------------------------------------------------------
    # Representative pixels
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("REPRESENTATIVE PIXEL COMPARISON")
    print("=" * 80)

    print_pixel_comparison(
        image,
        global_background,
        background2d_model,
    )

    # ------------------------------------------------------------------
    # Simple source-preservation diagnostic
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("BRIGHT-SOURCE PRESERVATION DIAGNOSTIC")
    print("=" * 80)

    brightest_index = np.nanargmax(image)
    brightest_row, brightest_col = np.unravel_index(
        brightest_index,
        image.shape,
    )

    brightest_input = image[
        brightest_row,
        brightest_col,
    ]

    brightest_global = global_corrected[
        brightest_row,
        brightest_col,
    ]

    brightest_b2d = background2d_corrected[
        brightest_row,
        brightest_col,
    ]

    print(
        f"Brightest pixel        : "
        f"row={brightest_row}, col={brightest_col}"
    )
    print(
        f"Input FLUX             : "
        f"{brightest_input:.9f} e-/s"
    )
    print(
        f"After global removal   : "
        f"{brightest_global:.9f} e-/s"
    )
    print(
        f"After Background2D     : "
        f"{brightest_b2d:.9f} e-/s"
    )
    print(
        f"Global removed         : "
        f"{global_background:.9f} e-/s"
    )
    print(
        f"Background2D removed   : "
        f"{background2d_model[brightest_row, brightest_col]:.9f} e-/s"
    )

    # ------------------------------------------------------------------
    # Final interpretation
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)

    b2d_median = float(np.median(b2d_finite))
    b2d_std = float(np.std(b2d_finite))
    model_delta = b2d_median - global_background

    print(
        f"Global sigma-clipped background : "
        f"{global_background:.6f} e-/s"
    )
    print(
        f"Background2D median             : "
        f"{b2d_median:.6f} e-/s"
    )
    print(
        f"Background2D spatial std        : "
        f"{b2d_std:.6f} e-/s"
    )
    print(
        f"Median model difference         : "
        f"{model_delta:.6f} e-/s"
    )

    print("\nInterpretation guidance:")
    print(
        "1. If Background2D is nearly spatially constant and its "
        "median is close to the global estimate, the simpler global "
        "method may be sufficient."
    )
    print(
        "2. If Background2D shows a meaningful smooth spatial gradient "
        "and produces materially different corrected pixels, investigate "
        "whether that variation is real background structure before "
        "changing the production pipeline."
    )
    print(
        "3. Do not select a production method from this single cadence "
        "alone. Repeat the comparison across representative TESS "
        "targets/cadences before making the final methodological choice."
    )

    print("=" * 80)


if __name__ == "__main__":
    sys.exit(main())
