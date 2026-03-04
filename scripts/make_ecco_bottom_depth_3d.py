#!/usr/bin/env python3
"""
Create a 3D ECCO bottom-depth array (HAB helper) from an ECCO T/S NetCDF file.

This script computes, for each horizontal grid cell (j,i), the deepest "wet" depth
based on the presence of valid THETA values (not equal to _FillValue). It then
broadcasts the 2D bottom depth field to 3D (k,j,i) and saves as a NumPy .npy file.

Output:
  - ecco_bottom_depth_3d.npy  (shape: [Nz, Ny, Nx], depth in metres, positive)

Typical usage:
  python make_ecco_bottom_depth_3d.py --input /path/to/OCEAN_TEMPERATURE_SALINITY_*.nc \
                                      --output ecco_bottom_depth_3d.npy
"""

import argparse
import numpy as np
from netCDF4 import Dataset


def compute_bottom_depth_3d(file, out_path="ecco_bottom_depth_3d.npy"):
    """
    Compute a broadcast 3D bottom depth array from ECCO THETA + Z.

    Parameters
    ----------
    file : str
        Path to ECCO NetCDF containing THETA and Z.
    out_path : str
        Output .npy file path.

    Returns
    -------
    bottom_depth_3d : np.ndarray
        Array of shape (Nz, Ny, Nx), positive depth (m), NaN over land.
    """

    # === Load ECCO file ===
    with Dataset(file, "r") as nc:
        nc.set_auto_mask(False)
        nc.set_auto_scale(False)

        theta = nc.variables["THETA"][0, :, :, :]      # (Nz, Ny, Nx) first time index
        Z = -nc.variables["Z"][:]                      # make positive (depth in metres)

        fill_value = nc.variables["THETA"]._FillValue

    # === Identify wet points ===
    # wet_mask[k,j,i] == True where THETA is present (ocean)
    wet_mask = theta != fill_value

    # === Vectorised bottom depth computation ===
    # We want, for each (j,i), the maximum Z[k] over wet levels.
    # Make Z broadcastable to (Nz, Ny, Nx) via reshape.
    Z3 = Z.reshape((-1, 1, 1))

    # Mask out dry points with NaN so nanmax gives deepest wet depth
    Z_wet = np.where(wet_mask, Z3, np.nan)

    # bottom_depth[j,i] = deepest wet depth (positive), NaN where all dry
    bottom_depth = np.nanmax(Z_wet, axis=0)

    # === Broadcast to 3D ===
    Nz, Ny, Nx = theta.shape
    bottom_depth_3d = np.broadcast_to(bottom_depth, (Nz, Ny, Nx))

    # === Save ===
    np.save(out_path, bottom_depth_3d)

    return bottom_depth_3d


def main():
    parser = argparse.ArgumentParser(
        description="Compute ECCO bottom depth (3D broadcast) from THETA and Z."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to an ECCO NetCDF file containing THETA and Z.",
    )
    parser.add_argument(
        "--output",
        default="ecco_bottom_depth_3d.npy",
        help="Output .npy path (default: ecco_bottom_depth_3d.npy).",
    )

    args = parser.parse_args()
    compute_bottom_depth_3d(args.input, args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()