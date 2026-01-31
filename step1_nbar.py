#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1: Compute NBAR reflectance from Sentinel-2 BRDF magnitude images

This script normalizes BRDF magnitude images to NBAR (Nadir BRDF-Adjusted Reflectance)
using either observed or modeled solar zenith angles.

Author: Hongxiao Jin
"""

import argparse
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

from utils import (
    Config, setup_logger, find_files, yyyymmdd_to_doy,
    latitude_array_for_window, Ra_normalize
)


def process_magnitude_to_nbar(mag_path, out_path, doy, sza_mode, sza_vrt, sza_scale, config, logger):
    """
    Process single magnitude file to NBAR
    
    Parameters:
    -----------
    mag_path : Path
        Input magnitude file
    out_path : Path
        Output NBAR file
    doy : int
        Day of year
    sza_mode : str
        'sza11' or 'sza-obs'
    sza_vrt : WarpedVRT or None
        SZA virtual raster (for sza-obs mode)
    sza_scale : float
        SZA scaling factor
    config : Config
        Configuration object
    logger : logging.Logger
        Logger instance
    """
    block_size = config.get('processing', 'block_size')
    wavelengths = config.get('wavelengths')
    out_dtype = np.int16
    out_nodata = config.get('nodata', 'int16')
    
    # Band order in magnitude files
    band_order = [
        ('b02', 'm'), ('b02', 'covm'),
        ('b03', 'm'), ('b03', 'covm'),
        ('b04', 'm'), ('b04', 'covm'),
        ('b08', 'm'), ('b08', 'covm'),
    ]
    
    band_names = [
        "nbar_b02", "cov_nbar_b02",
        "nbar_b03", "cov_nbar_b03",
        "nbar_b04", "cov_nbar_b04",
        "nbar_b08", "cov_nbar_b08",
    ]
    
    with rasterio.open(mag_path) as src:
        # Output profile
        prof = src.profile.copy()
        prof.update(
            dtype='int16',
            count=8,
            nodata=out_nodata,
            compress=config.get('processing', 'compress')
        )
        
        with rasterio.open(out_path, 'w', **prof) as dst:
            # Set metadata
            for i, name in enumerate(band_names, start=1):
                dst.set_band_description(i, name)
                dst.update_tags(
                    i,
                    SCALE_FACTOR="1e-4",
                    ADD_OFFSET="0.0",
                    UNITS="reflectance"
                )
            
            dst.update_tags(
                SZA_MODE=sza_mode,
                VALUE_ENCODING="int16_scaled_1e4",
                NODATA_VALUE=str(out_nodata)
            )
            
            # Process blocks
            for row in range(0, src.height, block_size):
                for col in range(0, src.width, block_size):
                    h = min(block_size, src.height - row)
                    w = min(block_size, src.width - col)
                    window = Window(col, row, w, h)
                    
                    # Read magnitude data
                    mag = src.read(window=window).astype(np.float32)
                    
                    # Get SZA for this block
                    if sza_mode == 'sza11':
                        lat = latitude_array_for_window(src, window)
                        from utils import solar_zenith_angle
                        sza = solar_zenith_angle(doy, lat, hour=11)
                    else:
                        sza = sza_vrt.read(1, window=window).astype(np.float32) * sza_scale
                    
                    if sza.shape != (h, w):
                        sza = sza.reshape(h, w)
                    
                    # Apply BRDF normalization
                    out_f = np.empty_like(mag, dtype=np.float32)
                    
                    for b, (band, kind) in enumerate(band_order):
                        Ra = Ra_normalize(sza, wavelengths[band]).reshape(h, w).astype(np.float32)
                        out_f[b] = mag[b] * Ra if kind == 'm' else mag[b] * (Ra * Ra)
                    
                    # Convert to int16
                    out_i = np.full(out_f.shape, out_nodata, dtype=out_dtype)
                    valid = np.isfinite(out_f)
                    out_i[valid] = np.clip(
                        np.rint(out_f[valid]),
                        np.iinfo(np.int16).min,
                        np.iinfo(np.int16).max
                    ).astype(out_dtype)
                    
                    dst.write(out_i, window=window)
    
    logger.info(f"Saved: {out_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Step 1: Compute NBAR from magnitude images"
    )
    parser.add_argument('--config', type=str, help='Configuration JSON file')
    parser.add_argument('--folder-mag', type=str, help='Magnitude folder (overrides config)')
    parser.add_argument('--out-dir', type=str, help='Output directory (overrides config)')
    parser.add_argument('--sza', type=str, choices=['sza11', 'sza-obs'], help='SZA mode (overrides config)')
    parser.add_argument('--log-file', type=str, help='Log file path')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config) if args.config else Config()
    
    # Setup logging
    log_file = args.log_file or f"logs/step1_nbar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logger('Step1_NBAR', log_file)
    
    logger.info("=" * 70)
    logger.info("STEP 1: NBAR CALCULATION FROM MAGNITUDE")
    logger.info("=" * 70)
    
    # Get parameters
    mag_dir = Path(args.folder_mag) if args.folder_mag else Path(config.get('paths', 'magnitude_dir'))
    out_dir = Path(args.out_dir) if args.out_dir else Path(config.get('paths', 'nbar_dir'))
    sza_mode = args.sza or config.get('sza', 'mode')
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Find magnitude files
    mag_files = find_files(mag_dir, "magnitude_*.tif*")
    logger.info(f"Found {len(mag_files)} magnitude files")
    
    if not mag_files:
        logger.error("No magnitude files found!")
        return
    
    # Setup SZA if needed
    sza_ds = None
    sza_scale = 1.0
    
    if sza_mode == 'sza-obs':
        sza_path = config.get('sza', 'sza_file')
        if not sza_path:
            # Search for SZA file
            sza_files = list(mag_dir.glob("*_SZA_10M*.tif*"))
            if sza_files:
                sza_path = sza_files[0]
            else:
                logger.error("No SZA file found for sza-obs mode!")
                return
        
        sza_ds = rasterio.open(sza_path)
        sza_scale = sza_ds.scales[0] if sza_ds.scales else 1.0
        logger.info(f"Using observed SZA: {sza_path}")
    
    # Process each file
    start_time = time.time()
    
    for i, (date, mag_path) in enumerate(mag_files, 1):
        doy = yyyymmdd_to_doy(date)
        out_path = out_dir / f"nbar_{date}.tiff"
        
        logger.info(f"[{i}/{len(mag_files)}] Processing {date} (DOY={doy})")
        
        # Create VRT if using observed SZA
        sza_vrt = None
        if sza_mode == 'sza-obs':
            with rasterio.open(mag_path) as src:
                sza_vrt = WarpedVRT(
                    sza_ds,
                    crs=src.crs,
                    transform=src.transform,
                    width=src.width,
                    height=src.height,
                    resampling=rasterio.enums.Resampling.bilinear
                )
        
        try:
            process_magnitude_to_nbar(
                mag_path, out_path, doy, sza_mode, 
                sza_vrt, sza_scale, config, logger
            )
        except Exception as e:
            logger.error(f"Error processing {date}: {e}", exc_info=True)
            continue
    
    if sza_ds:
        sza_ds.close()
    
    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info(f"COMPLETED in {elapsed/60:.2f} minutes")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
