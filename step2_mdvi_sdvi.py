#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2: Compute MDVI (per-pixel) and SDVI (20x20 blocks) from NBAR

This script calculates Maximum DVI (MDVI) at 95th percentile per pixel
and Soil DVI (SDVI) at 5th percentile per 20x20 block.

Author: Hongxiao Jin
"""

import argparse
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from osgeo import gdal

from utils import (
    Config, setup_logger, find_files, safe_open, create_geotiff,
    validate_raster_size, validate_date_range
)

gdal.UseExceptions()


def process_block_row(by, files_paths, xsize, ysize, config):
    """
    Process one block-row for MDVI and SDVI calculation
    
    Parameters:
    -----------
    by : int
        Block row index (0-19)
    files_paths : list of Path
        All NBAR file paths
    xsize, ysize : int
        Raster dimensions
    config : Config
        Configuration object
    
    Returns:
    --------
    by, mdvi_row, sdvi_row : int, np.ndarray, np.ndarray
    """
    block = config.get('processing', 'block_size')
    nblock = config.get('processing', 'n_blocks')
    b_red = config.get('bands', 'red')
    b_nir = config.get('bands', 'nir')
    nodata = config.get('nodata', 'int16')
    scale_in = config.get('scales', 'reflectance_in')
    scale_out = config.get('scales', 'mdvi')
    mdvi_q = config.get('mdvi_sdvi', 'mdvi_quantile')
    sdvi_q = config.get('mdvi_sdvi', 'sdvi_quantile')
    
    y0 = by * block
    h = min(block, ysize - y0)
    
    mdvi_row = np.full((h, xsize), nodata, dtype=np.int16)
    sdvi_row = np.full((nblock,), nodata, dtype=np.int16)
    
    # Histogram bins for SDVI
    dvi_bins = int(scale_out) + 1
    
    for bx in range(nblock):
        x0 = bx * block
        w = min(block, xsize - x0)
        
        stack = []
        hist = np.zeros(dvi_bins, dtype=np.int64)
        
        for fpath in files_paths:
            ds = gdal.Open(str(fpath), gdal.GA_ReadOnly)
            red = ds.GetRasterBand(b_red).ReadAsArray(x0, y0, w, h)
            nir = ds.GetRasterBand(b_nir).ReadAsArray(x0, y0, w, h)
            ds = None
            
            valid = (red != nodata) & (nir != nodata)
            if not np.any(valid):
                continue
            
            # Scale to reflectance [0..1]
            red_f = np.clip(red.astype(np.float32) * scale_in, 0.0, 1.0)
            nir_f = np.clip(nir.astype(np.float32) * scale_in, 0.0, 1.0)
            dvi = nir_f - red_f
            
            # Exclude DVI == 0 if configured
            if config.get('mdvi_sdvi', 'exclude_zero_dvi'):
                valid2 = valid & np.isfinite(dvi) & (dvi != 0.0)
            else:
                valid2 = valid & np.isfinite(dvi)
            
            if not np.any(valid2):
                continue
            
            dvi2 = dvi.astype(np.float32)
            dvi2[~valid2] = np.nan
            stack.append(dvi2)
            
            # SDVI histogram
            vals_f = dvi2[valid2]
            vals_i = np.rint(vals_f * scale_out).astype(np.int32)
            vals_i = vals_i[(vals_i >= 0) & (vals_i <= scale_out)]
            
            if config.get('mdvi_sdvi', 'exclude_zero_dvi'):
                vals_i = vals_i[vals_i != 0]
            
            if vals_i.size:
                hist += np.bincount(vals_i, minlength=dvi_bins)
        
        # MDVI for this block
        if stack:
            st = np.stack(stack, axis=0)
            mdvi = np.nanquantile(st, mdvi_q, axis=0).astype(np.float32)
            out = np.full((h, w), nodata, dtype=np.int16)
            ok = np.isfinite(mdvi)
            if np.any(ok):
                mdvi_int = np.rint(mdvi[ok] * scale_out).astype(np.int32)
                mdvi_int = np.clip(mdvi_int, 0, scale_out).astype(np.int16)
                out[ok] = mdvi_int
            mdvi_row[:, x0:x0+w] = out
        else:
            mdvi_row[:, x0:x0+w] = nodata
        
        # SDVI for this block
        total = int(hist.sum())
        if total > 0:
            target = int(np.ceil(sdvi_q * total))
            cdf = np.cumsum(hist)
            idx = int(np.searchsorted(cdf, target))
            sdvi_row[bx] = np.int16(idx)
        else:
            sdvi_row[bx] = np.int16(nodata)
    
    return by, mdvi_row, sdvi_row


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Compute MDVI and SDVI from NBAR"
    )
    parser.add_argument('--config', type=str, help='Configuration JSON file')
    parser.add_argument('--in-dir', type=str, help='NBAR directory (overrides config)')
    parser.add_argument('--out-mdvi', type=str, help='Output MDVI file (overrides config)')
    parser.add_argument('--out-sdvi', type=str, help='Output SDVI file (overrides config)')
    parser.add_argument('--start-date', type=str, help='Start date YYYYMMDD (overrides config)')
    parser.add_argument('--end-date', type=str, help='End date YYYYMMDD (overrides config)')
    parser.add_argument('--workers', type=int, help='Number of workers (overrides config)')
    parser.add_argument('--log-file', type=str, help='Log file path')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config) if args.config else Config()
    
    # Setup logging
    log_file = args.log_file or f"logs/step2_mdvi_sdvi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logger('Step2_MDVI_SDVI', log_file)
    
    logger.info("=" * 70)
    logger.info("STEP 2: MDVI AND SDVI CALCULATION")
    logger.info("=" * 70)
    
    # Get parameters
    in_dir = Path(args.in_dir) if args.in_dir else Path(config.get('paths', 'nbar_dir'))
    out_mdvi = Path(args.out_mdvi) if args.out_mdvi else Path(config.get('paths', 'mdvi_file'))
    out_sdvi = Path(args.out_sdvi) if args.out_sdvi else Path(config.get('paths', 'sdvi_file'))
    start_date = args.start_date or config.get('dates', 'start_date')
    end_date = args.end_date or config.get('dates', 'end_date')
    n_workers = args.workers or config.get('processing', 'workers')
    
    validate_date_range(start_date, end_date)
    
    # Find NBAR files
    files = find_files(in_dir, "nbar_*.tif*", start_date, end_date)
    logger.info(f"Found {len(files)} NBAR files")
    
    if not files:
        logger.error("No NBAR files found!")
        return
    
    files_paths = [f for _, f in files]
    
    # Get template info
    tmpl = safe_open(files_paths[0], "template NBAR")
    xsize, ysize = tmpl.RasterXSize, tmpl.RasterYSize
    gt = tmpl.GetGeoTransform()
    proj = tmpl.GetProjection()
    
    expected_size = (
        config.get('processing', 'n_blocks') * config.get('processing', 'block_size'),
        config.get('processing', 'n_blocks') * config.get('processing', 'block_size')
    )
    validate_raster_size(tmpl, expected_size)
    
    nodata = config.get('nodata', 'int16')
    compress = config.get('processing', 'compress')
    nblock = config.get('processing', 'n_blocks')
    
    # Create MDVI output
    logger.info(f"Creating MDVI: {out_mdvi}")
    mdvi_ds, mdvi_band = create_geotiff(
        out_mdvi, xsize, ysize, gdal.GDT_Int16, gt, proj, nodata, compress
    )
    mdvi_band.Fill(nodata)
    mdvi_band.SetDescription("MDVI_Q95 per-pixel")
    mdvi_ds.SetMetadata({
        "INDEX": "MDVI_Q95",
        "SCALE_FACTOR": str(config.get('scales', 'mdvi')),
        "NODATA": str(nodata),
        "DVI_DEF": "NBAR_B08 - NBAR_B04"
    })
    
    # SDVI array
    sdvi = np.full((nblock, nblock), nodata, dtype=np.int16)
    
    # Process blocks
    logger.info(f"Processing with {n_workers} workers...")
    start_time = time.time()
    
    if n_workers == 1:
        for by in range(nblock):
            by_out, mdvi_row, sdvi_row = process_block_row(by, files_paths, xsize, ysize, config)
            y0 = by_out * config.get('processing', 'block_size')
            mdvi_band.WriteArray(mdvi_row, 0, y0)
            sdvi[by_out, :] = sdvi_row
            logger.info(f"Processed block row {by_out+1}/{nblock}")
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = [ex.submit(process_block_row, by, files_paths, xsize, ysize, config) 
                    for by in range(nblock)]
            for fut in as_completed(futs):
                by_out, mdvi_row, sdvi_row = fut.result()
                y0 = by_out * config.get('processing', 'block_size')
                mdvi_band.WriteArray(mdvi_row, 0, y0)
                sdvi[by_out, :] = sdvi_row
                logger.info(f"Processed block row {by_out+1}/{nblock}")
    
    mdvi_ds.FlushCache()
    mdvi_ds = None
    logger.info(f"MDVI saved: {out_mdvi}")
    
    # Create SDVI output
    logger.info(f"Creating SDVI: {out_sdvi}")
    gt_sdvi = list(gt)
    gt_sdvi[1] *= config.get('processing', 'block_size')
    gt_sdvi[5] *= config.get('processing', 'block_size')
    
    sdvi_ds, sdvi_band = create_geotiff(
        out_sdvi, nblock, nblock, gdal.GDT_Int16, gt_sdvi, proj, nodata, compress
    )
    sdvi_band.WriteArray(sdvi.astype(np.int16))
    sdvi_band.SetDescription("SDVI_Q05 block-wise")
    sdvi_ds.SetMetadata({
        "INDEX": "SDVI_Q05",
        "BLOCK_SIZE": str(config.get('processing', 'block_size')),
        "SCALE_FACTOR": str(config.get('scales', 'sdvi')),
        "NODATA": str(nodata)
    })
    sdvi_ds.FlushCache()
    sdvi_ds = None
    logger.info(f"SDVI saved: {out_sdvi}")
    
    tmpl = None
    
    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info(f"COMPLETED in {elapsed/60:.2f} minutes")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()