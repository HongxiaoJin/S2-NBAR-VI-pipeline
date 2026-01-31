#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3: Compute vegetation indices (PPI, NDVI, NIRv, EVI2) with QA from NBAR

This script calculates vegetation indices with complete imagery and informational QA.

Author: Hongxiao Jin
"""

import argparse
import time
from pathlib import Path
from datetime import datetime

import numpy as np
from osgeo import gdal
from pyproj import CRS, Transformer

from utils import (
    Config, setup_logger, find_files, safe_open, create_geotiff,
    validate_raster_size, validate_date_range, yyyymmdd_to_doy,
    grid_xy_from_gt, apply_gaussian_filter, bilinear_interpolate_grid,
    solar_zenith_angle
)

# Suppress warnings
np.seterr(divide='ignore', invalid='ignore')
gdal.UseExceptions()


def process_date(date, nbar_path, config, mdvi, mdvi_raw, sdvi, gt, proj, crs, logger):
    """
    Process single date to generate vegetation indices
    
    Parameters:
    -----------
    date : str
        Date string YYYYMMDD
    nbar_path : Path
        NBAR file path
    config : Config
        Configuration object
    mdvi, mdvi_raw, sdvi : np.ndarray
        Preprocessed MDVI and SDVI arrays
    gt, proj, crs : geotransform, projection, CRS
        Spatial reference info
    logger : logging.Logger
        Logger instance
    
    Returns:
    --------
    None (writes files to disk)
    """
    doy = yyyymmdd_to_doy(date)
    logger.info(f"Processing {date} (DOY={doy})")
    
    # Configuration
    tile = config.get('project', 'tile')
    out_dir = Path(config.get('paths', 'output_dir'))
    prefix = "CLMS_S2_sza11"
    
    xsize, ysize = 10980, 10980
    block = config.get('processing', 'block_size')
    nblock = config.get('processing', 'n_blocks')
    
    nodata_i16 = config.get('nodata', 'int16')
    qa_nodata = config.get('nodata', 'uint8')
    scale_in = config.get('scales', 'reflectance_in')
    out_scale = config.get('scales', 'output')
    
    b_red = config.get('bands', 'red')
    b_nir = config.get('bands', 'nir')
    b_cov_red = config.get('bands', 'cov_red')
    b_cov_nir = config.get('bands', 'cov_nir')
    
    ppi_min = config.get('ppi', 'min')
    ppi_max = config.get('ppi', 'max')
    unc_factor = config.get('ppi', 'uncertainty_factor')
    
    qa_invalid = config.get('qa_flags', 'invalid')
    qa_sand = config.get('qa_flags', 'sand')
    qa_unsuccessful = config.get('qa_flags', 'unsuccessful')
    
    compress = config.get('processing', 'compress')
    
    # Open NBAR
    ds = safe_open(nbar_path, f"NBAR {date}")
    
    # Create outputs
    def oname(tag):
        return out_dir / f"{prefix}_{tile}_{tag}_{date}.tiff"
    
    ppi_ds, ppi_b = create_geotiff(oname("PPI"), xsize, ysize, gdal.GDT_Int16, gt, proj, nodata_i16, compress)
    qa_ds, qa_b = create_geotiff(oname("QA"), xsize, ysize, gdal.GDT_Byte, gt, proj, qa_nodata, compress)
    
    ndvi_ds = nirv_ds = evi2_ds = None
    ndvi_b = nirv_b = evi2_b = None
    
    if config.get('vegetation_indices', 'ndvi'):
        ndvi_ds, ndvi_b = create_geotiff(oname("NDVI"), xsize, ysize, gdal.GDT_Int16, gt, proj, nodata_i16, compress)
    if config.get('vegetation_indices', 'nirv'):
        nirv_ds, nirv_b = create_geotiff(oname("NIRv"), xsize, ysize, gdal.GDT_Int16, gt, proj, nodata_i16, compress)
    if config.get('vegetation_indices', 'evi2'):
        evi2_ds, evi2_b = create_geotiff(oname("EVI2"), xsize, ysize, gdal.GDT_Int16, gt, proj, nodata_i16, compress)
    
    # CRS transformer
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    
    # Process blocks
    for by in range(nblock):
        for bx in range(nblock):
            x0, y0 = bx * block, by * block
            w = h = block
            
            # Read bands
            red_i = ds.GetRasterBand(b_red).ReadAsArray(x0, y0, w, h).astype(np.int16)
            nir_i = ds.GetRasterBand(b_nir).ReadAsArray(x0, y0, w, h).astype(np.int16)
            covr = ds.GetRasterBand(b_cov_red).ReadAsArray(x0, y0, w, h).astype(np.int16)
            covn = ds.GetRasterBand(b_cov_nir).ReadAsArray(x0, y0, w, h).astype(np.int16)
            
            # Validity
            base_valid = (red_i != nodata_i16) & (nir_i != nodata_i16)
            all_zero = (red_i == 0) & (nir_i == 0) & (covr == 0) & (covn == 0)
            
            # Scale reflectance
            RED = np.clip(red_i.astype(np.float32) * scale_in, 0.0, 1.0)
            NIR = np.clip(nir_i.astype(np.float32) * scale_in, 0.0, 1.0)
            DVI = NIR - RED
            
            # Extract MDVI/SDVI for block
            MDVI_blk = mdvi[y0:y0+h, x0:x0+w]
            MDVI_raw_blk = mdvi_raw[y0:y0+h, x0:x0+w]
            DVI_soil_blk = sdvi[y0:y0+h, x0:x0+w]
            
            # Solar zenith angle
            X, Y = grid_xy_from_gt(gt, x0, y0, w, h)
            _, lat = transformer.transform(X, Y)
            lat = lat.astype(np.float32)
            
            sza_deg = solar_zenith_angle(doy, lat, hour=11).astype(np.float32)
            cos_sza = np.cos(np.deg2rad(sza_deg)).astype(np.float32)
            
            # Atmospheric term
            dc = np.clip(0.0336 + 0.0477 / np.maximum(cos_sza, 1e-6), 0.0, 1.0).astype(np.float32)
            
            # Initialize
            PPI = np.full((h, w), np.nan, dtype=np.float32)
            QA = np.full((h, w), qa_nodata, dtype=np.uint8)
            
            # QA: Relative uncertainty
            rel_r = np.sqrt(np.maximum(covr.astype(np.float32) * scale_in, 0.0)) / (RED + 1e-6)
            rel_n = np.sqrt(np.maximum(covn.astype(np.float32) * scale_in, 0.0)) / (NIR + 1e-6)
            qa_val = np.clip(
                np.rint((unc_factor * rel_n + (1.0 - unc_factor) * rel_r) * 100.0),
                0, 254
            ).astype(np.uint8)
            QA[base_valid] = qa_val[base_valid]
            
            # PPI calculation
            num = MDVI_blk - DVI
            den = MDVI_blk - DVI_soil_blk
            
            base_ok = base_valid & (cos_sza > 0) & np.isfinite(MDVI_blk) & np.isfinite(DVI) & (den != 0)
            ratio = np.full((h, w), np.nan, dtype=np.float32)
            ratio[base_ok] = num[base_ok] / den[base_ok]
            
            EPS = 1e-12
            log_ok = base_ok & np.isfinite(ratio) & (np.abs(ratio) > EPS)
            
            if np.any(log_ok):
                PPI[log_ok] = (
                    -0.25
                    * (1.0 + MDVI_blk[log_ok]) / (1.0 - MDVI_blk[log_ok])
                    * np.log(np.abs(ratio[log_ok]))
                    / ((0.5 / cos_sza[log_ok]) * (1.0 - dc[log_ok]) + dc[log_ok])
                )
            
            # QA flags (informational)
            invalid_sparse = np.isfinite(MDVI_raw_blk) & (MDVI_raw_blk < DVI_soil_blk)
            invalid_math = base_valid & (~log_ok)
            QA[invalid_sparse | invalid_math | np.isinf(PPI)] = qa_invalid
            
            # Sand detection
            sand = (RED > 0.35) & (DVI > 0.05) & base_valid
            if np.any(sand):
                PPI[sand] = 0.0
                QA[sand] = qa_sand
            
            # Unsuccessful
            QA[all_zero] = qa_unsuccessful
            
            # Finalize PPI
            PPI = np.real(PPI).astype(np.float32)
            PPI = np.clip(PPI, ppi_min, ppi_max)
            
            # Mask negative DVI if configured
            if config.get('ppi', 'mask_negative_dvi'):
                PPI[DVI < 0] = np.nan
            
            # Write PPI and QA
            ppi_out = np.where(np.isfinite(PPI), np.rint(PPI * out_scale), nodata_i16).astype(np.int16)
            ppi_b.WriteArray(ppi_out, x0, y0)
            qa_b.WriteArray(QA, x0, y0)
            
            # Other VIs
            if ndvi_b is not None:
                ndvi = (NIR - RED) / (NIR + RED + 1e-6)
                ndvi_out = np.where(np.isfinite(ndvi), np.rint(ndvi * out_scale), nodata_i16).astype(np.int16)
                ndvi_b.WriteArray(ndvi_out, x0, y0)
            
            if nirv_b is not None:
                nirv = NIR * ((NIR - RED) / (NIR + RED + 1e-6))
                nirv_out = np.where(np.isfinite(nirv), np.rint(nirv * out_scale), nodata_i16).astype(np.int16)
                nirv_b.WriteArray(nirv_out, x0, y0)
            
            if evi2_b is not None:
                evi2 = 2.5 * (NIR - RED) / (NIR + 2.4 * RED + 1)
                evi2_out = np.where(np.isfinite(evi2), np.rint(evi2 * out_scale), nodata_i16).astype(np.int16)
                evi2_b.WriteArray(evi2_out, x0, y0)
    
    # Close outputs
    for dso in [ppi_ds, qa_ds, ndvi_ds, nirv_ds, evi2_ds]:
        if dso:
            dso.FlushCache()
    
    ds = None


def main():
    parser = argparse.ArgumentParser(
        description="Step 3: Compute vegetation indices with QA"
    )
    parser.add_argument('--config', type=str, help='Configuration JSON file')
    parser.add_argument('--in-dir', type=str, help='NBAR directory (overrides config)')
    parser.add_argument('--mdvi', type=str, help='MDVI file (overrides config)')
    parser.add_argument('--sdvi', type=str, help='SDVI file (overrides config)')
    parser.add_argument('--out-dir', type=str, help='Output directory (overrides config)')
    parser.add_argument('--start-date', type=str, help='Start date YYYYMMDD (overrides config)')
    parser.add_argument('--end-date', type=str, help='End date YYYYMMDD (overrides config)')
    parser.add_argument('--log-file', type=str, help='Log file path')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config) if args.config else Config()
    
    # Setup logging
    log_file = args.log_file or f"logs/step3_vis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logger('Step3_VIs', log_file)
    
    logger.info("=" * 70)
    logger.info("STEP 3: VEGETATION INDICES CALCULATION")
    logger.info("=" * 70)
    
    # Get parameters
    in_dir = Path(args.in_dir) if args.in_dir else Path(config.get('paths', 'nbar_dir'))
    mdvi_path = Path(args.mdvi) if args.mdvi else Path(config.get('paths', 'mdvi_file'))
    sdvi_path = Path(args.sdvi) if args.sdvi else Path(config.get('paths', 'sdvi_file'))
    out_dir = Path(args.out_dir) if args.out_dir else Path(config.get('paths', 'output_dir'))
    start_date = args.start_date or config.get('dates', 'start_date')
    end_date = args.end_date or config.get('dates', 'end_date')
    
    out_dir.mkdir(parents=True, exist_ok=True)
    validate_date_range(start_date, end_date)
    
    # Find NBAR files
    files = find_files(in_dir, "nbar_*.tif*", start_date, end_date)
    logger.info(f"Found {len(files)} NBAR files")
    
    if not files:
        logger.error("No NBAR files found!")
        return
    
    # Get template info
    tmpl = safe_open(files[0][1], "template NBAR")
    xsize, ysize = tmpl.RasterXSize, tmpl.RasterYSize
    gt = tmpl.GetGeoTransform()
    proj = tmpl.GetProjection()
    validate_raster_size(tmpl, (10980, 10980))
    tmpl = None
    
    crs = CRS.from_wkt(proj)
    
    nodata_i16 = config.get('nodata', 'int16')
    mdvi_scale = config.get('scales', 'mdvi')
    sdvi_scale = config.get('scales', 'sdvi')
    
    # Load MDVI
    logger.info(f"Loading MDVI from {mdvi_path}")
    mdvi_ds = safe_open(mdvi_path, "MDVI")
    mdvi_array = mdvi_ds.GetRasterBand(1).ReadAsArray()
    mdvi_ds = None
    
    mdvi_raw = mdvi_array.astype(np.float32) / mdvi_scale
    mdvi_raw[mdvi_array == nodata_i16] = np.nan
    
    # Apply MDVI adjustments and filtering
    mdvi_adjusted = np.clip(
        mdvi_raw,
        config.get('filters', 'mdvi_clip_min'),
        config.get('filters', 'mdvi_clip_max')
    ) + config.get('filters', 'mdvi_adjustment')
    
    logger.info("Applying Gaussian filter to MDVI...")
    mdvi = apply_gaussian_filter(
        mdvi_adjusted,
        sigma=config.get('filters', 'gaussian_sigma'),
        nodata_value=np.nan
    )
    
    # Load SDVI
    logger.info(f"Loading SDVI from {sdvi_path}")
    sdvi_ds = safe_open(sdvi_path, "SDVI")
    sdvi_array_20x20 = sdvi_ds.GetRasterBand(1).ReadAsArray()
    sdvi_ds = None
    
    if sdvi_array_20x20.shape != (20, 20):
        raise ValueError(f"SDVI shape {sdvi_array_20x20.shape} != (20, 20)")
    
    sdvi_20x20 = sdvi_array_20x20.astype(np.float32) / sdvi_scale
    sdvi_20x20[sdvi_array_20x20 == nodata_i16] = np.nan
    
    logger.info("Interpolating SDVI from 20x20 to 10980x10980...")
    sdvi_full = bilinear_interpolate_grid(
        sdvi_20x20,
        target_shape=(10980, 10980),
        block_size=config.get('processing', 'block_size')
    )
    sdvi = np.clip(
        sdvi_full,
        config.get('filters', 'sdvi_min'),
        config.get('filters', 'sdvi_max')
    )
    
    # Process each date
    start_time = time.time()
    
    for i, (date, nbar_path) in enumerate(files, 1):
        logger.info(f"[{i}/{len(files)}] Processing {date}")
        try:
            process_date(date, nbar_path, config, mdvi, mdvi_raw, sdvi, gt, proj, crs, logger)
        except Exception as e:
            logger.error(f"Error processing {date}: {e}", exc_info=True)
            continue
    
    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info(f"COMPLETED in {elapsed/60:.2f} minutes")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()