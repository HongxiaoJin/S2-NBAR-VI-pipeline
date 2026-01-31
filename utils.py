#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared utilities for Sentinel-2 NBAR processing pipeline

Author: Hongxiao Jin, 2026-01-29
"""

import re
import json
import logging
from pathlib import Path
from datetime import datetime
import os

import numpy as np
from osgeo import gdal
from pyproj import CRS, Transformer
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator

gdal.UseExceptions()


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration manager for S2 NBAR processing"""
    
    def __init__(self, config_path=None):
        """Load configuration from JSON file or use defaults"""
        self.config = self._load_defaults()
        
        if config_path:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                self._update_nested(self.config, user_config)
    
    @staticmethod
    def _load_defaults():
        """Default configuration"""
        return {
            "processing": {
                "block_size": 549,
                "n_blocks": 20,
                "workers": 4
            },
            "scales": {
                "reflectance_in": 1e-4,
                "mdvi": 10000.0,
                "sdvi": 10000.0,
                "output": 1000.0
            },
            "nodata": {
                "int16": -9999,
                "uint8": 255
            },
            "bands": {
                "red": 5,
                "nir": 7,
                "cov_red": 6,
                "cov_nir": 8
            },
            "wavelengths": {
                "b02": 490,
                "b03": 560,
                "b04": 665,
                "b08": 842
            },
            "ppi": {
                "min": -1.0,
                "max": 5.0,
                "uncertainty_factor": 0.90
            },
            "qa_flags": {
                "invalid": 254,
                "sand": 252,
                "unsuccessful": 253,
                "nodata": 255
            },
            "filters": {
                "gaussian_sigma": 1.0,
                "mdvi_clip_min": 0.18,
                "mdvi_clip_max": 0.99,
                "mdvi_adjustment": 0.005,
                "sdvi_min": 0.005,
                "sdvi_max": 0.09
            }
        }
    
    @staticmethod
    def _update_nested(d, u):
        """Recursively update nested dict"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d:
                Config._update_nested(d[k], v)
            else:
                d[k] = v
    
    def get(self, *keys):
        """Get nested config value"""
        val = self.config
        for key in keys:
            val = val[key]
        return val
    
    def save(self, path):
        """Save configuration to JSON"""
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=2)


# ============================================================================
# LOGGING
# ============================================================================

def setup_logger(name, log_file=None, level=logging.INFO):
    """Setup logger with console and optional file output"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if called multiple times
    logger.handlers.clear()
    logger.propagate = False

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    console_fmt = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console.setFormatter(console_fmt)
    logger.addHandler(console)

    # File handler (ensure directory exists)
    if log_file:
        log_dir = os.path.dirname(os.path.abspath(log_file))
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(level)
        file_fmt = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)

    return logger

# ============================================================================
# FILE UTILITIES
# ============================================================================

def parse_date(fname):
    """Extract YYYYMMDD from filename"""
    m = re.search(r'(\d{8})', fname)
    return m.group(1) if m else None


def yyyymmdd_to_doy(date_str):
    """Convert YYYYMMDD to day of year"""
    return int(datetime.strptime(date_str, "%Y%m%d").strftime("%j"))


def find_files(folder, pattern, start_date=None, end_date=None):
    """
    Find files matching pattern within date range
    
    Parameters:
    -----------
    folder : Path
        Directory to search
    pattern : str
        Glob pattern (e.g., "nbar_*.tif")
    start_date : str, optional
        Start date YYYYMMDD (inclusive)
    end_date : str, optional
        End date YYYYMMDD (inclusive)
    
    Returns:
    --------
    list of (date, Path) tuples, sorted by date
    """
    folder = Path(folder)
    files = []
    
    for f in sorted(folder.glob(pattern)):
        # Skip auxiliary files
        if f.suffix.lower() not in ['.tif', '.tiff']:
            continue
        if '.aux.xml' in f.name:
            continue
            
        date = parse_date(f.name)
        if not date:
            continue
            
        if start_date and date < start_date:
            continue
        if end_date and date > end_date:
            continue
            
        files.append((date, f))
    
    return sorted(files, key=lambda x: x[0])


# ============================================================================
# GDAL UTILITIES
# ============================================================================

def safe_open(path, desc="file"):
    """Safely open GDAL dataset with error checking"""
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Cannot open {desc}: {path}")
    return ds


def create_geotiff(path, xsize, ysize, dtype, gt, proj, nodata, compress="ZSTD"):
    """
    Create a GeoTIFF file
    
    Parameters:
    -----------
    path : Path
        Output file path
    xsize, ysize : int
        Raster dimensions
    dtype : gdal.GDT_*
        GDAL data type
    gt : tuple
        Geotransform
    proj : str
        Projection WKT
    nodata : numeric
        NoData value
    compress : str
        Compression method
    
    Returns:
    --------
    ds, band : (Dataset, Band)
    """
    drv = gdal.GetDriverByName("GTiff")
    opts = ["TILED=YES", "BIGTIFF=IF_SAFER"]
    if compress and compress.upper() != "NONE":
        opts.append(f"COMPRESS={compress}")
    
    ds = drv.Create(str(path), xsize, ysize, 1, dtype, options=opts)
    ds.SetGeoTransform(gt)
    ds.SetProjection(proj)
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    
    return ds, band


# ============================================================================
# GEOMETRIC UTILITIES
# ============================================================================

def grid_xy_from_gt(gt, x0, y0, w, h):
    """
    Calculate X, Y coordinates for pixel centers
    
    Parameters:
    -----------
    gt : tuple
        Geotransform (6 elements)
    x0, y0 : int
        Block offset
    w, h : int
        Block dimensions
    
    Returns:
    --------
    X, Y : np.ndarray
        Coordinate arrays (h, w)
    """
    rr, cc = np.indices((h, w), dtype=np.float64)
    cc = cc + (x0 + 0.5)
    rr = rr + (y0 + 0.5)
    X = gt[0] + cc * gt[1] + rr * gt[2]
    Y = gt[3] + cc * gt[4] + rr * gt[5]
    return X, Y


def latitude_array_for_window(src, window):
    """
    Get latitude array for a rasterio window
    
    Parameters:
    -----------
    src : rasterio.DatasetReader
        Source dataset
    window : rasterio.windows.Window
        Window specification
    
    Returns:
    --------
    lat : np.ndarray
        Latitude array (window.height, window.width)
    """
    from pyproj import Transformer
    
    transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
    
    row0 = int(window.row_off)
    col0 = int(window.col_off)
    h = int(window.height)
    w = int(window.width)
    
    rr, cc = np.indices((h, w), dtype=np.float64)
    rr += row0 + 0.5
    cc += col0 + 0.5
    
    T = src.transform
    xs = T.a * cc + T.b * rr + T.c
    ys = T.d * cc + T.e * rr + T.f
    
    _, lats = transformer.transform(xs, ys)
    return lats.astype(np.float32)


# ============================================================================
# SOLAR GEOMETRY
# ============================================================================

def solar_zenith_angle(doy, lat, hour=11):
    """
    Calculate solar zenith angle (Modified from VITO script)
    
    Parameters:
    -----------
    doy : int or np.ndarray
        Day of year
    lat : float or np.ndarray
        Latitude (degrees)
    hour : float
        Hour of day (default: 11 for Sentinel-2 average passing time over EU area)
    
    Returns:
    --------
    sza : np.ndarray
        Solar zenith angle (degrees)
    """
    # Solar declination table (every 5 days)
    dec = np.array([
        -23.06, -22.57, -21.91, -21.06, -20.05, -18.88, -17.57, -16.13, -14.57, -12.91,
        -11.16, -9.34, -7.46, -5.54, -3.59, -1.62, 0.36, 2.33, 4.28, 6.19,
         8.06, 9.88, 11.62, 13.29, 14.87, 16.34, 17.70, 18.94, 20.04, 21.00,
         21.81, 22.47, 22.95, 23.28, 23.43, 23.40, 23.21, 22.85, 22.32, 21.63,
         20.79, 19.80, 18.67, 17.42, 16.05, 14.57, 13.00, 11.33, 9.60, 7.80,
         5.95, 4.06, 2.13, 0.19, -1.75, -3.69, -5.62, -7.51, -9.36, -11.16,
        -12.88, -14.53, -16.07, -17.50, -18.81, -19.98, -20.99, -21.85, -22.52, -23.02,
        -23.33, -23.44, -23.35, -23.06
    ], dtype=np.float32)
    
    nday = np.array([1 + i * 5 for i in range(73)] + [366], dtype=np.float32)
    dtor = np.pi / 180.0
    
    # Adjusted day for solar time
    tt = ((doy + hour / 24.0 - 1.0) % 365.25) + 1.0
    decang = np.interp(tt, nday, dec)
    
    # Zenith angle calculation
    t0 = (90.0 - lat) * dtor
    t1 = (90.0 - decang) * dtor
    zz = (
        np.cos(t0) * np.cos(t1)
        + np.sin(t0) * np.sin(t1) * np.cos((hour - 12.0) * 15.0 * dtor)
    )
    
    return np.arccos(np.clip(zz, -1, 1)) / dtor


# ============================================================================
# BRDF NORMALIZATION
# ============================================================================

def Ra_normalize(SZA_deg, band):
    band_VR = np.array([
        [443, 0.5411, 0.2133], [490, 0.5403, 0.211], [560, 0.5939, 0.1921],
        [665, 0.5189, 0.1944], [705, 0.51, 0.1832], [740, 0.5002, 0.171],
        [783, 0.4882, 0.1559], [842, 0.4718, 0.1353], [865, 0.4673, 0.1297],
        [945, 0.4605, 0.1324], [1375, 0.4233, 0.148], [1610, 0.4027, 0.1576],
        [2190, 0.4262, 0.1782]
    ])
    match = np.where(band_VR[:, 0] == band)[0]
    if len(match) == 0:
        raise ValueError(f"Band {band} not found.")

    V = band_VR[match[0], 1]
    R = band_VR[match[0], 2]

    SZA_rad = SZA_deg * np.pi / 180.0
    return 1 + V * kvol_0(SZA_rad) + R * kgeo_0(SZA_rad)


def kvol_0(theta_s):
    cos_t = np.cos(theta_s)
    sin_t = np.sin(theta_s)
    num = (np.pi/2 - theta_s)*cos_t + sin_t
    den = cos_t + 1
    return num / den - np.pi/4


def kgeo_0(theta_s):
    cos_t = np.cos(theta_s)
    tan_t = np.tan(theta_s)
    sec_t = 1.0 / cos_t
    cos_x = 2.0 * tan_t / (sec_t + 1.0)
    cos_x = np.clip(cos_x, -1.0, 1.0)
    x = np.arccos(cos_x)
    overlap = (1.0/np.pi) * (x - np.sin(x)*np.cos(x)) * (sec_t + 1.0)
    return overlap - sec_t - 1.0 + 0.5 * (1 + cos_t) * sec_t

# ============================================================================
# FILTERING AND INTERPOLATION
# ============================================================================

def apply_gaussian_filter(array, sigma=1.0, nodata_value=None):
    """
    Apply Gaussian filter to array
    
    Parameters:
    -----------
    array : np.ndarray
        Input array
    sigma : float
        Gaussian kernel sigma
    nodata_value : float, optional
        NoData value to mask
    
    Returns:
    --------
    filtered : np.ndarray
        Filtered array
    """
    if nodata_value is not None:
        valid_mask = ~np.isnan(array) if np.isnan(nodata_value) else (array != nodata_value)
        valid_data = array[valid_mask]
        
        if len(valid_data) > 0:
            fill_value = np.nanmean(valid_data)
        else:
            fill_value = 0.0
        
        array_filled = array.copy()
        array_filled[~valid_mask] = fill_value
        filtered = gaussian_filter(array_filled, sigma=sigma, mode='reflect', truncate=2.5)
    else:
        filtered = gaussian_filter(array, sigma=sigma, mode='reflect', truncate=2.5)
    
    return filtered


def bilinear_interpolate_grid(data_20x20, target_shape=(10980, 10980), block_size=549):
    """
    Interpolate 20x20 grid to full resolution
    
    Parameters:
    -----------
    data_20x20 : np.ndarray
        Input array (20, 20)
    target_shape : tuple
        Target dimensions
    block_size : int
        Size of each block (549 for Sentinel-2)
    
    Returns:
    --------
    data_full : np.ndarray
        Interpolated array at target resolution
    """
    # Source coordinates (block centers)
    src_y = np.arange(20) * block_size + block_size / 2.0
    src_x = np.arange(20) * block_size + block_size / 2.0
    
    interpolator = RegularGridInterpolator(
        (src_y, src_x), 
        data_20x20, 
        method='linear',
        bounds_error=False,
        fill_value=None
    )
    
    # Target coordinates
    tgt_y, tgt_x = np.meshgrid(
        np.arange(target_shape[0]),
        np.arange(target_shape[1]),
        indexing='ij'
    )
    
    points = np.stack([tgt_y.ravel(), tgt_x.ravel()], axis=1)
    data_full = interpolator(points).reshape(target_shape)
    
    return data_full.astype(np.float32)


# ============================================================================
# VALIDATION
# ============================================================================

def validate_raster_size(ds, expected_size=(10980, 10980)):
    """Validate raster dimensions"""
    actual = (ds.RasterXSize, ds.RasterYSize)
    if actual != expected_size:
        raise ValueError(f"Unexpected raster size {actual}, expected {expected_size}")


def validate_date_range(start_date, end_date):
    """Validate date range format and order"""
    try:
        start = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")
    except ValueError as e:
        raise ValueError(f"Invalid date format (expected YYYYMMDD): {e}")
    
    if start > end:
        raise ValueError(f"Start date {start_date} is after end date {end_date}")
