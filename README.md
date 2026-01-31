# S2-NBAR-VI-pipeline
Calculate S2 Vegetation indices (PPI, NIRv, EVI2, and NDVI) after get magnitude using WIMBALS inversion 

A complete, production-ready framework for processing Sentinel-2 BRDF magnitude data to vegetation indices (PPI, NDVI, NIRv, EVI2) with quality assessment.

**Author:** Hongxiao Jin  
**Version:** 1.0.0  
**Date:** January 2026

---

## Overview

This pipeline processes Sentinel-2 data through three main steps:

1. **NBAR Calculation** - Convert BRDF magnitude to NBAR (Nadir BRDF-Adjusted Reflectance)
2. **MDVI/SDVI Calculation** - Compute Maximum DVI and Soil DVI statistics
3. **Vegetation Indices** - Calculate PPI, NDVI, NIRv, EVI2 with QA flags

### Key Features

✅ **Modular Design** - Each step can run independently  
✅ **JSON Configuration** - Centralized settings management  
✅ **Comprehensive Logging** - Detailed logs for debugging and monitoring  
✅ **Parallel Processing** - Multi-worker support for faster computation  
✅ **Quality Assessment** - Informational QA flags without blocking outputs  
✅ **Noise Reduction** - Gaussian filtering and bilinear interpolation  

---

## Directory Structure

```
S2-NBAR-VI-pipeline/
├── utils.py                      # Shared utilities
├── step1_nbar.py                 # Step 1: NBAR calculation
├── step2_mdvi_sdvi.py            # Step 2: MDVI/SDVI calculation
├── step3_vegetation_indices.py   # Step 3: VI calculation
├── run_pipeline.py               # Master pipeline runner
├── config/
│   └── config_template.json      # Configuration template
└── logs/                         # Processing logs (auto-created)
```

---

## Installation

### Requirements

```bash
# Core dependencies
pip install --break-system-packages numpy scipy gdal rasterio pyproj

# Or using conda
conda install -c conda-forge numpy scipy gdal rasterio pyproj
```

### Python Version

- Python 3.8+

---

## Configuration

### 1. Create Your Configuration File

Copy the template and customize for your tile:

```bash
cp config/config_template.json config/config_32UMU.json
```

### 2. Edit Configuration

Key sections to modify:

```json
{
  "project": {
    "tile": "32UMU",  // Your Sentinel-2 tile ID
    "name": "S2_NBAR_Processing"
  },
  
  "paths": {
    "magnitude_dir": "/path/to/magnitude/data",
    "nbar_dir": "/path/to/nbar/output",
    "mdvi_file": "/path/to/mdvi/output.tif",
    "sdvi_file": "/path/to/sdvi/output.tif",
    "output_dir": "/path/to/final/outputs"
  },
  
  "dates": {
    "start_date": "20190101",  // YYYYMMDD
    "end_date": "20231231"     // YYYYMMDD
  },
  
  "processing": {
    "workers": 10,  // Parallel workers (adjust for your system)
    "compress": "ZSTD"
  }
}
```

See `config/config_template.json` for all available options.

---

## Usage

### Option 1: Run Complete Pipeline

Process all three steps sequentially:

```bash
python run_pipeline.py --config config/config_32UMU.json
```

### Option 2: Run Individual Steps

#### Step 1: NBAR from Magnitude

```bash
python step1_nbar.py \
  --config config/config_32UMU.json \
  --log-file logs/step1_nbar.log
```

Or override config values:

```bash
python step1_nbar.py \
  --config config/config_32UMU.json \
  --folder-mag /custom/magnitude/path \
  --out-dir /custom/nbar/output \
  --sza sza11
```

#### Step 2: MDVI and SDVI

```bash
python step2_mdvi_sdvi.py \
  --config config/config_32UMU.json \
  --workers 10 \
  --log-file logs/step2_mdvi_sdvi.log
```

#### Step 3: Vegetation Indices

```bash
python step3_vegetation_indices.py \
  --config config/config_32UMU.json \
  --log-file logs/step3_vis.log
```

### Option 3: Run Specific Steps

Run only steps 2 and 3 (skip NBAR if already done):

```bash
python run_pipeline.py \
  --config config/config_32UMU.json \
  --steps 2,3
```

---

## Output Files

### Step 1: NBAR

```
nbar_YYYYMMDD.tiff  (8 bands: B02, B03, B04, B08 + covariances)
```

### Step 2: MDVI/SDVI

```
{tile}_nbar_sza11_MDVI_Q95_int16.tif  (10980×10980, per-pixel MDVI)
{tile}_nbar_sza11_SDVI_Q05_20x20_int16.tif  (20×20, block-wise SDVI)
```

### Step 3: Vegetation Indices

For each date:

```
CLMS_S2_sza11_{TILE}_PPI_{YYYYMMDD}.tiff   (Plant Phenology Index)
CLMS_S2_sza11_{TILE}_QA_{YYYYMMDD}.tiff    (Quality Assessment)
CLMS_S2_sza11_{TILE}_NDVI_{YYYYMMDD}.tiff  (Normalized Difference VI)
CLMS_S2_sza11_{TILE}_NIRv_{YYYYMMDD}.tiff  (NIR Reflectance VI)
CLMS_S2_sza11_{TILE}_EVI2_{YYYYMMDD}.tiff  (Enhanced VI 2)
```

All outputs are Int16 GeoTIFFs with:
- Scale factor: 1000 (physical_value = stored_value / 1000)
- NoData: -9999
- Compression: ZSTD

---

## Quality Assessment (QA) Flags

The QA layer provides informational quality flags:

| Value | Meaning | Description |
|-------|---------|-------------|
| 0-254 | Relative Uncertainty | Lower = better quality (0=best) |
| 252 | Sand | Bright sand pixel, PPI forced to 0 |
| 253 | Unsuccessful | All bands are zero |
| 254 | Invalid | Sparse vegetation, math errors, or inf values |
| 255 | NoData | No data available |

**Important:** QA flags are **informational only** and do not mask the PPI output. Users can filter based on QA in downstream analysis.

---

## Algorithm Details

### PPI (Plant Phenology Index)

```
PPI = -0.25 × (1 + MDVI) / (1 - MDVI) × log(|ratio|) / atmospheric_term

where:
  ratio = (MDVI - DVI) / (MDVI - DVI_soil)
  atmospheric_term = (0.5/cos_sza) × (1 - dc) + dc
  dc = 0.0336 + 0.0477 / cos_sza
```

Features:
- MDVI smoothed with 5×5 Gaussian filter (σ=1.0)
- SDVI interpolated from 20×20 to 10980×10980 (bilinear)
- Masked where DVI < 0
- Clipped to [-1.0, 5.0]

### MDVI (Maximum DVI)

- 95th percentile of (NIR - RED) per pixel across time series
- Excludes NoData and DVI=0

### SDVI (Soil DVI)

- 5th percentile of DVI histogram per 20×20 block
- Excludes NoData and DVI=0
- Constrained to [0.005, 0.09]

---

## Performance Tips

### Parallel Processing

Adjust workers based on available CPU cores:

```json
{
  "processing": {
    "workers": 10  // Set to number of CPU cores
  }
}
```

### Memory Optimization

For large tiles or limited RAM:
- Reduce number of workers
- Process fewer dates per run
- Use more aggressive compression

### Compression

Available options (fastest to best compression):

```json
{
  "processing": {
    "compress": "LZW"     // Fast, good compression
    "compress": "ZSTD"    // Recommended: excellent speed/compression balance
    "compress": "DEFLATE" // Slower, universal compatibility
  }
}
```

---

## Troubleshooting

### "No NBAR files found"

**Problem:** Step 2 or 3 can't find input files

**Solution:**
1. Check file naming: `nbar_YYYYMMDD.tif` or `nbar_YYYYMMDD.tiff`
2. Verify date range in config
3. Check input directory path

### "Unexpected raster size"

**Problem:** Input rasters are not 10980×10980

**Solution:**
- Verify Sentinel-2 tile is complete
- Check for partial/corrupted files
- Adjust `block_size` and `n_blocks` in config if using different resolution

### Memory Errors

**Problem:** Out of memory during processing

**Solution:**
1. Reduce number of workers
2. Process date range in smaller chunks
3. Close other applications

### GDAL Errors

**Problem:** "Cannot open file" or GDAL-related errors

**Solution:**
```bash
# Check GDAL installation
gdalinfo --version

# Reinstall if needed
pip install --break-system-packages --upgrade gdal
```

---

## Logging

Logs are automatically created in the `logs/` directory:

```
logs/
├── pipeline_20260129_143052.log     # Master pipeline log
├── step1_nbar_20260129_143055.log   # Step 1 log
├── step2_mdvi_sdvi_20260129_151032.log  # Step 2 log
└── step3_vis_20260129_160211.log    # Step 3 log
```

Each log contains:
- Timestamp for each operation
- Progress updates
- Warning and error messages
- Processing statistics

---

## Example Workflows

### Process Single Tile

```bash
# 1. Create config
cp config/config_template.json config/config_32UMU.json
# Edit config_32UMU.json with your paths

# 2. Run full pipeline
python run_pipeline.py --config config/config_32UMU.json
```

### Process Multiple Tiles

```bash
# Create configs for each tile
for TILE in 29SQB 32UMU 33VVD; do
  cp config/config_template.json config/config_${TILE}.json
  # Edit each config file
  
  # Run pipeline
  python run_pipeline.py --config config/config_${TILE}.json
done
```

### Reprocess Only VIs

If MDVI/SDVI are already computed:

```bash
python run_pipeline.py \
  --config config/config_32UMU.json \
  --steps 3
```

---

## Advanced Configuration

### Custom Filter Parameters

```json
{
  "filters": {
    "gaussian_sigma": 1.5,      // Increase for more smoothing
    "mdvi_clip_min": 0.15,      // Adjust MDVI range
    "mdvi_clip_max": 0.95,
    "sdvi_min": 0.003,          // Adjust SDVI constraints
    "sdvi_max": 0.10
  }
}
```

### PPI Customization

```json
{
  "ppi": {
    "min": -2.0,                 // Extend PPI range
    "max": 6.0,
    "uncertainty_factor": 0.85,  // Adjust uncertainty weighting
    "mask_negative_dvi": false   // Keep negative DVI pixels
  }
}
```

### Select Specific VIs

```json
{
  "vegetation_indices": {
    "ppi": true,
    "ndvi": true,
    "nirv": false,  // Disable NIRv
    "evi2": false   // Disable EVI2
  }
}
```

---

## Citation

If you use this software in your research, please cite:

```
Jin, H. (2026). Sentinel-2 NBAR Vegetation Indices Processing Pipeline.
Version 1.0.0. https://github.com/yourusername/s2-nbar-processing
```

---

## License

This software is provided as-is for research and operational use.

---

## Support

For questions or issues:
1. Check this README
2. Review log files for error messages
3. Contact: hongxiao.jin@example.com

---

## Version History

### v1.0.0 (2026-01-29)
- Initial release
- Modular three-step pipeline
- JSON configuration system
- Comprehensive logging
- Quality assessment flags
- Gaussian filtering and bilinear interpolation
