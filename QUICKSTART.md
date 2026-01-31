# Quick Start Guide

Get up and running with the S2 NBAR processing pipeline in 5 minutes!

## Prerequisites

```bash
# Install dependencies
pip install --break-system-packages numpy scipy gdal rasterio pyproj
```

## Step 1: Configure (2 minutes)

```bash
# Copy template
cp config/config_template.json config/my_tile.json

# Edit the following in my_tile.json:
# - project.tile: "YOUR_TILE_ID"
# - paths.magnitude_dir: "/path/to/your/magnitude/data"
# - paths.nbar_dir: "/path/to/nbar/output"
# - paths.output_dir: "/path/to/final/output"
# - dates.start_date: "YYYYMMDD"
# - dates.end_date: "YYYYMMDD"
```

## Step 2: Run Pipeline (< 2 minutes to start)

```bash
# Run all three steps
python run_pipeline.py --config config/my_tile.json

# Or run specific step
python step1_nbar.py --config config/my_tile.json
python step2_mdvi_sdvi.py --config config/my_tile.json
python step3_vegetation_indices.py --config config/my_tile.json
```

## Step 3: Check Results

```bash
# View logs
tail -f logs/pipeline_*.log

# Check outputs
ls -lh /path/to/final/output/
```

## Common Issues

**"No files found"**
→ Check your paths in config file

**"Out of memory"**
→ Reduce `processing.workers` in config

**"GDAL error"**
→ Reinstall: `pip install --upgrade gdal --break-system-packages`

## Next Steps

- Read [README.md](README.md) for detailed documentation
- Adjust filtering parameters in config
- Process multiple tiles with different configs

---

That's it! You're processing Sentinel-2 data! 🛰️🌱