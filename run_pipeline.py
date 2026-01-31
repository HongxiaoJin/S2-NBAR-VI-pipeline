#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master pipeline runner for Sentinel-2 NBAR processing

This script runs all three steps of the processing chain:
1. NBAR calculation from magnitude
2. MDVI and SDVI calculation
3. Vegetation indices calculation (PPI, NDVI, NIRv, EVI2)
Run:
bash# Full pipeline
module load Anaconda3
conda activate cglops
python run_pipeline.py --config config/config_33WXR.json

# Or individual steps
python step2_mdvi_sdvi.py --config config/config_29SQB.json
python step3_vegetation_indices.py --config config/config_32UMU.json
Author: Hongxiao Jin
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime

from utils import Config, setup_logger


def run_step(step_name, script_path, config_path, logger):
    """
    Run a processing step
    
    Parameters:
    -----------
    step_name : str
        Name of the step
    script_path : str
        Path to the step script
    config_path : str
        Path to configuration file
    logger : logging.Logger
        Logger instance
    
    Returns:
    --------
    bool : True if successful, False otherwise
    """
    logger.info("=" * 70)
    logger.info(f"RUNNING: {step_name}")
    logger.info("=" * 70)
    
    cmd = [sys.executable, script_path, '--config', config_path]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        logger.info(f"{step_name} completed successfully")
        if result.stdout:
            logger.debug(result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"{step_name} failed!")
        logger.error(f"Return code: {e.returncode}")
        if e.stdout:
            logger.error(f"STDOUT:\n{e.stdout}")
        if e.stderr:
            logger.error(f"STDERR:\n{e.stderr}")
        
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Master pipeline for S2 NBAR processing"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Configuration JSON file'
    )
    parser.add_argument(
        '--steps',
        type=str,
        default='1,2,3',
        help='Steps to run (comma-separated, e.g., "1,2,3" or "2,3")'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        help='Master log file path'
    )
    
    args = parser.parse_args()
    
    # Parse steps
    steps_to_run = [int(s.strip()) for s in args.steps.split(',')]
    
    # Setup logging
    log_file = args.log_file or f"logs/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logger('Pipeline_Master', log_file)
    
    logger.info("=" * 70)
    logger.info("SENTINEL-2 NBAR PROCESSING PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Steps to run: {steps_to_run}")
    logger.info("")
    
    # Load config to verify
    config = Config(args.config)
    logger.info(f"Project: {config.get('project', 'name')}")
    logger.info(f"Tile: {config.get('project', 'tile')}")
    logger.info(f"Date range: {config.get('dates', 'start_date')} to {config.get('dates', 'end_date')}")
    logger.info("")
    
    # Define steps
    script_dir = Path(__file__).parent
    steps = {
        1: ("Step 1: NBAR from Magnitude", script_dir / "step1_nbar.py"),
        2: ("Step 2: MDVI and SDVI", script_dir / "step2_mdvi_sdvi.py"),
        3: ("Step 3: Vegetation Indices", script_dir / "step3_vegetation_indices.py")
    }
    
    # Run steps
    success_count = 0
    fail_count = 0
    
    for step_num in sorted(steps_to_run):
        if step_num not in steps:
            logger.warning(f"Unknown step: {step_num}, skipping")
            continue
        
        step_name, script_path = steps[step_num]
        
        if run_step(step_name, str(script_path), args.config, logger):
            success_count += 1
        else:
            fail_count += 1
            logger.error(f"Stopping pipeline due to failure in step {step_num}")
            break
    
    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Successful steps: {success_count}")
    logger.info(f"Failed steps: {fail_count}")
    
    if fail_count == 0:
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    else:
        logger.error("PIPELINE FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
