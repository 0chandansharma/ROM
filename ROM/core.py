#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Range of Motion (ROM) Assessment Tool

A specialized framework for remote physiotherapy assessment,
focusing on joint range of motion analysis for clinical purposes.

This tool helps physiotherapists measure, track, and visualize
joint mobility for remote patient assessments.
"""

import argparse
import logging
import sys
import toml
from datetime import datetime
from pathlib import Path

from ROM.processor import analyze_movement
from ROM.assessment.movements import AVAILABLE_ASSESSMENTS
from ROM.utils.angle_calc import DEFAULT_CONFIG

__version__ = "0.1.0"
VERSION = __version__


def read_config(config_path):
    """
    Load configuration from TOML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    try:
        config_dict = toml.load(config_path)
        return config_dict
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        sys.exit(1)


def setup_logging(log_path):
    """
    Configure logging for the application
    
    Args:
        log_path: Path where logs will be saved
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_path, 'a+') as log_file:
        pass
    
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )


def process(config=None):
    """
    Process video for ROM assessment based on configuration
    
    Args:
        config: Configuration dictionary or path to config file
    """
    # Handle configuration
    if config is None:
        config = DEFAULT_CONFIG
    elif isinstance(config, str):
        config = read_config(config)
    
    # Setup paths
    result_dir = Path(config.get('process', {}).get('result_dir', ''))
    if not result_dir:
        result_dir = Path.cwd()
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_path = result_dir / 'rom_assessment.log'
    setup_logging(log_path)
    
    # Log start of assessment
    assessment_type = config.get('project', {}).get('assessment', 'general')
    logging.info(f"Starting ROM assessment: {assessment_type}")
    logging.info(f"ROM version: {__version__}")
    
    # Process videos sequentially
    video_input = config.get('project', {}).get('video_input', [])
    if isinstance(video_input, str):
        video_input = [video_input]
    
    for video_file in video_input:
        logging.info(f"Processing video: {video_file}")
        start_time = datetime.now()
        
        # Core processing function
        analyze_movement(config, video_file, result_dir)
        
        # Log completion
        duration = (datetime.now() - start_time).total_seconds()
        logging.info(f"Assessment completed in {duration:.2f} seconds")


def main():
    """
    Command-line interface for ROM assessment tool
    """
    parser = argparse.ArgumentParser(
        description="ROM: Range of Motion Assessment Tool for Remote Physiotherapy"
    )
    
    # Main arguments
    parser.add_argument('-C', '--config', type=str, 
                        help='Path to TOML configuration file')
    parser.add_argument('-i', '--video_input', type=str, nargs='+',
                        help='Video file path(s) or "webcam" for live capture')
    parser.add_argument('-a', '--assessment', type=str, choices=AVAILABLE_ASSESSMENTS,
                        help='Assessment type to perform')
    parser.add_argument('-o', '--output_dir', type=str,
                        help='Directory to save results')
    parser.add_argument('-v', '--version', action='version', 
                        version=f'ROM v{__version__}')
    
    # Process arguments
    args = parser.parse_args()
    
    # Initialize configuration
    if args.config:
        config = read_config(args.config)
    else:
        config = DEFAULT_CONFIG.copy()
    
    # Override with command line arguments
    if args.video_input:
        if not isinstance(config['project'], dict):
            config['project'] = {}
        config['project']['video_input'] = args.video_input
    
    if args.assessment:
        if not isinstance(config['project'], dict):
            config['project'] = {}
        config['project']['assessment'] = args.assessment
    
    if args.output_dir:
        if not isinstance(config['process'], dict):
            config['process'] = {}
        config['process']['result_dir'] = args.output_dir
    
    # Run processing
    process(config)


if __name__ == "__main__":
    main()