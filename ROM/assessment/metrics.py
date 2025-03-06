#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Range of Motion Metrics

This module provides functions for calculating range of motion metrics
from joint angle data, including minimum, maximum, average values, and
comparison to clinical norms.
"""

import json
import numpy as np
import pandas as pd
import logging
from pathlib import Path


def calculate_rom_metrics(angles_df, assessment_config):
    """
    Calculate ROM metrics from angle measurements
    
    Args:
        angles_df: DataFrame containing angle measurements over time
        assessment_config: Configuration for the assessment
    
    Returns:
        Dictionary of ROM metrics for each joint
    """
    metrics = {}
    
    # Process each joint in the assessment
    for joint in assessment_config['target_joints']:
        # Extract angle data for the joint
        joint_data = {}
        
        # Check if we need to process right and left sides separately
        if assessment_config['primary_side'] == 'both':
            for side in ['right', 'left']:
                column_name = f"{side}_{joint}"
                if column_name in angles_df.columns:
                    # Filter out NaN values
                    valid_angles = angles_df[column_name].dropna()
                    
                    if len(valid_angles) > 0:
                        joint_data[side] = {
                            'min': valid_angles.min(),
                            'max': valid_angles.max(),
                            'mean': valid_angles.mean(),
                            'range': valid_angles.max() - valid_angles.min(),
                            'std': valid_angles.std()
                        }
                    else:
                        joint_data[side] = {
                            'min': 0, 'max': 0, 'mean': 0, 'range': 0, 'std': 0
                        }
        else:
            # Process single side or center
            column_name = joint
            if column_name in angles_df.columns:
                valid_angles = angles_df[column_name].dropna()
                
                if len(valid_angles) > 0:
                    joint_data = {
                        'min': valid_angles.min(),
                        'max': valid_angles.max(),
                        'mean': valid_angles.mean(),
                        'range': valid_angles.max() - valid_angles.min(),
                        'std': valid_angles.std()
                    }
                else:
                    joint_data = {
                        'min': 0, 'max': 0, 'mean': 0, 'range': 0, 'std': 0
                    }
        
        # Store metrics for this joint
        metrics[joint] = joint_data
    
    return metrics


def compare_to_normal_rom(metrics, assessment_config):
    """
    Compare measured ROM to clinical norms
    
    Args:
        metrics: Dictionary of ROM metrics
        assessment_config: Assessment configuration with normal ROM values
    
    Returns:
        Dictionary with comparison results
    """
    comparison = {}
    
    normal_min = assessment_config['normal_min']
    normal_max = assessment_config['normal_max']
    normal_range = normal_max - normal_min
    
    for joint, data in metrics.items():
        # Check if we have data for both sides
        if isinstance(data, dict) and ('right' in data or 'left' in data):
            comparison[joint] = {}
            
            for side, side_data in data.items():
                # Calculate percentage of normal ROM
                achieved_range = side_data['range']
                percent_of_normal = (achieved_range / normal_range) * 100 if normal_range > 0 else 0
                
                # Determine limitation status
                if achieved_range >= normal_range * 0.9:
                    status = "Normal"
                elif achieved_range >= normal_range * 0.7:
                    status = "Mild limitation"
                elif achieved_range >= normal_range * 0.5:
                    status = "Moderate limitation"
                else:
                    status = "Severe limitation"
                
                comparison[joint][side] = {
                    'percent_of_normal': percent_of_normal,
                    'status': status,
                    'normal_min': normal_min,
                    'normal_max': normal_max,
                    'normal_range': normal_range,
                    'difference_from_normal': normal_range - achieved_range
                }
        else:
            # Single side or center joint
            achieved_range = data['range']
            percent_of_normal = (achieved_range / normal_range) * 100 if normal_range > 0 else 0
            
            if achieved_range >= normal_range * 0.9:
                status = "Normal"
            elif achieved_range >= normal_range * 0.7:
                status = "Mild limitation"
            elif achieved_range >= normal_range * 0.5:
                status = "Moderate limitation"
            else:
                status = "Severe limitation"
            
            comparison[joint] = {
                'percent_of_normal': percent_of_normal,
                'status': status,
                'normal_min': normal_min,
                'normal_max': normal_max,
                'normal_range': normal_range,
                'difference_from_normal': normal_range - achieved_range
            }
    
    return comparison


def save_metrics(metrics, assessment_type, output_path):
    """
    Save ROM metrics to a JSON file
    
    Args:
        metrics: Dictionary of ROM metrics
        assessment_type: Type of assessment performed
        output_path: Path to save metrics
    
    Returns:
        Path to saved metrics file
    """
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Format data for saving
    data = {
        'assessment_type': assessment_type,
        'metrics': metrics
    }
    
    # Convert numeric values to floats for JSON serialization
    def convert_to_float(obj):
        if isinstance(obj, dict):
            return {k: convert_to_float(v) for k, v in obj.items()}
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        else:
            return obj
    
    data = convert_to_float(data)
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    logging.info(f"ROM metrics saved to {output_path}")
    return output_path


def load_previous_assessment(file_path):
    """
    Load ROM metrics from a previous assessment
    
    Args:
        file_path: Path to previous assessment metrics file
    
    Returns:
        Dictionary of previous assessment metrics
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logging.error(f"Failed to load previous assessment: {e}")
        return None


def compare_assessments(current, previous):
    """
    Compare current assessment with a previous one
    
    Args:
        current: Current assessment metrics
        previous: Previous assessment metrics
    
    Returns:
        Dictionary with comparison results
    """
    if not previous:
        return None
    
    comparison = {
        'assessment_date': {
            'current': current.get('assessment_date', 'Current'),
            'previous': previous.get('assessment_date', 'Previous')
        },
        'joints': {}
    }
    
    # Compare metrics for each joint
    for joint, current_data in current.get('metrics', {}).items():
        if joint in previous.get('metrics', {}):
            prev_data = previous['metrics'][joint]
            
            # Check if we have data for both sides
            if isinstance(current_data, dict) and ('right' in current_data or 'left' in current_data):
                comparison['joints'][joint] = {}
                
                for side in ['right', 'left']:
                    if side in current_data and side in prev_data:
                        curr_range = current_data[side].get('range', 0)
                        prev_range = prev_data[side].get('range', 0)
                        
                        # Calculate improvement
                        difference = curr_range - prev_range
                        percent_change = (difference / prev_range) * 100 if prev_range > 0 else 0
                        
                        comparison['joints'][joint][side] = {
                            'current_range': curr_range,
                            'previous_range': prev_range,
                            'difference': difference,
                            'percent_change': percent_change,
                            'improved': difference > 0
                        }
            else:
                # Single side or center joint
                curr_range = current_data.get('range', 0)
                prev_range = prev_data.get('range', 0)
                
                difference = curr_range - prev_range
                percent_change = (difference / prev_range) * 100 if prev_range > 0 else 0
                
                comparison['joints'][joint] = {
                    'current_range': curr_range,
                    'previous_range': prev_range,
                    'difference': difference,
                    'percent_change': percent_change,
                    'improved': difference > 0
                }
    
    return comparison


def get_assessment_summary(metrics, assessment_config):
    """
    Generate a summary of the assessment results
    
    Args:
        metrics: Dictionary of ROM metrics
        assessment_config: Assessment configuration
    
    Returns:
        String with assessment summary
    """
    summary = []
    assessment_type = assessment_config['type']
    
    summary.append(f"ROM Assessment: {assessment_type}")
    summary.append("=" * 50)
    
    # Compare to normal values
    comparison = compare_to_normal_rom(metrics, assessment_config)
    
    for joint, data in comparison.items():
        summary.append(f"\nJoint: {joint.upper()}")
        
        # Check if we have data for both sides
        if isinstance(data, dict) and ('right' in data or 'left' in data):
            for side, side_data in data.items():
                rom_percentage = side_data['percent_of_normal']
                status = side_data['status']
                
                summary.append(f"  {side.capitalize()}:")
                summary.append(f"    Range of Motion: {metrics[joint][side]['range']:.1f}° " +
                              f"({rom_percentage:.1f}% of normal)")
                summary.append(f"    Status: {status}")
                summary.append(f"    Min: {metrics[joint][side]['min']:.1f}°, " +
                              f"Max: {metrics[joint][side]['max']:.1f}°")
        else:
            # Single side or center joint
            rom_percentage = data['percent_of_normal']
            status = data['status']
            
            summary.append(f"  Range of Motion: {metrics[joint]['range']:.1f}° " +
                          f"({rom_percentage:.1f}% of normal)")
            summary.append(f"  Status: {status}")
            summary.append(f"  Min: {metrics[joint]['min']:.1f}°, " +
                          f"Max: {metrics[joint]['max']:.1f}°")
    
    # Add recommendations based on status
    summary.append("\nRECOMMENDATIONS:")
    
    # Find the most limited joint
    most_limited = None
    lowest_percent = 100
    
    for joint, data in comparison.items():
        if isinstance(data, dict) and ('right' in data or 'left' in data):
            for side, side_data in data.items():
                if side_data['percent_of_normal'] < lowest_percent:
                    lowest_percent = side_data['percent_of_normal']
                    most_limited = f"{side} {joint}"
        else:
            if data['percent_of_normal'] < lowest_percent:
                lowest_percent = data['percent_of_normal']
                most_limited = joint
    
    if lowest_percent < 70:
        summary.append(f"- Focus on improving mobility in the {most_limited}.")
        summary.append("- Consider the following exercises:")
        
        # Suggest exercises based on the most limited joint
        if "shoulder" in most_limited.lower():
            summary.append("  * Pendulum exercises")
            summary.append("  * Wall slides")
            summary.append("  * Assisted shoulder stretches")
        elif "elbow" in most_limited.lower():
            summary.append("  * Active-assisted elbow flexion/extension")
            summary.append("  * Wrist weight exercises")
        elif "hip" in most_limited.lower():
            summary.append("  * Hip flexor stretches")
            summary.append("  * Gentle leg raises")
            summary.append("  * Seated marching")
        elif "knee" in most_limited.lower():
            summary.append("  * Heel slides")
            summary.append("  * Seated knee extensions")
            summary.append("  * Wall slides")
        elif "ankle" in most_limited.lower():
            summary.append("  * Ankle pumps")
            summary.append("  * Towel stretches")
            summary.append("  * Calf stretches")
    else:
        summary.append("- Maintenance exercises are recommended.")
        summary.append("- Continue with the current exercise program.")
    
    return "\n".join(summary)