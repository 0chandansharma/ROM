#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Angle Calculation Utilities

This module contains functions for calculating joint angles from keypoints,
filtering low-confidence keypoints, and interpolating missing values.
"""

import numpy as np
import pandas as pd
import logging
import math

# Default configuration
DEFAULT_CONFIG = {
    'project': {
        'video_input': ['demo.mp4'],
        'assessment': 'shoulder_flexion',
        'px_to_m_from_person_id': 0,
        'px_to_m_person_height': 1.65,
        'visible_side': ['side', 'none'],
        'time_range': [],
        'video_dir': '',
        'webcam_id': 0,
        'input_size': [1280, 720]
    },
    'process': {
        'multiperson': False,
        'show_realtime_results': True,
        'save_vid': True,
        'save_img': True,
        'save_pose': True,
        'calculate_angles': True,
        'save_angles': True,
        'result_dir': ''
    },
    'pose': {
        'slowmo_factor': 1,
        'pose_model': 'body_with_feet',
        'mode': 'balanced',
        'det_frequency': 4,
        'device': 'auto',
        'backend': 'auto',
        'tracking_mode': 'rom',
        'keypoint_likelihood_threshold': 0.3,
        'average_likelihood_threshold': 0.5,
        'keypoint_number_threshold': 0.3
    },
    'assessment': {
        'display_angle_values': True,
        'fontSize': 0.5,
        'show_normal_rom': True,
        'show_progress_bar': True,
        'joint_angles': [
            'Right ankle',
            'Left ankle',
            'Right knee',
            'Left knee',
            'Right hip',
            'Left hip',
            'Right shoulder',
            'Left shoulder',
            'Right elbow',
            'Left elbow'
        ],
        'normal_rom_thresholds': {
            'shoulder_flexion': [0, 180],
            'shoulder_abduction': [0, 180],
            'elbow_flexion': [0, 150],
            'hip_flexion': [0, 120],
            'knee_extension': [0, 135],
            'ankle_dorsiflexion': [0, 20]
        }
    },
    'post-processing': {
        'interpolate': True,
        'interp_gap_smaller_than': 10,
        'fill_large_gaps_with': 'last_value',
        'filter': True,
        'show_graphs': True,
        'show_rom_metrics': True,
        'filter_type': 'butterworth',
        'butterworth': {'order': 4, 'cut_off_frequency': 6},
        'gaussian': {'sigma_kernel': 1},
        'loess': {'nb_values_used': 5},
        'median': {'kernel_size': 3}
    }
}


def filter_keypoints(keypoints, scores, threshold=0.3):
    """
    Filter keypoints based on confidence scores
    
    Args:
        keypoints: Detected keypoints array (shape: [num_persons, num_keypoints, 2])
        scores: Confidence scores array (shape: [num_persons, num_keypoints])
        threshold: Confidence threshold
    
    Returns:
        Filtered keypoints with low-confidence points set to NaN
    """
    if len(keypoints) == 0:
        return keypoints
    
    # Create a copy to avoid modifying the original
    filtered = keypoints.copy()
    
    # Set keypoints with low confidence to NaN
    for person_idx in range(len(keypoints)):
        for kpt_idx in range(len(keypoints[person_idx])):
            if scores[person_idx][kpt_idx] < threshold:
                filtered[person_idx][kpt_idx] = np.array([np.nan, np.nan])
    
    return filtered


def compute_joint_angles(keypoints, target_joints):
    """
    Calculate joint angles from keypoints
    
    Args:
        keypoints: Filtered keypoints array
        target_joints: List of joints to calculate angles for
    
    Returns:
        Dictionary of joint angles
    """
    if len(keypoints) == 0:
        return {}
    
    angles = {}
    
    # Get keypoint mappings from assessment module
    from ROM.assessment.movements import get_joint_keypoints
    
    # Calculate angle for each target joint
    for joint in target_joints:
        # Get keypoint connections for both sides
        joint_keypoints = get_joint_keypoints(joint, 'both')
        
        if not joint_keypoints:
            continue
        
        # Check if we have separate right/left sides
        if isinstance(joint_keypoints, dict):
            for side, side_keypoints in joint_keypoints.items():
                angle = calculate_angle_from_keypoints(keypoints[0], side_keypoints)
                if angle is not None:
                    angles[f"{side}_{joint}"] = angle
        else:
            # Single joint (e.g., neck, trunk)
            angle = calculate_angle_from_keypoints(keypoints[0], joint_keypoints)
            if angle is not None:
                angles[joint] = angle
    
    return angles


def calculate_angle_from_keypoints(keypoints, keypoint_names):
    """
    Calculate angle between three keypoints
    
    Args:
        keypoints: Array of keypoints
        keypoint_names: List of three keypoint names defining the angle
    
    Returns:
        Angle in degrees or None if calculation fails
    """
    # This is a simplified implementation - in a real implementation,
    # you would need a proper mapping between keypoint names and indices
    keypoint_mapping = {
        'Nose': 0,
        'Neck': 1,
        'RShoulder': 2,
        'RElbow': 3,
        'RWrist': 4,
        'LShoulder': 5,
        'LElbow': 6,
        'LWrist': 7,
        'RHip': 8,
        'RKnee': 9,
        'RAnkle': 10,
        'LHip': 11,
        'LKnee': 12,
        'LAnkle': 13,
        'REye': 14,
        'LEye': 15,
        'REar': 16,
        'LEar': 17,
        'Head': 1,  # Same as Neck for simplicity
        'Spine': 1,  # Same as Neck for simplicity
        'Hip': 8,  # Using RHip for simplicity
        'RFoot': 10,  # Same as RAnkle for simplicity
        'LFoot': 13,  # Same as LAnkle for simplicity
        'RHand': 4,  # Same as RWrist for simplicity
        'LHand': 7   # Same as LWrist for simplicity
    }
    
    try:
        # Need at least 3 points to define an angle
        if len(keypoint_names) < 3:
            return None
        
        # Get keypoint indices
        indices = [keypoint_mapping.get(name) for name in keypoint_names]
        
        # Check if any keypoint is missing
        if None in indices:
            missing = [name for name, idx in zip(keypoint_names, indices) if idx is None]
            logging.warning(f"Missing keypoints in mapping: {missing}")
            return None
        
        # Extract coordinates
        pts = [keypoints[idx] for idx in indices]
        
        # Check if any point is invalid (NaN)
        if any(np.isnan(pt).any() for pt in pts):
            return None
        
        # Calculate vectors
        v1 = pts[0] - pts[1]  # Vector from middle point to first point
        v2 = pts[2] - pts[1]  # Vector from middle point to third point
        
        # Calculate unit vectors
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm == 0 or v2_norm == 0:
            return None
        
        v1_unit = v1 / v1_norm
        v2_unit = v2 / v2_norm
        
        # Calculate angle using dot product
        dot_product = np.clip(np.dot(v1_unit, v2_unit), -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        angle_deg = np.degrees(angle_rad)
        
        # Determine the direction of rotation (clockwise/counterclockwise)
        cross_product = np.cross([v1[0], v1[1], 0], [v2[0], v2[1], 0])
        
        # For some joints, we want to measure the complementary angle
        complementary_joints = ['knee', 'elbow', 'hip', 'ankle']
        
        # For joints in this list, use the complementary angle (180 - angle)
        joint_name = keypoint_names[1].lower()
        if any(j in joint_name for j in complementary_joints):
            angle_deg = 180 - angle_deg
        
        # Apply specific angle adjustments
        if 'ankle' in joint_name:
            # Ankle dorsiflexion is typically measured from 90 degrees
            angle_deg = angle_deg - 90
        
        return angle_deg
        
    except (IndexError, ValueError) as e:
        logging.warning(f"Failed to calculate angle: {e}")
        return None


def calculate_segment_angle(keypoints, keypoint_names):
    """
    Calculate angle of a segment relative to horizontal
    
    Args:
        keypoints: Array of keypoints
        keypoint_names: List of two keypoint names defining the segment
    
    Returns:
        Angle in degrees or None if calculation fails
    """
    # This is a simplified implementation - in a real implementation,
    # you would need a proper mapping between keypoint names and indices
    keypoint_mapping = {
        'Nose': 0,
        'Neck': 1,
        'RShoulder': 2,
        'RElbow': 3,
        'RWrist': 4,
        'LShoulder': 5,
        'LElbow': 6,
        'LWrist': 7,
        'RHip': 8,
        'RKnee': 9,
        'RAnkle': 10,
        'LHip': 11,
        'LKnee': 12,
        'LAnkle': 13,
        'REye': 14,
        'LEye': 15,
        'REar': 16,
        'LEar': 17,
        'Head': 1,  # Same as Neck for simplicity
        'Spine': 1,  # Same as Neck for simplicity
        'Hip': 8,  # Using RHip for simplicity
        'RFoot': 10,  # Same as RAnkle for simplicity
        'LFoot': 13,  # Same as LAnkle for simplicity
        'RHand': 4,  # Same as RWrist for simplicity
        'LHand': 7   # Same as LWrist for simplicity
    }
    
    try:
        # Need exactly 2 points to define a segment
        if len(keypoint_names) != 2:
            return None
        
        # Get keypoint indices
        indices = [keypoint_mapping.get(name) for name in keypoint_names]
        
        # Check if any keypoint is missing
        if None in indices:
            missing = [name for name, idx in zip(keypoint_names, indices) if idx is None]
            logging.warning(f"Missing keypoints in mapping: {missing}")
            return None
        
        # Extract coordinates
        pts = [keypoints[idx] for idx in indices]
        
        # Check if any point is invalid (NaN)
        if any(np.isnan(pt).any() for pt in pts):
            return None
        
        # Calculate vector of segment
        segment = pts[1] - pts[0]
        
        # Calculate angle with horizontal (positive x-axis)
        angle_rad = np.arctan2(segment[1], segment[0])  # angle in [-pi, pi]
        angle_deg = np.degrees(angle_rad)
        
        # Convert to [0, 360) range
        if angle_deg < 0:
            angle_deg += 360
        
        return angle_deg
        
    except (IndexError, ValueError) as e:
        logging.warning(f"Failed to calculate segment angle: {e}")
        return None


def interpolate_missing_points(data, max_gap_size=10, fill_method='last_value'):
    """
    Interpolate missing points in angle data
    
    Args:
        data: DataFrame with angle data
        max_gap_size: Maximum gap size to interpolate
        fill_method: Method to fill large gaps ('last_value', 'nan', 'zeros')
    
    Returns:
        DataFrame with interpolated data
    """
    # Create a copy to avoid modifying original data
    interpolated_data = data.copy()
    
    for col in interpolated_data.columns:
        # Get mask of valid values
        mask = ~pd.isna(interpolated_data[col])
        
        # Find indices of valid values
        valid_indices = np.where(mask)[0]
        
        if len(valid_indices) < 2:
            # Not enough valid points for interpolation
            continue
        
        # Find gaps (sequences of missing values)
        gaps = np.split(np.arange(len(interpolated_data)), valid_indices)
        
        # Remove the first gap if it doesn't start with 0
        if valid_indices[0] != 0:
            gaps = gaps[1:]
        
        # Process each gap
        for gap in gaps:
            if len(gap) == 0:
                continue
            
            # Skip if gap is too large
            if len(gap) > max_gap_size:
                if fill_method == 'last_value':
                    # Fill with last valid value
                    last_valid = valid_indices[valid_indices < gap[0]]
                    if len(last_valid) > 0:
                        fill_value = interpolated_data.loc[last_valid[-1], col]
                        interpolated_data.loc[gap, col] = fill_value
                elif fill_method == 'zeros':
                    # Fill with zeros
                    interpolated_data.loc[gap, col] = 0
                # 'nan' method: leave as NaN
                continue
            
            # Get surrounding valid indices
            prev_idx = valid_indices[valid_indices < gap[0]][-1] if any(valid_indices < gap[0]) else gap[0]
            next_idx = valid_indices[valid_indices > gap[-1]][0] if any(valid_indices > gap[-1]) else gap[-1]
            
            # Get values at surrounding indices
            prev_val = interpolated_data.loc[prev_idx, col]
            next_val = interpolated_data.loc[next_idx, col]
            
            # Linear interpolation for the gap
            for i, idx in enumerate(gap):
                # Calculate interpolation factor
                alpha = i / len(gap)
                # Interpolate value
                interpolated_data.loc[idx, col] = prev_val * (1 - alpha) + next_val * alpha
    
    return interpolated_data


def calculate_rom(angles):
    """
    Calculate Range of Motion from angle time series
    
    Args:
        angles: Time series of angle measurements
    
    Returns:
        Dictionary with ROM metrics (min, max, range, etc.)
    """
    # Filter out NaN values
    valid_angles = angles[~np.isnan(angles)]
    
    if len(valid_angles) == 0:
        return {
            'min': None,
            'max': None,
            'range': None,
            'mean': None,
            'std': None
        }
    
    # Calculate ROM metrics
    min_angle = np.min(valid_angles)
    max_angle = np.max(valid_angles)
    mean_angle = np.mean(valid_angles)
    std_angle = np.std(valid_angles)
    
    # Range of motion
    rom = max_angle - min_angle
    
    return {
        'min': min_angle,
        'max': max_angle,
        'range': rom,
        'mean': mean_angle,
        'std': std_angle
    }


def detect_movement_phases(angles, threshold=5.0, min_duration=5):
    """
    Detect movement phases (increasing, decreasing, plateau)
    
    Args:
        angles: Time series of angle measurements
        threshold: Angular velocity threshold (degrees/frame)
        min_duration: Minimum duration of a phase in frames
    
    Returns:
        List of dictionaries describing detected phases
    """
    # Calculate angular velocity
    velocity = np.gradient(angles)
    
    # Initialize phases
    phases = []
    current_phase = None
    phase_start = 0
    
    # Detect phases based on velocity
    for i, v in enumerate(velocity):
        if np.isnan(v):
            continue
        
        if v > threshold:
            phase_type = "increasing"
        elif v < -threshold:
            phase_type = "decreasing"
        else:
            phase_type = "plateau"
        
        # Check if phase has changed
        if current_phase != phase_type:
            # Record previous phase if it was long enough
            if current_phase and i - phase_start >= min_duration:
                phases.append({
                    "type": current_phase,
                    "start": phase_start,
                    "end": i - 1,
                    "duration": i - phase_start,
                    "start_angle": angles[phase_start],
                    "end_angle": angles[i - 1],
                    "change": angles[i - 1] - angles[phase_start]
                })
            
            # Start new phase
            current_phase = phase_type
            phase_start = i
    
    # Add final phase
    if current_phase and len(angles) - phase_start >= min_duration:
        phases.append({
            "type": current_phase,
            "start": phase_start,
            "end": len(angles) - 1,
            "duration": len(angles) - phase_start,
            "start_angle": angles[phase_start],
            "end_angle": angles[-1],
            "change": angles[-1] - angles[phase_start]
        })
    
    return phases