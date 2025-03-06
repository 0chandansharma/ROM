#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Physiotherapy Movement Assessments

This module defines the available physiotherapy assessments and their
specific configurations, including which joints to track, normal range
of motion values, and visualization settings.
"""

import logging

# List of available assessments
AVAILABLE_ASSESSMENTS = [
    'shoulder_flexion',
    'shoulder_abduction',
    'elbow_flexion',
    'hip_flexion',
    'knee_extension',
    'ankle_dorsiflexion',
    'neck_rotation',
    'wrist_flexion',
    'trunk_flexion',
    'general'  # For multi-joint assessment
]

# Clinical normal ROM values (in degrees)
# Source: American Academy of Orthopaedic Surgeons
NORMAL_ROM = {
    'shoulder_flexion': {
        'min': 0,
        'max': 180,
        'target_joints': ['shoulder'],
        'primary_side': 'both',
        'description': 'Forward elevation of the arm'
    },
    'shoulder_abduction': {
        'min': 0,
        'max': 180,
        'target_joints': ['shoulder'],
        'primary_side': 'both',
        'description': 'Lateral elevation of the arm'
    },
    'elbow_flexion': {
        'min': 0,
        'max': 150,
        'target_joints': ['elbow'],
        'primary_side': 'both',
        'description': 'Bending of the elbow'
    },
    'hip_flexion': {
        'min': 0,
        'max': 120,
        'target_joints': ['hip'],
        'primary_side': 'both',
        'description': 'Forward movement of the leg at the hip'
    },
    'knee_extension': {
        'min': 0,
        'max': 135,
        'target_joints': ['knee'],
        'primary_side': 'both',
        'description': 'Straightening of the knee'
    },
    'ankle_dorsiflexion': {
        'min': 0,
        'max': 20,
        'target_joints': ['ankle'],
        'primary_side': 'both',
        'description': 'Upward movement of the foot'
    },
    'neck_rotation': {
        'min': 0,
        'max': 80,
        'target_joints': ['neck'],
        'primary_side': 'both',
        'description': 'Turning the head side to side'
    },
    'wrist_flexion': {
        'min': 0,
        'max': 80,
        'target_joints': ['wrist'],
        'primary_side': 'both',
        'description': 'Bending the wrist'
    },
    'trunk_flexion': {
        'min': 0,
        'max': 80,
        'target_joints': ['trunk'],
        'primary_side': 'both',
        'description': 'Forward bending of the torso'
    },
    'general': {
        'min': 0,
        'max': 0,  # Will be determined per joint
        'target_joints': ['shoulder', 'elbow', 'wrist', 'hip', 'knee', 'ankle'],
        'primary_side': 'both',
        'description': 'General assessment of multiple joints'
    }
}

# Mapping from joint names to keypoint connections
JOINT_KEYPOINTS = {
    'shoulder': {
        'right': ['RHip', 'RShoulder', 'RElbow'],
        'left': ['LHip', 'LShoulder', 'LElbow']
    },
    'elbow': {
        'right': ['RShoulder', 'RElbow', 'RWrist'],
        'left': ['LShoulder', 'LElbow', 'LWrist']
    },
    'wrist': {
        'right': ['RElbow', 'RWrist', 'RHand'],
        'left': ['LElbow', 'LWrist', 'LHand']
    },
    'hip': {
        'right': ['RKnee', 'RHip', 'Spine'],
        'left': ['LKnee', 'LHip', 'Spine']
    },
    'knee': {
        'right': ['RHip', 'RKnee', 'RAnkle'],
        'left': ['LHip', 'LKnee', 'LAnkle']
    },
    'ankle': {
        'right': ['RKnee', 'RAnkle', 'RFoot'],
        'left': ['LKnee', 'LAnkle', 'LFoot']
    },
    'neck': {
        'center': ['Head', 'Neck', 'Spine']
    },
    'trunk': {
        'center': ['Neck', 'Spine', 'Hip']
    }
}

# Color schemes for different ranges
# Green: Within normal range
# Yellow: Slightly outside normal range
# Red: Significantly outside normal range
COLOR_SCHEMES = {
    'normal': (0, 255, 0),       # Green
    'borderline': (0, 255, 255),  # Yellow
    'limited': (0, 0, 255)        # Red
}


def get_assessment_config(assessment_type, config):
    """
    Get the configuration for a specific assessment type
    
    Args:
        assessment_type: Type of assessment to perform
        config: Base configuration dictionary
    
    Returns:
        Configuration dictionary for the specific assessment
    """
    # Check if assessment type is valid
    if assessment_type not in AVAILABLE_ASSESSMENTS:
        logging.warning(f"Unknown assessment type: {assessment_type}. Defaulting to 'general'.")
        assessment_type = 'general'
    
    # Get normal ROM values
    normal_rom = NORMAL_ROM.get(assessment_type, NORMAL_ROM['general'])
    
    # Get user-defined thresholds if available
    user_thresholds = config.get('assessment', {}).get('normal_rom_thresholds', {}).get(assessment_type)
    
    if user_thresholds:
        normal_rom['min'] = user_thresholds[0]
        normal_rom['max'] = user_thresholds[1]
    
    # Create assessment configuration
    assessment_config = {
        'type': assessment_type,
        'description': normal_rom['description'],
        'target_joints': normal_rom['target_joints'],
        'primary_side': normal_rom['primary_side'],
        'normal_min': normal_rom['min'],
        'normal_max': normal_rom['max'],
        'keypoint_threshold': config.get('pose', {}).get('keypoint_likelihood_threshold', 0.3),
        'display_options': {
            'show_angles': config.get('assessment', {}).get('display_angle_values', True),
            'font_size': config.get('assessment', {}).get('fontSize', 0.5),
            'show_normal_range': config.get('assessment', {}).get('show_normal_rom', True),
            'show_progress_bar': config.get('assessment', {}).get('show_progress_bar', True),
            'colors': COLOR_SCHEMES
        },
        'keypoint_mappings': {}
    }
    
    # Add keypoint mappings for targeted joints
    for joint in normal_rom['target_joints']:
        if joint in JOINT_KEYPOINTS:
            assessment_config['keypoint_mappings'][joint] = JOINT_KEYPOINTS[joint]
    
    return assessment_config


def get_joint_keypoints(joint_name, side='right'):
    """
    Get the keypoint connections for a specific joint
    
    Args:
        joint_name: Name of the joint (e.g., 'shoulder', 'knee')
        side: 'right', 'left', or 'center'
    
    Returns:
        List of keypoints that define the joint angle
    """
    if joint_name not in JOINT_KEYPOINTS:
        logging.warning(f"Unknown joint: {joint_name}")
        return []
    
    joint_data = JOINT_KEYPOINTS[joint_name]
    
    if side in joint_data:
        return joint_data[side]
    elif 'center' in joint_data:
        return joint_data['center']
    elif side == 'both' and 'right' in joint_data and 'left' in joint_data:
        return {'right': joint_data['right'], 'left': joint_data['left']}
    else:
        logging.warning(f"Side {side} not available for joint {joint_name}")
        return []


def is_rom_normal(angle, assessment_type, joint_name):
    """
    Check if the measured angle is within normal ROM
    
    Args:
        angle: Measured angle in degrees
        assessment_type: Type of assessment
        joint_name: Name of the joint
    
    Returns:
        Status string: 'normal', 'borderline', or 'limited'
    """
    rom_info = NORMAL_ROM.get(assessment_type)
    
    if not rom_info:
        return 'normal'  # Default if unknown
    
    min_angle = rom_info['min']
    max_angle = rom_info['max']
    
    # Define thresholds for borderline (80% of normal range)
    borderline_min = min_angle - (max_angle - min_angle) * 0.2
    borderline_max = max_angle + (max_angle - min_angle) * 0.2
    
    if min_angle <= angle <= max_angle:
        return 'normal'
    elif borderline_min <= angle <= borderline_max:
        return 'borderline'
    else:
        return 'limited'


def get_movement_instructions(assessment_type):
    """
    Get patient instructions for a specific assessment
    
    Args:
        assessment_type: Type of assessment
    
    Returns:
        String with instructions for performing the movement
    """
    instructions = {
        'shoulder_flexion': 
            "Raise your arm straight in front of you as high as possible, keeping your elbow straight.",
        'shoulder_abduction': 
            "Raise your arm out to the side as high as possible, keeping your elbow straight.",
        'elbow_flexion': 
            "Bend your elbow, bringing your hand towards your shoulder as far as possible.",
        'hip_flexion': 
            "Lift your knee towards your chest as high as comfortable while keeping your back straight.",
        'knee_extension': 
            "Sit with your leg bent, then straighten your knee as much as possible.",
        'ankle_dorsiflexion': 
            "Pull your foot upwards towards your shin as far as possible.",
        'neck_rotation': 
            "Turn your head to look over your shoulder as far as comfortable, then to the other side.",
        'wrist_flexion': 
            "Bend your wrist downward as far as possible, then upward.",
        'trunk_flexion': 
            "Stand straight, then bend forward at the waist as far as comfortable.",
        'general': 
            "Perform a series of movements as directed by your therapist."
    }
    
    return instructions.get(
        assessment_type, 
        "Follow your therapist's instructions for this assessment."
    )