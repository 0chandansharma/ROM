#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Skeletal Models

This module defines skeletal models for different pose estimation frameworks.
It maps keypoint names to their relationships in the skeletal structure.
"""

import logging
import numpy as np
from enum import Enum


class PoseModel(Enum):
    """Enum for supported pose models"""
    BODY_25 = "BODY_25"              # OpenPose
    COCO_17 = "COCO_17"              # COCO/OpenPose/YOLO/MMPose
    HALPE_26 = "HALPE_26"            # AlphaPose/MMPose
    BLAZEPOSE = "BLAZEPOSE"          # MediaPipe BlazePose
    WHOLE_BODY = "WHOLE_BODY"        # MMPose/OpenPose
    
    @classmethod
    def from_string(cls, model_name):
        """Convert string to enum value"""
        name = model_name.upper().replace(' ', '_')
        
        if name in ["BODY_WITH_FEET", "HALPE26", "HALPE_26"]:
            return cls.HALPE_26
        elif name in ["BODY", "COCO17", "COCO_17"]:
            return cls.COCO_17
        elif name in ["BODY25", "BODY_25"]:
            return cls.BODY_25
        elif name in ["MEDIAPIPE", "BLAZEPOSE"]:
            return cls.BLAZEPOSE
        elif name in ["WHOLEBODY", "WHOLE_BODY", "COCO_133"]:
            return cls.WHOLE_BODY
        else:
            logging.warning(f"Unknown model: {model_name}. Using HALPE_26 as default.")
            return cls.HALPE_26


# Skeletal connections for different pose models
POSE_CONNECTIONS = {
    PoseModel.HALPE_26: [
        # Torso
        ("Hip", "Neck"),
        ("Hip", "RHip"),
        ("Hip", "LHip"),
        ("Neck", "RShoulder"),
        ("Neck", "LShoulder"),
        ("Neck", "Head"),
        ("Head", "Nose"),
        
        # Right arm
        ("RShoulder", "RElbow"),
        ("RElbow", "RWrist"),
        
        # Left arm
        ("LShoulder", "LElbow"),
        ("LElbow", "LWrist"),
        
        # Right leg
        ("RHip", "RKnee"),
        ("RKnee", "RAnkle"),
        ("RAnkle", "RHeel"),
        ("RAnkle", "RBigToe"),
        ("RBigToe", "RSmallToe"),
        
        # Left leg
        ("LHip", "LKnee"),
        ("LKnee", "LAnkle"),
        ("LAnkle", "LHeel"),
        ("LAnkle", "LBigToe"),
        ("LBigToe", "LSmallToe"),
    ],
    
    PoseModel.COCO_17: [
        # Torso
        ("Neck", "RShoulder"),
        ("Neck", "LShoulder"),
        ("RShoulder", "RHip"),
        ("LShoulder", "LHip"),
        ("RHip", "LHip"),
        ("Neck", "Nose"),
        
        # Right arm
        ("RShoulder", "RElbow"),
        ("RElbow", "RWrist"),
        
        # Left arm
        ("LShoulder", "LElbow"),
        ("LElbow", "LWrist"),
        
        # Right leg
        ("RHip", "RKnee"),
        ("RKnee", "RAnkle"),
        
        # Left leg
        ("LHip", "LKnee"),
        ("LKnee", "LAnkle"),
    ],
    
    PoseModel.BODY_25: [
        # Torso
        ("CHip", "Neck"),
        ("CHip", "RHip"),
        ("CHip", "LHip"),
        ("Neck", "RShoulder"),
        ("Neck", "LShoulder"),
        ("Neck", "Nose"),
        
        # Right arm
        ("RShoulder", "RElbow"),
        ("RElbow", "RWrist"),
        
        # Left arm
        ("LShoulder", "LElbow"),
        ("LElbow", "LWrist"),
        
        # Right leg
        ("RHip", "RKnee"),
        ("RKnee", "RAnkle"),
        ("RAnkle", "RBigToe"),
        ("RAnkle", "RSmallToe"),
        ("RAnkle", "RHeel"),
        
        # Left leg
        ("LHip", "LKnee"),
        ("LKnee", "LAnkle"),
        ("LAnkle", "LBigToe"),
        ("LAnkle", "LSmallToe"),
        ("LAnkle", "LHeel"),
    ],
    
    PoseModel.BLAZEPOSE: [
        # Torso
        ("nose", "right_shoulder"),
        ("nose", "left_shoulder"),
        ("right_shoulder", "left_shoulder"),
        ("right_shoulder", "right_hip"),
        ("left_shoulder", "left_hip"),
        ("right_hip", "left_hip"),
        
        # Right arm
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("right_wrist", "right_index"),
        ("right_wrist", "right_pinky"),
        ("right_wrist", "right_thumb"),
        
        # Left arm
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("left_wrist", "left_index"),
        ("left_wrist", "left_pinky"),
        ("left_wrist", "left_thumb"),
        
        # Right leg
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
        ("right_ankle", "right_heel"),
        ("right_ankle", "right_foot_index"),
        
        # Left leg
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("left_ankle", "left_heel"),
        ("left_ankle", "left_foot_index"),
    ],
    
    PoseModel.WHOLE_BODY: [
        # Torso
        ("Hip", "Neck"),
        ("Hip", "RHip"),
        ("Hip", "LHip"),
        ("Neck", "RShoulder"),
        ("Neck", "LShoulder"),
        ("Neck", "Nose"),
        ("Nose", "REye"),
        ("Nose", "LEye"),
        ("REye", "REar"),
        ("LEye", "LEar"),
        
        # Right arm
        ("RShoulder", "RElbow"),
        ("RElbow", "RWrist"),
        ("RWrist", "RThumb"),
        ("RWrist", "RIndex"),
        ("RWrist", "RPinky"),
        
        # Left arm
        ("LShoulder", "LElbow"),
        ("LElbow", "LWrist"),
        ("LWrist", "LThumb"),
        ("LWrist", "LIndex"),
        ("LWrist", "LPinky"),
        
        # Right leg
        ("RHip", "RKnee"),
        ("RKnee", "RAnkle"),
        ("RAnkle", "RBigToe"),
        ("RAnkle", "RSmallToe"),
        ("RAnkle", "RHeel"),
        
        # Left leg
        ("LHip", "LKnee"),
        ("LKnee", "LAnkle"),
        ("LAnkle", "LBigToe"),
        ("LAnkle", "LSmallToe"),
        ("LAnkle", "LHeel"),
    ]
}


# Keypoint indices for different pose models
KEYPOINT_INDICES = {
    PoseModel.HALPE_26: {
        "Nose": 0,
        "LEye": 1,
        "REye": 2,
        "LEar": 3,
        "REar": 4,
        "LShoulder": 5,
        "RShoulder": 6,
        "LElbow": 7,
        "RElbow": 8,
        "LWrist": 9,
        "RWrist": 10,
        "LHip": 11,
        "RHip": 12,
        "LKnee": 13,
        "RKnee": 14,
        "LAnkle": 15,
        "RAnkle": 16,
        "Head": 17,
        "Neck": 18,
        "Hip": 19,
        "LBigToe": 20,
        "RBigToe": 21,
        "LSmallToe": 22,
        "RSmallToe": 23,
        "LHeel": 24,
        "RHeel": 25
    },
    
    PoseModel.COCO_17: {
        "Nose": 0,
        "LEye": 1,
        "REye": 2,
        "LEar": 3,
        "REar": 4,
        "LShoulder": 5,
        "RShoulder": 6,
        "LElbow": 7,
        "RElbow": 8,
        "LWrist": 9,
        "RWrist": 10,
        "LHip": 11,
        "RHip": 12,
        "LKnee": 13,
        "RKnee": 14,
        "LAnkle": 15,
        "RAnkle": 16,
        # Derived keypoints
        "Neck": None,  # Derived from shoulders
        "Hip": None    # Derived from hips
    },
    
    PoseModel.BODY_25: {
        "Nose": 0,
        "Neck": 1,
        "RShoulder": 2,
        "RElbow": 3,
        "RWrist": 4,
        "LShoulder": 5,
        "LElbow": 6,
        "LWrist": 7,
        "CHip": 8,
        "RHip": 9,
        "RKnee": 10,
        "RAnkle": 11,
        "LHip": 12,
        "LKnee": 13,
        "LAnkle": 14,
        "REye": 15,
        "LEye": 16,
        "REar": 17,
        "LEar": 18,
        "LBigToe": 19,
        "LSmallToe": 20,
        "LHeel": 21,
        "RBigToe": 22,
        "RSmallToe": 23,
        "RHeel": 24,
        # Derived keypoints
        "Hip": None    # Same as CHip
    },
    
    PoseModel.BLAZEPOSE: {
        "nose": 0,
        "left_eye_inner": 1,
        "left_eye": 2,
        "left_eye_outer": 3,
        "right_eye_inner": 4,
        "right_eye": 5,
        "right_eye_outer": 6,
        "left_ear": 7,
        "right_ear": 8,
        "mouth_left": 9,
        "mouth_right": 10,
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_wrist": 15,
        "right_wrist": 16,
        "left_pinky": 17,
        "right_pinky": 18,
        "left_index": 19,
        "right_index": 20,
        "left_thumb": 21,
        "right_thumb": 22,
        "left_hip": 23,
        "right_hip": 24,
        "left_knee": 25,
        "right_knee": 26,
        "left_ankle": 27,
        "right_ankle": 28,
        "left_heel": 29,
        "right_heel": 30,
        "left_foot_index": 31,
        "right_foot_index": 32
    }
}


def get_keypoint_index(keypoint_name, model=PoseModel.HALPE_26):
    """
    Get index of a keypoint for a specific model
    
    Args:
        keypoint_name: Name of the keypoint
        model: PoseModel enum value
    
    Returns:
        Index of the keypoint or None if not found
    """
    if model not in KEYPOINT_INDICES:
        logging.warning(f"Model {model} not found in keypoint indices. Using HALPE_26.")
        model = PoseModel.HALPE_26
    
    return KEYPOINT_INDICES[model].get(keypoint_name)


def get_connections(model=PoseModel.HALPE_26):
    """
    Get list of skeletal connections for a specific model
    
    Args:
        model: PoseModel enum value
    
    Returns:
        List of tuples representing connections between keypoints
    """
    if model not in POSE_CONNECTIONS:
        logging.warning(f"Model {model} not found in connections. Using HALPE_26.")
        model = PoseModel.HALPE_26
    
    return POSE_CONNECTIONS[model]


def get_derived_keypoints(keypoints, model=PoseModel.HALPE_26):
    """
    Calculate derived keypoints (e.g., neck from shoulders)
    
    Args:
        keypoints: Array of keypoints
        model: PoseModel enum value
    
    Returns:
        Updated keypoints array with derived points
    """
    if model == PoseModel.COCO_17:
        # Derive neck as midpoint between shoulders
        l_shoulder_idx = get_keypoint_index("LShoulder", model)
        r_shoulder_idx = get_keypoint_index("RShoulder", model)
        
        if l_shoulder_idx is not None and r_shoulder_idx is not None:
            l_shoulder = keypoints[l_shoulder_idx]
            r_shoulder = keypoints[r_shoulder_idx]
            
            if not (np.isnan(l_shoulder).any() or np.isnan(r_shoulder).any()):
                neck = (l_shoulder + r_shoulder) / 2
                # Add neck to keypoints
                keypoints = np.vstack([keypoints, neck])
        
        # Derive hip as midpoint between hips
        l_hip_idx = get_keypoint_index("LHip", model)
        r_hip_idx = get_keypoint_index("RHip", model)
        
        if l_hip_idx is not None and r_hip_idx is not None:
            l_hip = keypoints[l_hip_idx]
            r_hip = keypoints[r_hip_idx]
            
            if not (np.isnan(l_hip).any() or np.isnan(r_hip).any()):
                hip = (l_hip + r_hip) / 2
                # Add hip to keypoints
                keypoints = np.vstack([keypoints, hip])
    
    return keypoints


def get_joint_keypoints(joint_name, side="right", model=PoseModel.HALPE_26):
    """
    Get keypoints that define a specific joint angle
    
    Args:
        joint_name: Name of the joint (e.g., 'shoulder', 'knee')
        side: Side of the body ('right', 'left', or 'center')
        model: PoseModel enum value
    
    Returns:
        List of keypoint names that define the joint angle
    """
    joint_definitions = {
        "shoulder": {
            "right": ["RHip", "RShoulder", "RElbow"],
            "left": ["LHip", "LShoulder", "LElbow"]
        },
        "elbow": {
            "right": ["RShoulder", "RElbow", "RWrist"],
            "left": ["LShoulder", "LElbow", "LWrist"]
        },
        "wrist": {
            "right": ["RElbow", "RWrist", "RIndex" if model != PoseModel.COCO_17 else None],
            "left": ["LElbow", "LWrist", "LIndex" if model != PoseModel.COCO_17 else None]
        },
        "hip": {
            "right": ["RKnee", "RHip", "Neck"],
            "left": ["LKnee", "LHip", "Neck"]
        },
        "knee": {
            "right": ["RHip", "RKnee", "RAnkle"],
            "left": ["LHip", "LKnee", "LAnkle"]
        },
        "ankle": {
            "right": ["RKnee", "RAnkle", "RBigToe" if model != PoseModel.COCO_17 else None],
            "left": ["LKnee", "LAnkle", "LBigToe" if model != PoseModel.COCO_17 else None]
        },
        "neck": {
            "center": ["Head", "Neck", "Hip"]
        },
        "trunk": {
            "center": ["Neck", "Hip", None]
        }
    }
    
    if joint_name not in joint_definitions:
        logging.warning(f"Unknown joint: {joint_name}")
        return []
    
    joint_data = joint_definitions[joint_name]
    
    if side in joint_data:
        keypoints = joint_data[side]
        return [kpt for kpt in keypoints if kpt is not None]
    elif "center" in joint_data:
        keypoints = joint_data["center"]
        return [kpt for kpt in keypoints if kpt is not None]
    elif side == "both" and "right" in joint_data and "left" in joint_data:
        return {
            "right": [kpt for kpt in joint_data["right"] if kpt is not None],
            "left": [kpt for kpt in joint_data["left"] if kpt is not None]
        }
    else:
        logging.warning(f"Side {side} not available for joint {joint_name}")
        return []


def convert_keypoints_format(keypoints, source_model, target_model):
    """
    Convert keypoints from one model format to another
    
    Args:
        keypoints: Array of keypoints in source model format
        source_model: Source PoseModel enum value
        target_model: Target PoseModel enum value
    
    Returns:
        Keypoints array in target model format
    """
    if source_model == target_model:
        return keypoints
    
    # Create mapping between source and target indices
    mapping = {}
    
    for keypoint_name in KEYPOINT_INDICES[source_model]:
        source_idx = get_keypoint_index(keypoint_name, source_model)
        target_idx = get_keypoint_index(keypoint_name, target_model)
        
        if source_idx is not None and target_idx is not None:
            mapping[source_idx] = target_idx
    
    # Create new keypoints array for target model
    target_keypoints = np.full((len(KEYPOINT_INDICES[target_model]), 2), np.nan)
    
    # Copy keypoints from source to target positions
    for source_idx, target_idx in mapping.items():
        if source_idx < len(keypoints):
            target_keypoints[target_idx] = keypoints[source_idx]
    
    return target_keypoints