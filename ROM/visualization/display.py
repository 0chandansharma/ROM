#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization Display Module

This module handles the real-time visualization of joint angles
and range of motion feedback during assessments.
"""

import cv2
import numpy as np
import math
from datetime import datetime


def create_visualization(frame, keypoints, joint_angles, assessment_config):
    """
    Create visualization of joint angles and pose
    
    Args:
        frame: Input video frame
        keypoints: Detected keypoints
        joint_angles: Calculated joint angles
        assessment_config: Assessment configuration
    
    Returns:
        Frame with visualization overlays
    """
    # Create a copy of the frame for drawing
    viz_frame = frame.copy()
    
    # Draw skeleton and keypoints
    viz_frame = draw_skeleton(viz_frame, keypoints, assessment_config)
    
    # Draw joint angles
    viz_frame = draw_joint_angles(viz_frame, keypoints, joint_angles, assessment_config)
    
    # Add assessment name and timestamp
    assessment_type = assessment_config['type']
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Add header with assessment info
    cv2.rectangle(viz_frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(viz_frame, f"Assessment: {assessment_type}", (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(viz_frame, timestamp, (frame.shape[1] - 200, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return viz_frame


def draw_skeleton(frame, keypoints, assessment_config):
    """
    Draw skeleton connections between keypoints
    
    Args:
        frame: Input video frame
        keypoints: Detected keypoints
        assessment_config: Assessment configuration
    
    Returns:
        Frame with skeleton overlay
    """
    if len(keypoints) == 0:
        return frame
    
    # Get keypoint mappings for the assessment
    keypoint_mappings = assessment_config.get('keypoint_mappings', {})
    
    # Define colors for different joint types
    colors = {
        'target': (0, 255, 0),    # Green for target joints
        'other': (255, 255, 255)  # White for other joints
    }
    
    thickness = 2
    
    # Draw connections for all targeted joints
    for joint_name, joint_data in keypoint_mappings.items():
        for side, keypoint_names in joint_data.items():
            # Check if we have at least 2 keypoints to connect
            if len(keypoint_names) < 2:
                continue
            
            # Get keypoint indices (assuming keypoints are in the same order)
            keypoint_indices = [get_keypoint_index(name) for name in keypoint_names]
            
            # Draw lines between keypoints
            for i in range(len(keypoint_indices) - 1):
                start_idx = keypoint_indices[i]
                end_idx = keypoint_indices[i + 1]
                
                if start_idx is not None and end_idx is not None:
                    try:
                        start_pt = tuple(map(int, keypoints[0][start_idx]))
                        end_pt = tuple(map(int, keypoints[0][end_idx]))
                        
                        # Check if keypoints are valid
                        if not (np.isnan(start_pt).any() or np.isnan(end_pt).any()):
                            color = colors['target'] if joint_name in assessment_config['target_joints'] else colors['other']
                            cv2.line(frame, start_pt, end_pt, color, thickness)
                    except (IndexError, ValueError):
                        continue
    
    # Draw keypoints
    for i in range(len(keypoints[0])):
        try:
            pt = tuple(map(int, keypoints[0][i]))
            if not np.isnan(pt).any():
                cv2.circle(frame, pt, 5, (0, 0, 255), -1)
        except (IndexError, ValueError):
            continue
    
    return frame


def draw_joint_angles(frame, keypoints, joint_angles, assessment_config):
    """
    Draw joint angles on the frame
    
    Args:
        frame: Input video frame
        keypoints: Detected keypoints
        joint_angles: Calculated joint angles
        assessment_config: Assessment configuration
    
    Returns:
        Frame with joint angle overlays
    """
    if not assessment_config['display_options']['show_angles']:
        return frame
    
    font_size = assessment_config['display_options'].get('font_size', 0.5)
    thickness = 1
    
    # Get keypoint mappings for the assessment
    keypoint_mappings = assessment_config.get('keypoint_mappings', {})
    
    # Draw angle for each targeted joint
    for joint_name, value in joint_angles.items():
        if joint_name in keypoint_mappings:
            for side, keypoint_names in keypoint_mappings[joint_name].items():
                # We need at least 3 points to define an angle
                if len(keypoint_names) < 3:
                    continue
                
                angle_key = f"{side}_{joint_name}" if side in ['right', 'left'] else joint_name
                angle = joint_angles.get(angle_key)
                
                if angle is None:
                    continue
                
                # Get keypoint indices
                keypoint_indices = [get_keypoint_index(name) for name in keypoint_names]
                if None in keypoint_indices or len(keypoint_indices) < 3:
                    continue
                
                # Get middle keypoint (vertex of the angle)
                try:
                    vertex_idx = keypoint_indices[1]
                    vertex_pt = tuple(map(int, keypoints[0][vertex_idx]))
                    
                    # Skip if vertex point is not valid
                    if np.isnan(vertex_pt).any():
                        continue
                    
                    # Get angle status (normal, borderline, limited)
                    status = get_angle_status(angle, assessment_config)
                    color = assessment_config['display_options']['colors'].get(status, (255, 255, 255))
                    
                    # Draw angle text
                    text_pos = (vertex_pt[0] + 10, vertex_pt[1] - 10)
                    cv2.putText(frame, f"{angle:.1f}°", text_pos, 
                                cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), thickness + 1)
                    cv2.putText(frame, f"{angle:.1f}°", text_pos, 
                                cv2.FONT_HERSHEY_SIMPLEX, font_size, color, thickness)
                    
                    # Draw angle arc
                    draw_angle_arc(frame, keypoints[0], keypoint_indices, color, thickness)
                    
                except (IndexError, ValueError):
                    continue
    
    return frame


def draw_angle_arc(frame, keypoints, keypoint_indices, color, thickness=1):
    """
    Draw an arc representing the measured angle
    
    Args:
        frame: Input video frame
        keypoints: Array of keypoints
        keypoint_indices: Indices of keypoints forming the angle
        color: Color of the arc
        thickness: Line thickness
    """
    try:
        # Get the three points that form the angle
        pt1 = tuple(map(int, keypoints[keypoint_indices[0]]))
        vertex = tuple(map(int, keypoints[keypoint_indices[1]]))
        pt2 = tuple(map(int, keypoints[keypoint_indices[2]]))
        
        # Check if any point is invalid
        if np.isnan(pt1).any() or np.isnan(vertex).any() or np.isnan(pt2).any():
            return
        
        # Calculate vectors
        v1 = np.array([pt1[0] - vertex[0], pt1[1] - vertex[1]])
        v2 = np.array([pt2[0] - vertex[0], pt2[1] - vertex[1]])
        
        # Calculate unit vectors
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm == 0 or v2_norm == 0:
            return
        
        v1_unit = v1 / v1_norm
        v2_unit = v2 / v2_norm
        
        # Calculate angle in radians
        dot_product = np.clip(np.dot(v1_unit, v2_unit), -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        
        # Calculate start and end angles for arc
        start_angle = np.arctan2(v1[1], v1[0])
        end_angle = np.arctan2(v2[1], v2[0])
        
        # Ensure correct arc direction
        if end_angle < start_angle:
            end_angle += 2 * np.pi
        
        # Convert to degrees for OpenCV
        start_angle_deg = np.degrees(start_angle)
        end_angle_deg = np.degrees(end_angle)
        
        # Draw the arc
        radius = int(min(v1_norm, v2_norm) / 3)
        cv2.ellipse(frame, vertex, (radius, radius), 0, 
                    start_angle_deg, end_angle_deg, color, thickness)
    
    except (ValueError, ZeroDivisionError):
        pass


def draw_assessment_ui(frame, joint_angles, assessment_config, frame_count):
    """
    Draw assessment UI elements
    
    Args:
        frame: Input video frame
        joint_angles: Calculated joint angles
        assessment_config: Assessment configuration
        frame_count: Current frame number
    
    Returns:
        Frame with UI elements
    """
    # Add instructional text
    instructions = get_instructions(assessment_config['type'])
    cv2.putText(frame, instructions, (10, frame.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(frame, instructions, (10, frame.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Add ROM feedback sidebar if enabled
    if assessment_config['display_options']['show_normal_range']:
        frame = draw_rom_feedback_sidebar(frame, joint_angles, assessment_config)
    
    # Add progress bar if enabled
    if assessment_config['display_options']['show_progress_bar']:
        frame = draw_progress_bar(frame, frame_count)
    
    return frame


def draw_rom_feedback_sidebar(frame, joint_angles, assessment_config):
    """
    Draw ROM feedback sidebar
    
    Args:
        frame: Input video frame
        joint_angles: Calculated joint angles
        assessment_config: Assessment configuration
    
    Returns:
        Frame with ROM feedback sidebar
    """
    # Set sidebar dimensions
    sidebar_width = 200
    sidebar_height = frame.shape[0]
    sidebar_x = frame.shape[1] - sidebar_width
    
    # Create semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (sidebar_x, 0), (frame.shape[1], sidebar_height), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Add title
    cv2.putText(frame, "ROM Feedback", (sidebar_x + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw ROM gauge for each joint
    y_offset = 70
    
    for joint_name in assessment_config['target_joints']:
        # Check if we have both right and left sides
        if joint_name in assessment_config['keypoint_mappings']:
            sides = assessment_config['keypoint_mappings'][joint_name].keys()
            
            for side in sides:
                angle_key = f"{side}_{joint_name}" if side in ['right', 'left'] else joint_name
                angle = joint_angles.get(angle_key)
                
                if angle is not None:
                    # Draw joint label
                    joint_label = f"{side.capitalize()} {joint_name.capitalize()}"
                    cv2.putText(frame, joint_label, (sidebar_x + 10, y_offset), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Draw angle value
                    cv2.putText(frame, f"{angle:.1f}°", (sidebar_x + sidebar_width - 50, y_offset), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Draw ROM gauge
                    y_offset = draw_rom_gauge(frame, angle, assessment_config, sidebar_x, y_offset + 20)
                    y_offset += 30
    
    return frame


def draw_rom_gauge(frame, angle, assessment_config, x, y):
    """
    Draw ROM gauge showing current angle relative to normal range
    
    Args:
        frame: Input video frame
        angle: Current angle value
        assessment_config: Assessment configuration
        x: X position of gauge
        y: Y position of gauge
    
    Returns:
        New Y position after drawing gauge
    """
    gauge_width = 180
    gauge_height = 10
    
    normal_min = assessment_config['normal_min']
    normal_max = assessment_config['normal_max']
    
    # Draw gauge background
    cv2.rectangle(frame, (x + 10, y), (x + 10 + gauge_width, y + gauge_height), (100, 100, 100), -1)
    
    # Calculate position for normal range markers
    normal_min_x = x + 10 + int((normal_min / 180) * gauge_width)
    normal_max_x = x + 10 + int((normal_max / 180) * gauge_width)
    
    # Draw normal range zone
    cv2.rectangle(frame, (normal_min_x, y), (normal_max_x, y + gauge_height), (0, 255, 0), -1)
    
    # Calculate position for current angle marker
    angle_x = x + 10 + int((min(max(angle, 0), 180) / 180) * gauge_width)
    
    # Draw angle marker
    cv2.circle(frame, (angle_x, y + gauge_height // 2), 6, (0, 0, 255), -1)
    
    # Draw min/max labels
    cv2.putText(frame, "0°", (x + 5, y + gauge_height + 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(frame, "180°", (x + gauge_width - 5, y + gauge_height + 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    return y + gauge_height + 20


def draw_progress_bar(frame, frame_count, max_frames=300):
    """
    Draw progress bar at the bottom of the frame
    
    Args:
        frame: Input video frame
        frame_count: Current frame number
        max_frames: Maximum number of frames
    
    Returns:
        Frame with progress bar
    """
    bar_height = 5
    bar_y = frame.shape[0] - bar_height - 5
    
    # Draw background
    cv2.rectangle(frame, (0, bar_y), (frame.shape[1], bar_y + bar_height), (100, 100, 100), -1)
    
    # Calculate progress
    progress = min(frame_count / max_frames, 1.0)
    progress_width = int(progress * frame.shape[1])
    
    # Draw progress
    cv2.rectangle(frame, (0, bar_y), (progress_width, bar_y + bar_height), (0, 255, 0), -1)
    
    return frame


def show_rom_feedback(frame, angle, joint_name, assessment_config):
    """
    Add visual feedback about ROM status
    
    Args:
        frame: Input video frame
        angle: Current angle value
        joint_name: Name of the joint
        assessment_config: Assessment configuration
    
    Returns:
        Frame with ROM feedback
    """
    status = get_angle_status(angle, assessment_config)
    color = assessment_config['display_options']['colors'].get(status, (255, 255, 255))
    
    # Create a status indicator in the corner
    indicator_size = 30
    cv2.rectangle(frame, (10, 50), (10 + indicator_size, 50 + indicator_size), color, -1)
    
    # Add status text
    status_text = status.capitalize()
    cv2.putText(frame, status_text, (50, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Add guidance text based on status
    if status == 'normal':
        guidance = "Great! Maintain this range of motion."
    elif status == 'borderline':
        guidance = "Almost there! Try to increase your range slightly."
    else:  # limited
        guidance = "Keep working on increasing your range of motion."
    
    cv2.putText(frame, guidance, (10, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame


def get_keypoint_index(keypoint_name):
    """
    Get index of keypoint based on name
    
    Args:
        keypoint_name: Name of the keypoint
    
    Returns:
        Index of the keypoint or None if not found
    """
    # This is a simplified mapping - in a real implementation,
    # this would need to match the pose estimation model's output format
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
        'Hip': 8,  # Same as RHip for simplicity
        'RFoot': 10,  # Same as RAnkle for simplicity
        'LFoot': 13,  # Same as LAnkle for simplicity
        'RHand': 4,  # Same as RWrist for simplicity
        'LHand': 7   # Same as LWrist for simplicity
    }
    
    return keypoint_mapping.get(keypoint_name)


def get_angle_status(angle, assessment_config):
    """
    Determine status of angle relative to normal range
    
    Args:
        angle: Current angle value
        assessment_config: Assessment configuration
    
    Returns:
        Status string: 'normal', 'borderline', or 'limited'
    """
    normal_min = assessment_config['normal_min']
    normal_max = assessment_config['normal_max']
    
    # Define buffer zone for borderline (10% of normal range)
    buffer = (normal_max - normal_min) * 0.1
    
    if normal_min - buffer <= angle <= normal_max + buffer:
        return 'normal'
    elif normal_min - buffer * 2 <= angle <= normal_max + buffer * 2:
        return 'borderline'
    else:
        return 'limited'


def get_instructions(assessment_type):
    """
    Get instructional text for the assessment
    
    Args:
        assessment_type: Type of assessment
    
    Returns:
        Instruction text string
    """
    instructions = {
        'shoulder_flexion': "Raise arm forward and upward",
        'shoulder_abduction': "Raise arm sideways and upward",
        'elbow_flexion': "Bend and straighten your elbow",
        'hip_flexion': "Lift your leg forward",
        'knee_extension': "Straighten your knee",
        'ankle_dorsiflexion': "Point foot upward toward shin",
        'general': "Follow therapist instructions"
    }
    
    return instructions.get(assessment_type, "Perform the movement as instructed")