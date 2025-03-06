#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ROM Analysis Processor

This module contains the core functionality for processing video input,
detecting poses, calculating joint angles, and determining range of motion
for physiotherapy assessments.
"""

import logging
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Import relevant pose estimation library
try:
    from rtmlib import PoseTracker, BodyWithFeet, Body, Wholebody
    POSE_ENGINE = 'rtmlib'
except ImportError:
    try:
        import mediapipe as mp
        POSE_ENGINE = 'mediapipe'
    except ImportError:
        logging.error("No supported pose estimation library found. Please install rtmlib or mediapipe.")
        raise

from ROM.assessment.movements import get_assessment_config
from ROM.assessment.metrics import calculate_rom_metrics
from ROM.visualization.display import (
    create_visualization, 
    draw_assessment_ui,
    show_rom_feedback
)
from ROM.utils.angle_calc import (
    compute_joint_angles,
    filter_keypoints,
    interpolate_missing_points
)
from ROM.utils.filtering import filter_signal
from ROM.visualization.reporting import generate_assessment_report


def setup_video_source(config, video_path):
    """
    Initialize video capture from file or webcam
    
    Args:
        config: Configuration dictionary
        video_path: Path to video file or "webcam"
        
    Returns:
        Video capture object, frame dimensions, and frame rate
    """
    if video_path == "webcam":
        webcam_id = config.get('project', {}).get('webcam_id', 0)
        input_size = config.get('project', {}).get('input_size', [1280, 720])
        
        # Initialize webcam
        cap = cv2.VideoCapture(webcam_id)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open webcam with ID {webcam_id}")
        
        # Set resolution if specified
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, input_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, input_size[1])
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        if fps == 0:  # Sometimes webcams don't report FPS correctly
            fps = 30
        
        frame_count = float('inf')  # Webcam has unlimited frames
        
    else:
        # Initialize video file
        video_dir = config.get('project', {}).get('video_dir', '')
        if video_dir:
            video_path = Path(video_dir) / video_path
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    return cap, width, height, fps, frame_count


def setup_pose_detector(config):
    """
    Initialize the pose detection model based on configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Pose detection model
    """
    pose_model_name = config.get('pose', {}).get('pose_model', 'body_with_feet').lower()
    mode = config.get('pose', {}).get('mode', 'balanced')
    device = config.get('pose', {}).get('device', 'auto')
    backend = config.get('pose', {}).get('backend', 'auto')
    det_frequency = config.get('pose', {}).get('det_frequency', 4)
    
    if POSE_ENGINE == 'rtmlib':
        # Configure pose detector based on model type
        if pose_model_name in ('halpe_26', 'body_with_feet'):
            model_class = BodyWithFeet
            logging.info("Using body with feet pose model")
        elif pose_model_name in ('coco_17', 'body'):
            model_class = Body
            logging.info("Using standard body pose model")
        elif pose_model_name in ('coco_133', 'whole_body', 'whole_body_wrist'):
            model_class = Wholebody
            logging.info("Using whole body pose model")
        else:
            logging.warning(f"Unknown model: {pose_model_name}, defaulting to body with feet")
            model_class = BodyWithFeet
        
        # Set up tracker with appropriate parameters
        detector = PoseTracker(
            model_class,
            det_frequency=det_frequency,
            mode=mode,
            device=device,
            backend=backend,
            tracking=True
        )
        
    elif POSE_ENGINE == 'mediapipe':
        mp_pose = mp.solutions.pose
        detector = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2 if mode == 'performance' else 1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    return detector


def setup_output_files(config, video_path, result_dir):
    """
    Prepare output directories and files
    
    Args:
        config: Configuration dictionary
        video_path: Path to input video
        result_dir: Directory to save results
        
    Returns:
        Dictionary of output paths
    """
    # Create timestamp for unique identification
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create base name for output files
    if video_path == "webcam":
        base_name = f"webcam_assessment_{timestamp}"
    else:
        video_name = Path(video_path).stem
        base_name = f"{video_name}_assessment_{timestamp}"
    
    # Create output directory
    output_dir = result_dir / base_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create paths for various outputs
    outputs = {
        'dir': output_dir,
        'video': output_dir / f"{base_name}.mp4",
        'images': output_dir / "frames",
        'data': output_dir / f"{base_name}_data.csv",
        'report': output_dir / f"{base_name}_report.pdf",
        'angles': output_dir / f"{base_name}_angles.csv",
        'metrics': output_dir / f"{base_name}_metrics.json"
    }
    
    # Create images directory if needed
    if config.get('process', {}).get('save_img', False):
        outputs['images'].mkdir(parents=True, exist_ok=True)
    
    return outputs


def process_frame(frame, detector, assessment_config):
    """
    Process a single frame to detect pose and calculate angles
    
    Args:
        frame: Input video frame
        detector: Pose detection model
        assessment_config: Configuration for the specific assessment
        
    Returns:
        Keypoints, joint angles, and visualization frame
    """
    # Detect pose
    if POSE_ENGINE == 'rtmlib':
        keypoints, scores = detector(frame)
    elif POSE_ENGINE == 'mediapipe':
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(frame_rgb)
        
        if results.pose_landmarks:
            # Convert MediaPipe landmarks to keypoints format
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape
            keypoints = np.array([[[lm.x * w, lm.y * h] for lm in landmarks]])
            scores = np.array([[lm.visibility for lm in landmarks]])
        else:
            # No pose detected
            keypoints = np.array([])
            scores = np.array([])
    
    # Filter unreliable keypoints
    filtered_keypoints = filter_keypoints(
        keypoints, 
        scores,
        threshold=assessment_config.get('keypoint_threshold', 0.3)
    )
    
    # Calculate joint angles for the assessment
    joint_angles = compute_joint_angles(
        filtered_keypoints,
        assessment_config.get('target_joints', [])
    )
    
    # Create visualization
    viz_frame = create_visualization(
        frame.copy(),
        filtered_keypoints,
        joint_angles,
        assessment_config
    )
    
    return filtered_keypoints, joint_angles, viz_frame


def analyze_movement(config, video_path, result_dir):
    """
    Main function to analyze movement from video input
    
    Args:
        config: Configuration dictionary
        video_path: Path to video file or "webcam"
        result_dir: Directory to save results
    
    Returns:
        Dictionary containing assessment results
    """
    # Get assessment-specific configuration
    assessment_type = config.get('project', {}).get('assessment', 'shoulder_flexion')
    assessment_config = get_assessment_config(assessment_type, config)
    
    logging.info(f"Starting {assessment_type} assessment")
    
    # Setup video source
    cap, width, height, fps, frame_count = setup_video_source(config, video_path)
    
    # Setup pose detector
    detector = setup_pose_detector(config)
    
    # Setup output files
    outputs = setup_output_files(config, video_path, result_dir)
    
    # Time range filtering
    time_range = config.get('project', {}).get('time_range', [])
    if time_range:
        start_frame = int(time_range[0] * fps)
        end_frame = int(time_range[1] * fps)
    else:
        start_frame = 0
        end_frame = frame_count
    
    # Setup video writer if needed
    video_writer = None
    if config.get('process', {}).get('save_vid', False):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(outputs['video']), 
            fourcc, 
            fps, 
            (width, height)
        )
    
    # Data collection
    all_keypoints = []
    all_angles = []
    frame_count = 0
    
    # Process video frames
    logging.info("Processing video frames...")
    frame_iterator = range(start_frame, end_frame)
    
    # Use tqdm for progress bar if not webcam
    if video_path != "webcam":
        frame_iterator = tqdm(frame_iterator)
    
    # Skip to start frame
    for _ in range(start_frame):
        cap.read()
    
    while cap.isOpened() and frame_count < end_frame:
        success, frame = cap.read()
        
        if not success:
            break
        
        # Process current frame
        keypoints, angles, viz_frame = process_frame(frame, detector, assessment_config)
        
        # Add ROM feedback
        viz_frame = draw_assessment_ui(
            viz_frame, 
            angles, 
            assessment_config,
            frame_count
        )
        
        # Show real-time results if configured
        if config.get('process', {}).get('show_realtime_results', True):
            cv2.imshow('ROM Assessment', viz_frame)
            
            # Exit on 'q' or ESC
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 27 is ESC key
                break
        
        # Save video frame if configured
        if video_writer:
            video_writer.write(viz_frame)
        
        # Save individual frame if configured
        if config.get('process', {}).get('save_img', False):
            cv2.imwrite(
                str(outputs['images'] / f"frame_{frame_count:04d}.jpg"),
                viz_frame
            )
        
        # Store data
        all_keypoints.append(keypoints)
        all_angles.append(angles)
        
        frame_count += 1
        
        # Update progress
        if video_path != "webcam":
            if frame_count % 10 == 0:
                frame_iterator.set_description(f"Processed {frame_count} frames")
    
    # Release resources
    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    
    # Post-processing
    logging.info("Post-processing data...")
    
    # Convert to DataFrame
    angles_df = pd.DataFrame(all_angles)
    
    # Interpolate missing values
    if config.get('post-processing', {}).get('interpolate', True):
        gap_size = config.get('post-processing', {}).get('interp_gap_smaller_than', 10)
        fill_method = config.get('post-processing', {}).get('fill_large_gaps_with', 'last_value')
        
        angles_df = interpolate_missing_points(angles_df, gap_size, fill_method)
    
    # Apply filtering if configured
    if config.get('post-processing', {}).get('filter', True):
        filter_type = config.get('post-processing', {}).get('filter_type', 'butterworth')
        
        if filter_type == 'butterworth':
            order = config.get('post-processing', {}).get('butterworth', {}).get('order', 4)
            cutoff = config.get('post-processing', {}).get('butterworth', {}).get('cut_off_frequency', 6)
            angles_df = filter_signal(angles_df, 'butterworth', fps, order, cutoff)
        
        elif filter_type == 'gaussian':
            sigma = config.get('post-processing', {}).get('gaussian', {}).get('sigma_kernel', 1)
            angles_df = filter_signal(angles_df, 'gaussian', sigma=sigma)
        
        elif filter_type == 'median':
            kernel = config.get('post-processing', {}).get('median', {}).get('kernel_size', 3)
            angles_df = filter_signal(angles_df, 'median', kernel_size=kernel)
    
    # Calculate ROM metrics
    metrics = calculate_rom_metrics(angles_df, assessment_config)
    
    # Save angle data
    if config.get('process', {}).get('save_angles', True):
        angles_df.to_csv(outputs['angles'], index=False)
        logging.info(f"Angle data saved to {outputs['angles']}")
    
    # Generate assessment report
    if config.get('reporting', {}).get('generate_report', True):
        report_path = generate_assessment_report(
            outputs['report'],
            assessment_type,
            metrics,
            angles_df,
            assessment_config
        )
        logging.info(f"Assessment report saved to {report_path}")
    
    # Return results
    results = {
        'assessment_type': assessment_type,
        'metrics': metrics,
        'outputs': outputs
    }
    
    # Log key results
    logging.info(f"Assessment completed: {assessment_type}")
    for joint, values in metrics.items():
        logging.info(f"{joint}: Range {values['min']}° to {values['max']}° (ROM: {values['range']}°)")
    
    return results