#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Signal Filtering Module

This module provides functions for filtering angle and position data
using various filtering methods like Butterworth, Gaussian, and median filters.
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import logging


def filter_signal(data, filter_type, *args, **kwargs):
    """
    Apply specified filter to angle or position data
    
    Args:
        data: DataFrame or array of values to filter
        filter_type: Type of filter to apply ('butterworth', 'gaussian', 'median')
        *args, **kwargs: Additional parameters for the specific filter
    
    Returns:
        Filtered data in the same format as input
    """
    # Handle DataFrame input
    if isinstance(data, pd.DataFrame):
        filtered_data = data.copy()
        
        for col in filtered_data.columns:
            # Skip columns with all NaN values
            if filtered_data[col].isna().all():
                continue
            
            # Apply filter to column
            if filter_type == 'butterworth':
                if len(args) >= 3:
                    fps, order, cutoff = args[0], args[1], args[2]
                else:
                    fps = kwargs.get('fps', 30)
                    order = kwargs.get('order', 4)
                    cutoff = kwargs.get('cutoff', 6)
                
                filtered_data[col] = butterworth_filter(filtered_data[col], fps, order, cutoff)
            
            elif filter_type == 'gaussian':
                sigma = kwargs.get('sigma', 1)
                filtered_data[col] = gaussian_filter(filtered_data[col], sigma)
            
            elif filter_type == 'median':
                kernel_size = kwargs.get('kernel_size', 3)
                filtered_data[col] = median_filter(filtered_data[col], kernel_size)
            
            else:
                logging.warning(f"Unknown filter type: {filter_type}. No filtering applied.")
        
        return filtered_data
    
    # Handle array input
    else:
        if filter_type == 'butterworth':
            if len(args) >= 3:
                fps, order, cutoff = args[0], args[1], args[2]
            else:
                fps = kwargs.get('fps', 30)
                order = kwargs.get('order', 4)
                cutoff = kwargs.get('cutoff', 6)
            
            return butterworth_filter(data, fps, order, cutoff)
        
        elif filter_type == 'gaussian':
            sigma = kwargs.get('sigma', 1)
            return gaussian_filter(data, sigma)
        
        elif filter_type == 'median':
            kernel_size = kwargs.get('kernel_size', 3)
            return median_filter(data, kernel_size)
        
        else:
            logging.warning(f"Unknown filter type: {filter_type}. No filtering applied.")
            return data


def butterworth_filter(data, fps, order=4, cutoff=6):
    """
    Apply Butterworth low-pass filter to data
    
    Args:
        data: Array or Series of values to filter
        fps: Sampling frequency in Hz
        order: Filter order
        cutoff: Cutoff frequency in Hz
    
    Returns:
        Filtered data
    """
    # Convert pandas Series to numpy array
    if isinstance(data, pd.Series):
        is_series = True
        index = data.index
        data_array = data.values
    else:
        is_series = False
        data_array = data
    
    # Handle NaN values by creating a mask and interpolating
    mask = ~np.isnan(data_array)
    
    # Not enough valid data points
    if np.sum(mask) < order + 1:
        if is_series:
            return data
        else:
            return data_array
    
    # Create temporary array for filtering
    temp_array = np.copy(data_array)
    
    # Interpolate NaN values for filtering
    valid_indices = np.where(mask)[0]
    
    if len(valid_indices) > 0:
        # Get valid values
        valid_values = data_array[mask]
        
        # Linear interpolation for missing values
        for i in range(len(data_array)):
            if not mask[i]:
                # Find closest valid indices before and after
                before = valid_indices[valid_indices < i]
                after = valid_indices[valid_indices > i]
                
                if len(before) > 0 and len(after) > 0:
                    # Interpolate between closest valid points
                    before_idx = before[-1]
                    after_idx = after[0]
                    
                    before_val = data_array[before_idx]
                    after_val = data_array[after_idx]
                    
                    # Linear interpolation
                    weight = (i - before_idx) / (after_idx - before_idx)
                    temp_array[i] = before_val * (1 - weight) + after_val * weight
                elif len(before) > 0:
                    # Only valid points before
                    temp_array[i] = data_array[before[-1]]
                elif len(after) > 0:
                    # Only valid points after
                    temp_array[i] = data_array[after[0]]
    
    # Design Butterworth low-pass filter
    nyquist = 0.5 * fps
    normal_cutoff = cutoff / nyquist
    
    # Ensure cutoff is valid
    if normal_cutoff >= 1.0:
        logging.warning(f"Cutoff frequency {cutoff} Hz is too high for sampling rate {fps} Hz. Using maximum valid cutoff.")
        normal_cutoff = 0.99
    
    # Create filter
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    
    # Apply filter
    filtered_array = signal.filtfilt(b, a, temp_array)
    
    # Restore NaN values in original positions
    filtered_array[~mask] = np.nan
    
    # Convert back to series if input was a series
    if is_series:
        return pd.Series(filtered_array, index=index)
    else:
        return filtered_array


def gaussian_filter(data, sigma=1):
    """
    Apply Gaussian filter to data
    
    Args:
        data: Array or Series of values to filter
        sigma: Standard deviation of Gaussian kernel
    
    Returns:
        Filtered data
    """
    # Convert pandas Series to numpy array
    if isinstance(data, pd.Series):
        is_series = True
        index = data.index
        data_array = data.values
    else:
        is_series = False
        data_array = data
    
    # Handle NaN values
    mask = ~np.isnan(data_array)
    
    # Not enough valid data points
    if np.sum(mask) < 3:
        if is_series:
            return data
        else:
            return data_array
    
    # Create temporary array for filtering
    temp_array = np.copy(data_array)
    
    # Interpolate NaN values for filtering
    valid_indices = np.where(mask)[0]
    
    if len(valid_indices) > 0:
        # Get valid values
        valid_values = data_array[mask]
        
        # Linear interpolation for missing values
        for i in range(len(data_array)):
            if not mask[i]:
                # Find closest valid indices before and after
                before = valid_indices[valid_indices < i]
                after = valid_indices[valid_indices > i]
                
                if len(before) > 0 and len(after) > 0:
                    # Interpolate between closest valid points
                    before_idx = before[-1]
                    after_idx = after[0]
                    
                    before_val = data_array[before_idx]
                    after_val = data_array[after_idx]
                    
                    # Linear interpolation
                    weight = (i - before_idx) / (after_idx - before_idx)
                    temp_array[i] = before_val * (1 - weight) + after_val * weight
                elif len(before) > 0:
                    # Only valid points before
                    temp_array[i] = data_array[before[-1]]
                elif len(after) > 0:
                    # Only valid points after
                    temp_array[i] = data_array[after[0]]
    
    # Apply Gaussian filter
    filtered_array = gaussian_filter1d(temp_array, sigma)
    
    # Restore NaN values in original positions
    filtered_array[~mask] = np.nan
    
    # Convert back to series if input was a series
    if is_series:
        return pd.Series(filtered_array, index=index)
    else:
        return filtered_array


def median_filter(data, kernel_size=3):
    """
    Apply median filter to data
    
    Args:
        data: Array or Series of values to filter
        kernel_size: Size of the median filter kernel
    
    Returns:
        Filtered data
    """
    # Convert pandas Series to numpy array
    if isinstance(data, pd.Series):
        is_series = True
        index = data.index
        data_array = data.values
    else:
        is_series = False
        data_array = data
    
    # Handle NaN values
    mask = ~np.isnan(data_array)
    
    # Not enough valid data points
    if np.sum(mask) < kernel_size:
        if is_series:
            return data
        else:
            return data_array
    
    # Create temporary array for filtering
    temp_array = np.copy(data_array)
    
    # Interpolate NaN values for filtering
    valid_indices = np.where(mask)[0]
    
    if len(valid_indices) > 0:
        # Get valid values
        valid_values = data_array[mask]
        
        # Linear interpolation for missing values
        for i in range(len(data_array)):
            if not mask[i]:
                # Find closest valid indices before and after
                before = valid_indices[valid_indices < i]
                after = valid_indices[valid_indices > i]
                
                if len(before) > 0 and len(after) > 0:
                    # Interpolate between closest valid points
                    before_idx = before[-1]
                    after_idx = after[0]
                    
                    before_val = data_array[before_idx]
                    after_val = data_array[after_idx]
                    
                    # Linear interpolation
                    weight = (i - before_idx) / (after_idx - before_idx)
                    temp_array[i] = before_val * (1 - weight) + after_val * weight
                elif len(before) > 0:
                    # Only valid points before
                    temp_array[i] = data_array[before[-1]]
                elif len(after) > 0:
                    # Only valid points after
                    temp_array[i] = data_array[after[0]]
    
    # Apply median filter
    filtered_array = signal.medfilt(temp_array, kernel_size)
    
    # Restore NaN values in original positions
    filtered_array[~mask] = np.nan
    
    # Convert back to series if input was a series
    if is_series:
        return pd.Series(filtered_array, index=index)
    else:
        return filtered_array


def detect_outliers(data, window_size=5, threshold=2.0):
    """
    Detect outliers in angle or position data using rolling statistics
    
    Args:
        data: Array or Series of values
        window_size: Size of rolling window
        threshold: Z-score threshold for outlier detection
    
    Returns:
        Boolean mask with True for outliers
    """
    # Convert to pandas Series for rolling calculations
    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    
    # Calculate rolling mean and standard deviation
    rolling_mean = data.rolling(window=window_size, center=True).mean()
    rolling_std = data.rolling(window=window_size, center=True).std()
    
    # Calculate z-scores
    z_scores = (data - rolling_mean) / rolling_std
    
    # Handle NaN values in z-scores (e.g., at boundaries)
    z_scores = z_scores.fillna(0)
    
    # Identify outliers
    outliers = np.abs(z_scores) > threshold
    
    return outliers


def remove_outliers(data, outlier_mask, method='interpolate'):
    """
    Remove outliers from data
    
    Args:
        data: Array or Series of values
        outlier_mask: Boolean mask with True for outliers
        method: Method for handling outliers ('interpolate', 'nan', 'mean')
    
    Returns:
        Data with outliers removed or replaced
    """
    # Convert to pandas Series
    if not isinstance(data, pd.Series):
        is_series = False
        data = pd.Series(data)
    else:
        is_series = True
    
    # Create a copy to avoid modifying the original
    result = data.copy()
    
    if method == 'interpolate':
        # Set outliers to NaN, then interpolate
        result[outlier_mask] = np.nan
        result = result.interpolate(method='linear')
        
        # Handle boundary cases
        result = result.fillna(method='ffill').fillna(method='bfill')
    
    elif method == 'nan':
        # Simply set outliers to NaN
        result[outlier_mask] = np.nan
    
    elif method == 'mean':
        # Replace outliers with window mean
        window_size = 5  # Default window size
        for i in np.where(outlier_mask)[0]:
            # Define window boundaries
            start = max(0, i - window_size // 2)
            end = min(len(data), i + window_size // 2 + 1)
            
            # Get non-outlier values in window
            window_values = data.iloc[start:end][~outlier_mask.iloc[start:end]]
            
            if len(window_values) > 0:
                result.iloc[i] = window_values.mean()
            else:
                # No non-outlier values in window, set to NaN
                result.iloc[i] = np.nan
    
    else:
        logging.warning(f"Unknown method: {method}. Outliers not removed.")
    
    # Convert back to numpy array if input was array
    if not is_series:
        result = result.values
    
    return result