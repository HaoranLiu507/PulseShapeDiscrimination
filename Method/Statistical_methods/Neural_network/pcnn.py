"""
Pulse-Coupled Neural Network (PCNN) implementation for pulse shape discrimination.

This implementation uses a PCNN model to process raw pulse signals and generate
ignition maps for discrimination.

Reference:
- Liu, Hao-Ran, Yu-Xin Cheng, Zhuo Zuo, Tian-Tian Sun, and Kai-Min Wang.
  "Discrimination of neutrons and gamma rays in plastic scintillator based on pulse-coupled neural network."
  Nuclear Science and Techniques 32, no. 8 (2021): 82.
"""
import numpy as np
from scipy.signal import convolve2d
from typing import Union


def pcnn(pulse_signal_vector: np.ndarray) -> np.ndarray:
    """
    Process a pulse signal using the Pulse-Coupled Neural Network model.
    
    Implements a PCNN with optimized parameters for pulse shape discrimination between
    two particle classes, including feedback, linking, and dynamic threshold mechanisms.
    The model generates an ignition map that captures temporal characteristics of the input pulse.
    
    Args:
        pulse_signal_vector: 1D numpy array representing a single pulse signal
        
    Returns:
        1D numpy array containing the accumulated ignition map
    """
    # Ensure the pulse_signal_vector is 2D with shape (rows, cols)
    # If the input is a 1D array, treat it as a single-row signal.
    if pulse_signal_vector.ndim == 1:
        pulse_signal_vector = pulse_signal_vector.reshape(1, -1)

    rows, cols = pulse_signal_vector.shape

    # Set PCNN parameters
    l = 0.1091
    r = 0.1409
    kernel = np.array([[l, r, l],
                       [r, 0, r],
                       [l, r, l]])
    al = 0.356
    vl = 0.0005
    ve = 15.5
    ae = 0.081
    af = 0.325
    vf = 0.0005
    beta = 0.67
    iterations = 180

    # Initialize variables as 2D arrays of shape (rows, cols)
    time_pulse_sequence = np.zeros((rows, cols))
    feedback_input = np.zeros((rows, cols))
    link_input = np.zeros((rows, cols))
    dynamic_threshold = np.zeros((rows, cols))
    ignition_map = np.zeros((rows, cols))

    # Iterate to update the network state and accumulate the ignition map
    for _ in range(iterations):
        conv_tp = convolve2d(time_pulse_sequence, kernel, mode='same', boundary='fill', fillvalue=0)
        feedback_input = np.exp(-af) * feedback_input + vf * conv_tp + pulse_signal_vector
        link_input = np.exp(-al) * link_input + vl * conv_tp

        total_input = feedback_input * (1 + beta * link_input)

        # Generate binary time pulse sequence
        pulse_generator = 1.0 / (1.0 + np.exp(dynamic_threshold - total_input))
        time_pulse_sequence = (pulse_generator > 0.5).astype(float)

        dynamic_threshold = np.exp(-ae) * dynamic_threshold + ve * time_pulse_sequence

        ignition_map += time_pulse_sequence

    # Return a flattened ignition map (1D array) for further processing
    return ignition_map.flatten()


def get_psd_factor(
    pulse_signal: np.ndarray,
    roi_end: int = 123
) -> np.ndarray:
    """
    Perform pulse shape discrimination using the Pulse-Coupled Neural Network.
    
    For each pulse signal, generates an ignition map using PCNN and computes
    the discrimination factor as the sum of map values within a region of
    interest (ROI) starting from the pulse maximum. This approach effectively
    captures the temporal differences between class 1 and class 2 pulses.
    
    Args:
        pulse_signal: 2D numpy array where each row represents a pulse signal
        roi_end: Number of points after the maximum position to include in the ROI (default: 123)
        
    Returns:
        1D numpy array of discrimination factors for each pulse signal
    """
    num_signals = pulse_signal.shape[0]
    discrimination_factors = np.full(num_signals, np.nan)

    # Process each pulse signal
    for i in range(num_signals):
        # Get the pulse signal vector (1D array)
        pulse_vector = pulse_signal[i, :]

        # Compute the ignition map using PCNN
        ignition_map = pcnn(pulse_vector)

        # Find the index of the maximum value in the original pulse signal
        max_position = np.argmax(pulse_vector)

        # Ensure the ROI does not exceed the signal length
        end_index = min(max_position + roi_end, ignition_map.size)

        # Compute the discrimination factor as the sum over the ROI
        discrimination_factors[i] = np.sum(ignition_map[max_position:end_index])

    return discrimination_factors

