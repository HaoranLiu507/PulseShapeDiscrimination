"""
Ladder Gradient (LG) method for pulse shape discrimination using a quasi-continuous
spiking cortical model (QC-SCM).

This implementation uses a QC-SCM to generate an ignition map, then computes
discrimination factors based on the slope between key points in the map.

Reference:
- Liu, Hao-Ran, Ming-Zhe Liu, Yu-Long Xiao, Peng Li, Zhuo Zuo, and Yi-Han Zhan.
  "Discrimination of neutron and gamma ray using the ladder gradient method and
  analysis of filter adaptability." Nuclear Science and Techniques 33, no. 12 (2022): 159.
"""
import numpy as np
from scipy.signal import convolve2d


def qc_scm(pulse_signal_vector: np.ndarray) -> np.ndarray:
    """
    Compute the Ignition Map using a quasi-continuous spiking cortical model (QC-SCM).
    
    Processes a single pulse signal to generate an ignition map using a QC-SCM with
    optimized parameters for pulse shape discrimination between two particle classes.
    The model uses a specific connection weight matrix and dynamic thresholding mechanism.
    
    Args:
        pulse_signal_vector: 1D numpy array representing a single pulse signal
        
    Returns:
        1D numpy array representing the computed ignition map
    """
    # Ensure the pulse signal is a row vector (2D array of shape (1, N))
    pulse_signal_vector = pulse_signal_vector.reshape(1, -1)
    rows, cols = pulse_signal_vector.shape

    # Set the connection weight matrix
    matrix = np.array([[0, 0, 0],
                       [0.44, 0, 0.44],
                       [0, 0, 0]])

    # Initialize sequences
    time_pulse_sequence = np.zeros((rows, cols))
    neural_potential = np.copy(time_pulse_sequence)
    ignition_map = np.copy(time_pulse_sequence)
    dynamic_threshold = time_pulse_sequence + 1.0

    # Model parameters
    f_val = 0.38
    g_val = 0.8
    h_val = 8.45
    time_step = 0.5
    total_time = 50

    # Adjust parameters by exponentiation (as in MATLAB: g = g^t, f = f^t)
    g_val = g_val ** time_step
    f_val = f_val ** time_step

    # Loop from time=1 to total_time (with step time_step)
    for t in np.arange(1, total_time + time_step, time_step):
        # Convolve the time pulse sequence with the weight matrix using 'same' mode
        conv_result = convolve2d(time_pulse_sequence, matrix, mode='same')
        neural_potential = f_val * neural_potential + pulse_signal_vector * conv_result + pulse_signal_vector
        dynamic_threshold = g_val * dynamic_threshold + h_val * time_pulse_sequence
        pulse_generator = 1.0 / (1.0 + np.exp(dynamic_threshold - neural_potential))

        # Update the time pulse sequence to be binary based on a threshold of 0.5
        time_pulse_sequence = (pulse_generator > 0.5).astype(float)

        # Accumulate the ignition map over iterations
        ignition_map = ignition_map + time_pulse_sequence

    # Return the ignition map as a 1D array (flattened)
    return ignition_map.flatten()


def get_psd_factor(
    pulse_signal: np.ndarray,
    m: int = 20
) -> np.ndarray:
    """
    Perform pulse shape discrimination using the Ladder Gradient method.
    
    For each pulse signal, computes an ignition map using QC-SCM and determines
    the discrimination factor as the slope between the maximum point and the m-th
    occurrence of the most frequent value in the map. This gradient-based approach
    provides effective separation between class 1 and class 2 pulses.
    
    Args:
        pulse_signal: 2D numpy array where each row represents a pulse signal
        m: The m-th occurrence after the maximum point to use in slope calculation (default: 20)
        
    Returns:
        1D numpy array of discrimination factors for each pulse signal
    """
    num_signals = pulse_signal.shape[0]
    discrimination_factors = np.full(num_signals, np.nan)

    for i in range(num_signals):
        signal_vector = pulse_signal[i, :]

        # Compute the ignition map using QC_SCM
        ignition_map = qc_scm(signal_vector)

        # Find the maximum position in the pulse signal
        max_position = np.argmax(signal_vector)

        # Compute the most frequent number in the ignition map (mode)
        ignition_map_int = ignition_map.astype(int)
        if ignition_map_int.size > 0:
            frequent_number = np.bincount(ignition_map_int).argmax()
        else:
            discrimination_factors[i] = np.nan
            continue

        # Define the region of interest (ROI) from the maximum position onward
        ROI = ignition_map[max_position:]

        # Find indices in ROI where the value equals the frequent number
        indices = np.flatnonzero(ROI == frequent_number)

        # If at least one occurrence exists, use the m-th if available, otherwise the last occurrence.
        if len(indices) > 0:
            index_to_use = indices[m - 1] if len(indices) >= m else indices[-1]
            second_point_position = index_to_use + max_position
            # Avoid division by zero
            if (max_position - second_point_position) != 0:
                discrimination_factors[i] = (ignition_map[max_position] - ignition_map[second_point_position]) / (
                    max_position - second_point_position)
            else:
                discrimination_factors[i] = np.nan
        else:
            discrimination_factors[i] = np.nan

    return discrimination_factors

