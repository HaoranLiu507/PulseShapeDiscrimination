"""
Principal Component Analysis (PCA) method for pulse shape discrimination.

The method computes discrimination factors by projecting pulse signals onto 
the first principal component derived from the covariance matrix of a training 
subset of pulses.

Reference:
- Alharbi, T.
  "Principal component analysis for pulse-shape discrimination of scintillation radiation detectors."
  Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers,
  Detectors and Associated Equipment 806 (2016): 240-243.
"""

import numpy as np

def get_psd_factor(
    pulse_signal: np.ndarray,
    num_rows: int = 1000,
    signal_cut: int = 79
) -> np.ndarray:
    """
    Calculate PSD factors using Principal Component Analysis.
    
    Args:
        pulse_signal: Input signals array of shape (N, L) where N is the number
                     of pulses and L is the length of each pulse
        num_rows: Number of pulses to use for computing the covariance matrix
                 and principal components (default: 3000)
        signal_cut: Starting index for the region of interest in each pulse,
                   using 1-based indexing (default: 79)
    
    Returns:
        numpy.ndarray: Array of PSD factors for each input pulse, computed as
                      the absolute projection onto the first principal component
    
    Raises:
        ValueError: If num_rows > N, signal_cut > L, or if input contains NaN
    
    Note:
        The method uses only the tail portion of each pulse (from signal_cut
        onwards) to focus on the region with maximum discrimination power.
    """
    num_signals, num_columns = pulse_signal.shape
    if num_rows > num_signals:
        raise ValueError("num_rows exceeds the number of available pulse signals.")
    if signal_cut > num_columns:
        raise ValueError("signal_cut exceeds the number of columns in pulse_signal.")

    # Adjust for Python's 0-indexing (MATLAB's signal_cut corresponds to the signal_cut-th column)
    cut_index = signal_cut - 1

    # Extract the region of interest (ROI) for PCA from the first num_rows signals
    X = pulse_signal[:num_rows, cut_index:]

    if np.isnan(X).any():
        raise ValueError("Input data contains NaN values. Please clean your data.")

    # Compute the covariance matrix with each column representing a variable
    covariance = np.cov(X, rowvar=False)

    # Compute eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance)

    # Find the eigenvector corresponding to the largest eigenvalue
    max_index = np.argmax(eigenvalues)
    max_eigenvector = eigenvectors[:, max_index]

    # Use all pulse signals (all rows) for projection onto the principal component
    Y = pulse_signal[:, cut_index:]

    # Project each pulse signal onto the principal component and take the absolute value
    # as the discrimination factor
    discrimination_factors = np.abs(Y.dot(max_eigenvector))

    return discrimination_factors
