"""
Filters for radiation pulse signal denoising.

Reference:
- Liu, Hao-Ran, Ming-Zhe Liu, Yu-Long Xiao, Peng Li, Zhuo Zuo, and Yi-Han Zhan. 
"Discrimination of neutron and gamma ray using the ladder gradient method and 
analysis of filter adaptability." Nuclear Science and Techniques 33, no. 12 (2022): 159.
"""
import numpy as np
from scipy import signal
from scipy.ndimage import convolve1d, minimum_filter, maximum_filter
import pywt

def butterworth_filter(signals, order, cutoff_freq, fs, btype='low'):
    """Apply a Butterworth IIR filter (low-pass by default)."""
    nyq = 0.5 * fs
    cutoff = cutoff_freq / nyq
    b, a = signal.butter(order, cutoff, btype=btype)
    filtered_signals = signal.filtfilt(b, a, signals, axis=1)
    return filtered_signals

def chebyshev_filter(signals, order, ripple, cutoff_freq, fs, btype='low'):
    """Apply a Chebyshev Type I IIR filter with specified passband ripple."""
    nyq = 0.5 * fs
    cutoff = cutoff_freq / nyq
    b, a = signal.cheby1(order, ripple, cutoff, btype=btype)
    filtered_signals = signal.filtfilt(b, a, signals, axis=1)
    return filtered_signals

def elliptic_filter(signals, order, rp, rs, cutoff_freq, fs, btype='low'):
    """Apply an Elliptic IIR filter with ripple in passband and stopband."""
    nyq = 0.5 * fs
    cutoff = cutoff_freq / nyq
    b, a = signal.ellip(order, rp, rs, cutoff, btype=btype)
    filtered_signals = signal.filtfilt(b, a, signals, axis=1)
    return filtered_signals

def fourier_filter(signals, cutoff_freq, fs):
    """Apply a Fourier-based low-pass filter by zeroing high-frequency components."""
    n = signals.shape[1]
    fft = np.fft.fft(signals, axis=1)
    freqs = np.fft.fftfreq(n, d=1/fs)
    mask = np.abs(freqs) <= cutoff_freq
    fft[:, ~mask] = 0
    filtered_signals = np.fft.ifft(fft, axis=1).real
    return filtered_signals

def least_mean_square_filter(signals, mu, order):
    """Apply an LMS adaptive filter to enhance predictable signal components."""
    filtered_signals = np.zeros_like(signals)
    for i in range(signals.shape[0]):
        filtered_signals[i] = lms_filter(signals[i], mu, order)
    return filtered_signals

def lms_filter(signal, mu, order):
    """Helper function: Apply LMS adaptive filter to a single signal."""
    n = len(signal)
    w = np.zeros(order)
    filtered = np.zeros(n)
    for i in range(order, n):
        x = signal[i-order:i][::-1]
        y = np.dot(w, x)
        e = signal[i] - y
        w = w + 2 * mu * e * x
        filtered[i] = y
    return filtered

def median_filter(signals, kernel_size):
    """Apply a median filter to remove impulsive noise."""
    filtered_signals = signal.medfilt(signals, kernel_size=(1, kernel_size))
    return filtered_signals

def morphological_filter(signals, size):
    """Apply morphological opening and closing to remove noise spikes."""
    erosion = minimum_filter(signals, size=(1, size))
    opening = maximum_filter(erosion, size=(1, size))
    dilation = maximum_filter(signals, size=(1, size))
    closing = minimum_filter(dilation, size=(1, size))
    filtered_signals = (opening + closing) / 2
    return filtered_signals

def moving_average_filter(signals, window_size):
    """Smooth signals using a moving average with a rectangular window."""
    window = np.ones(window_size) / window_size
    filtered_signals = convolve1d(signals, window, axis=1, mode='nearest')
    return filtered_signals

def wavelet_filter(signals, wavelet='db4', level=4, threshold=0.1):
    """Denoise signals using wavelet decomposition and soft thresholding."""
    filtered_signals = np.zeros_like(signals)
    for i in range(signals.shape[0]):
        sig = signals[i]
        coeffs = pywt.wavedec(sig, wavelet, level=level)
        coeffs_thresholded = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
        filtered_sig = pywt.waverec(coeffs_thresholded, wavelet)
        if len(filtered_sig) > len(sig):
            filtered_sig = filtered_sig[:len(sig)]
        elif len(filtered_sig) < len(sig):
            filtered_sig = np.pad(filtered_sig, (0, len(sig) - len(filtered_sig)), 'constant')
        filtered_signals[i] = filtered_sig
    return filtered_signals

def wiener_filter(signals, mysize=None, noise=None):
    """Apply a Wiener filter for optimal denoising given noise characteristics."""
    filtered_signals = np.zeros_like(signals)
    for i in range(signals.shape[0]):
        filtered_signals[i] = signal.wiener(signals[i], mysize=mysize, noise=noise)
    return filtered_signals

def windowed_sinc_filter(signals, cutoff_freq, fs, numtaps, window='hamming'):
    """Apply a windowed-sinc FIR filter (low-pass)."""
    nyq = 0.5 * fs
    cutoff = cutoff_freq / nyq
    h = signal.firwin(numtaps, cutoff, window=window)
    filtered_signals = signal.filtfilt(h, 1, signals, axis=1)
    return filtered_signals