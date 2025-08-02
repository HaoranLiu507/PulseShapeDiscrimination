function [Pulse_shape_discrimination_factor] = FS(Pulse_signal_original, ROI_range, fs)

%--------------------------------------------------------------------------
% Performs pulse shape discrimination using Fractal Spectrum (FS) analysis.
%
% Uses Fourier transform to compute the fractal dimension of pulse signals
% in the frequency domain for discrimination.
%
% Inputs:
%   ROI_range: Region of interest [start, end] where neutron and gamma 
%             signal decay rates differ most
%   fs: Sampling frequency in Hz
%
% Reference:
% Liu, Ming-Zhe, et al. "Toward a fractal spectrum approach for neutron 
% and gamma pulse shape discrimination." Chinese Physics C 40.6 (2016): 
% 066201.
%--------------------------------------------------------------------------
    
% Set default parameters
if nargin < 2
    ROI_range = [60, 130];  % Default ROI range
end
if nargin < 3
    fs = 2;  % Default sampling rate (Hz)
end

% Constants for fractal dimension calculation
a = 2; 
b = 5; 

% Initialize output array
Num_signal = size(Pulse_signal_original, 1);
Pulse_shape_discrimination_factor = zeros(Num_signal, 1);

% Process signals in parallel
parfor i = 1:Num_signal
    Pulse_signal = Pulse_signal_original(i, :);
    
    % Extract ROI window after signal peak
    [~, Max_position] = max(Pulse_signal);
    Pulse_signal_roi = Pulse_signal(Max_position + ROI_range(1): Max_position + ROI_range(2));

    % Get ROI length for spectrum analysis
    Region_interested_length = length(Pulse_signal_roi);
    
    % Calculate power spectrum using periodogram
    Window = boxcar(Region_interested_length);
    [Power_spectrum_estimation, Frequency] = periodogram(Pulse_signal_roi, Window, length(Pulse_signal_roi), fs);
    
    % Convert to log scale for fractal analysis
    Power_spectrum_estimation_log = log10(Power_spectrum_estimation);
    Frequency_log = log10(Frequency);
    
    % Linear regression to find fractal dimension
    Coefficient_of_the_regression_fitting = regress(Frequency_log(2:end), Power_spectrum_estimation_log(2:end));
    
    % Calculate discrimination factor:
    % Factor = b/a - regression_coefficient
    Pulse_shape_discrimination_factor(i) = b/a - Coefficient_of_the_regression_fitting;
end

end
