function [Pulse_shape_discrimination_factor] = WT2(Pulse_signal, Sigma, Wavelet_length, Dt)

%--------------------------------------------------------------------------
% Performs pulse shape discrimination using Marr Wavelet Transform (WT2).
% 
% Uses Mexican Hat (Marr) wavelet transform to analyze signal features.
% Discrimination factor is computed as ratio between original signal integral
% and positive parts of the wavelet-transformed signal.
%
% Inputs:
%   Sigma: Wavelet width parameter (default: sqrt(5))
%   Wavelet_length: Half-width of wavelet support (default: 100)
%   Dt: Sampling time interval (default: 1.0)
%
% Reference:
% Langeveld, Willem GJ, et al. "Pulse shape discrimination algorithms, 
% figures of merit, and gamma-rejection for liquid and solid scintillators."
% IEEE Transactions on Nuclear Science 64.7 (2017): 1801-1809.
%--------------------------------------------------------------------------
    
    % Set default parameters
    if nargin < 2
        Sigma = sqrt(5);        % Controls wavelet spread
    end
    if nargin < 3
        Wavelet_length = 100;   % Controls analysis window
    end
    if nargin < 4
        Dt = 1.0;              % Time step between samples
    end
    
    % Initialize arrays
    Num_signals = size(Pulse_signal, 1);
    Pulse_shape_discrimination_factor = zeros(Num_signals, 1);
    
    % Generate Marr (Mexican Hat) wavelet
    T_wavelet = linspace(-Wavelet_length, Wavelet_length, 2 * Wavelet_length + 1);
    Wavelet = marr_wavelet(T_wavelet, Sigma);
    
    % Process each signal
    for i = 1:Num_signals
        % Get current signal
        Signal = Pulse_signal(i, :);
        
        % Create time vector for integration
        Time_vector = (0:length(Signal)-1) * Dt;
        
        % Calculate original signal integral
        Integral_signal = trapz(Time_vector, Signal);
        
        % Apply wavelet transform via convolution
        Convolved_signal = conv(Signal, Wavelet, 'same');
        
        % Calculate integral of positive wavelet components
        Positive_convolved = max(0, Convolved_signal);
        Integral_wavelet = trapz(Time_vector, Positive_convolved);
        
        % Calculate discrimination factor:
        % 2 * Original_Integral / (Original_Integral + Wavelet_Integral)
        Pulse_shape_discrimination_factor(i) = 2 * Integral_signal / (Integral_signal + Integral_wavelet);
    end
end

function Wavelet = marr_wavelet(t, Sigma)
    %--------------------------------------------------------------------------
    % Generates Marr (Mexican Hat) wavelet.
    %
    % Formula: ψ(t) = (1 - t²/σ²) * exp(-t²/(2σ²))
    %
    % Inputs:
    %   t: Time points for wavelet evaluation
    %   Sigma: Width parameter controlling spread
    %--------------------------------------------------------------------------
    Wavelet = (1 - (t.^2) / (Sigma^2)) .* exp(-t.^2 / (2 * Sigma^2));
end

