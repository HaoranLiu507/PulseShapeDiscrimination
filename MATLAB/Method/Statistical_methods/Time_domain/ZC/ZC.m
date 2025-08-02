function [Pulse_shape_discrimination_factor] = ZC(Pulse_signal, T, constant)

%--------------------------------------------------------------------------
% Zero Crossing (ZC) method for pulse shape discrimination.
%
% Discriminates particle types by analyzing the timing characteristics of
% processed pulse signals. Applies a digital filter to transform the pulse
% shape and measures the time between pulse start and zero-crossing point.
%
% Input:
%   Pulse_signal: Matrix of pulse signals to analyze [n_signals × signal_length]
%   T: Sampling period (default: 1e-10 seconds)
%   constant: Filter time constant (default: 7e-9 seconds)
%
% Output:
%   Pulse_shape_discrimination_factor: Time to zero-crossing [n_signals × 1]
%                                    Units: sample periods
%
% Algorithm:
%   1. Apply digital filter with transfer function:
%      H(s) = s/(s + 1/τ)³
%      where τ is the filter time constant
%   2. Find maximum of filtered signal
%   3. Locate first zero-crossing after maximum
%   4. Compute time difference: zero-crossing - (0.1 × peak_position)
%
% Filter implementation:
%   - 3rd order recursive digital filter
%   - Alpha = exp(-T/constant): decay factor
%   - Includes amplitude and timing corrections
%
% Note:
%   - Returns NaN if no zero-crossing found
%   - Start point set at 10% of peak position
%   - Filter parameters affect discrimination performance
%
% Reference:
% Nakhostin, Mohammad. "A comparison of digital zero-crossing and
% charge-comparison methods for neutron/γ-ray discrimination with liquid
% scintillation detectors." Nuclear Instruments and Methods in Physics
% Research Section A: Accelerators, Spectrometers, Detectors and Associated
% Equipment 797 (2015): 77-82.
%--------------------------------------------------------------------------

% Set filter parameters
if nargin < 2
    T = 1e-10;        % Sampling period (seconds)
end

if nargin < 3
    constant = 7e-9;  % Filter time constant (seconds)
end

% Initialize output array
Num_signal = size(Pulse_signal, 1);
Pulse_shape_discrimination_factor = zeros(Num_signal, 1);

% Calculate filter coefficient
Alpha = exp(-T / constant);  % Decay factor for digital filter

% Process signals in parallel
parfor i = 1:Num_signal
    % Get current signal
    Pulse_signal_vector = Pulse_signal(i, :);
    Pulse_signal_vector_length = length(Pulse_signal_vector);
    
    % Prepare signals with padding for filter
    Pulse_signal_filtered_vector = [0, 0, 0, Pulse_signal_vector];  % Zero-padded input
    Processed_signal = zeros(1, Pulse_signal_vector_length + 3);    % Filter output
    Data_processed_signal = zeros(1, Pulse_signal_vector_length + 3);  % Working copy

    % Apply recursive digital filter
    for n = 4:Pulse_signal_vector_length + 3
        % Implementation of H(s) = s/(s + 1/τ)³
        Processed_signal(n) = 3 * Alpha * (Processed_signal(n-1)) - ...
            3 * (Alpha^2) * (Processed_signal(n-2)) + ...
            (Alpha^3) * (Processed_signal(n-3)) + ...
            T * Alpha * (1 - (1/(constant * T/2))) * (Pulse_signal_filtered_vector(n-1)) - ...
            T * (Alpha^2) * (1 + (1/(constant * T/2))) * (Pulse_signal_filtered_vector(n-2));
        
        Data_processed_signal(n) = Processed_signal(n);
    end

    % Find signal features
    [~, Maxposition] = max(Data_processed_signal);  % Peak position
    Stop_point = NaN;  % Initialize zero-crossing point

    % Locate zero-crossing after peak
    for j = Maxposition:Pulse_signal_vector_length + 3
        if Data_processed_signal(j) < 0
            Stop_point = j;  % First negative point
            break;
        end
    end

    % Calculate discrimination factor
    Start_point = round(Maxposition * 0.1);  % 10% of peak position
    
    if isnan(Stop_point)
        Pulse_shape_discrimination_factor(i) = NaN;  % No zero-crossing found
    else
        Pulse_shape_discrimination_factor(i) = Stop_point - Start_point;  % Time difference
    end
end

end
