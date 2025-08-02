function [Pulse_shape_discrimination_factor] = PGA(Pulse_signal, t)

%--------------------------------------------------------------------------
% Pulse Gradient Analysis (PGA) method for pulse shape discrimination.
%
% Discriminates particle types by analyzing the decay rate of pulse signals
% immediately after their peak. Computes the gradient between the peak and
% a fixed time delay to capture differences in decay characteristics.
%
% Input:
%   Pulse_signal: Matrix of pulse signals to analyze [n_signals × signal_length]
%   t: Time delay after peak for gradient calculation (default: 20 samples)
%
% Output:
%   Pulse_shape_discrimination_factor: Signal gradients after peak
%                                    [n_signals × 1]
%
% Algorithm:
%   1. Find peak amplitude and position for each signal
%   2. Sample amplitude at fixed delay t after peak
%   3. Calculate gradient: g = (A(t_peak + t) - A_peak) / t
%   where A is amplitude, t_peak is peak position
%
% Note:
%   - Gradient calculation requires t samples after peak
%   - Shorter t focuses on initial decay
%   - Longer t captures more of decay curve
%   - Error if peak + t exceeds signal length
%
% Reference:
% D'Mellow, Bob, et al. "Digital discrimination of neutrons and γ-rays in
% liquid scintillators using pulse gradient analysis." Nuclear Instruments
% and Methods in Physics Research Section A: Accelerators, Spectrometers,
% Detectors and Associated Equipment 578.1 (2007): 191-197.
%--------------------------------------------------------------------------

% Set default time delay
if nargin < 2
    t = 20;  % Samples after peak
end

% Initialize output array
Num_signal = size(Pulse_signal, 1);
Pulse_shape_discrimination_factor = zeros(Num_signal, 1);

% Process signals in parallel
parfor i = 1:Num_signal
    % Find peak characteristics
    [Max_value, Max_index] = max(Pulse_signal(i, :));  % [amplitude, position]
    
    % Calculate delayed sample position
    Second_sample_index = Max_index + t;
    
    % Validate delayed position
    if Second_sample_index > size(Pulse_signal, 2)
        error('Delayed sample at peak+%d exceeds signal length for signal %d.', t, i);
    end
    
    % Get delayed amplitude
    Second_sample_value = Pulse_signal(i, Second_sample_index);
    
    % Compute decay gradient
    Pulse_shape_discrimination_factor(i) = (Second_sample_value - Max_value) / t;
end

end
