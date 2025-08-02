function [Pulse_shape_discrimination_factor] = FEPS(Pulse_signal)

%--------------------------------------------------------------------------
% Falling-Edge Percentage Slope (FEPS) method for pulse shape discrimination.
%
% Discriminates particle types by analyzing the falling edge characteristics
% of the pulse signal. Computes the slope between two amplitude thresholds
% on the falling edge to capture decay time differences.
%
% Input:
%   Pulse_signal: Matrix of pulse signals to analyze [n_signals × signal_length]
%
% Output:
%   Pulse_shape_discrimination_factor: Slope between threshold points
%                                    [n_signals × 1]
%
% Threshold levels:
%   - Upper threshold: 60% of pulse maximum
%   - Lower threshold: 10% of pulse maximum
%
% Algorithm:
%   1. Find pulse peak
%   2. Locate upper (60%) and lower (10%) threshold crossings
%   3. Calculate slope between threshold points
%   4. Slope = (Lower_value - Upper_value) / (Lower_time - Upper_time)
%
% Reference:
% Zuo, Zhuo, et al. "Discrimination of neutrons and gamma-rays in plastic
% scintillator based on falling-edge percentage slope method." Nuclear
% Instruments and Methods in Physics Research Section A: Accelerators,
% Spectrometers, Detectors and Associated Equipment 1010 (2021): 165483.
%--------------------------------------------------------------------------

% Initialize output array
Num_signal = size(Pulse_signal, 1);
Pulse_shape_discrimination_factor = zeros(Num_signal, 1);

% Process signals in parallel
parfor i = 1:Num_signal
    % Get current pulse signal
    Pulse_signal_vector = Pulse_signal(i, :);
    
    % Find pulse peak amplitude and position
    [Max_pulse, Max_index] = max(Pulse_signal_vector);
    
    % Define amplitude thresholds relative to peak
    Upper_threshold = Max_pulse * 0.6;  % 60% threshold
    Lower_threshold = Max_pulse * 0.1;  % 10% threshold

    % Find threshold crossing points on falling edge
    [~, Min_index_A] = min(abs(Pulse_signal_vector(Max_index:end) - Upper_threshold));
    [~, Min_index_B] = min(abs(Pulse_signal_vector(Max_index:end) - Lower_threshold));
    
    % Convert to absolute indices
    Min_index_A = Min_index_A + Max_index - 1;  % Upper threshold position
    Min_index_B = Min_index_B + Max_index - 1;  % Lower threshold position
    
    % Validate threshold crossings
    if Min_index_A < Min_index_B && ...        % Upper occurs before lower
       Min_index_A > Max_index && ...          % Both after peak
       Min_index_B <= length(Pulse_signal_vector)  % Within signal bounds
        % Calculate falling edge slope
        Pulse_shape_discrimination_factor(i) = ...
            (Pulse_signal_vector(Min_index_B) - Pulse_signal_vector(Min_index_A)) / ...
            (Min_index_B - Min_index_A);
    else
        Pulse_shape_discrimination_factor(i) = NaN;  % Invalid threshold crossing
    end
end

end
