function [Pulse_shape_discrimination_factor] = LG_method(Pulse_signal, m)

%--------------------------------------------------------------------------
% Performs pulse shape discrimination using Ladder Gradient (LG) method.
% 
% Uses a spiking neural network to generate activation maps, then computes
% discrimination factor as the slope between peak activation and the m-th
% occurrence of the most frequent activation level after the peak.
%
% Input:
%   Pulse_signal: Matrix of pulse signals to analyze
%   m: Number of mode occurrences to look for after peak (default: 20)
%
% Reference:
% Liu, Hao-Ran, et al. "Discrimination of neutron and gamma ray using the 
% ladder gradient method and analysis of filter adaptability." Nuclear Science 
% and Techniques 33.12 (2022): 159.
%--------------------------------------------------------------------------
    
% Set default mode count
if nargin < 2
    m = 20;
end

% Initialize output array
Num_signal = size(Pulse_signal, 1);
Pulse_shape_discrimination_factor = zeros(Num_signal, 1);

% Process signals in parallel
parfor i = 1:Num_signal
    Pulse_signal_vector = Pulse_signal(i, :);

    % Generate neural activation map
    Ignition_map = QC_SCM(Pulse_signal_vector);

    % Find signal peak position
    [~, Maxposition] = max(Pulse_signal_vector);

    % Find most common activation level
    Frequent_number = mode(Ignition_map);

    % Search for m-th occurrence of most common level after peak
    ROI = Ignition_map(Maxposition:end);
    X = find(ROI == Frequent_number, m);

    % Calculate gradient if m occurrences found
    if length(X) >= m
        % Position of m-th occurrence (relative to full signal)
        Second_point_position = X(m) + Maxposition - 1;

        % Calculate discrimination factor:
        % slope = (peak_value - m_th_value) / (peak_pos - m_th_pos)
        Pulse_shape_discrimination_factor(i) = ...
            (Ignition_map(Maxposition) - Ignition_map(Second_point_position)) / ...
            (Maxposition - Second_point_position);
    else
        % Not enough occurrences found
        Pulse_shape_discrimination_factor(i) = NaN;
    end
end

end
