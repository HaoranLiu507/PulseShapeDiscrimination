function [Pulse_shape_discrimination_factor] = RCNN_method(Pulse_signal, ROI_end)

%--------------------------------------------------------------------------
% Performs pulse shape discrimination using Random-Coupled Neural Network.
%
% Uses a neural network with randomized connectivity to transform signals
% into activation patterns (Ignition Maps). The discrimination factor is
% computed as the sum of activations in a region of interest after the
% pulse peak.
%
% Input:
%   Pulse_signal: Matrix of pulse signals to analyze
%   ROI_end: Length of region of interest after peak for feature extraction
%           (default: 123 points)
%
% Reference:
% Liu, Haoran, et al. "Random-coupled Neural Network." Electronics 13.21 
% (2024): 4297.
%--------------------------------------------------------------------------

% Set default ROI length
if nargin < 2
    ROI_end = 123;
end

% Initialize output array
Num_signal = size(Pulse_signal, 1);
Pulse_shape_discrimination_factor = zeros(Num_signal, 1);

% Process signals in parallel
parfor i = 1:Num_signal
    Pulse_signal_vector = Pulse_signal(i, :);

    % Generate neural activation map
    Ignition_map = RCNN(Pulse_signal_vector);

    % Find signal peak position
    [~, Maxposition] = max(Pulse_signal_vector);

    % Define ROI bounds
    End_index = min(Maxposition + ROI_end, length(Ignition_map));
    
    % Calculate discrimination factor as sum of activations in ROI
    SUM = sum(Ignition_map(Maxposition:End_index));
    Pulse_shape_discrimination_factor(i) = SUM;
end

end
