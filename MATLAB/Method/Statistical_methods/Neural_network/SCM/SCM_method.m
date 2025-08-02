function [Pulse_shape_discrimination_factor] = SCM_method(Pulse_signal, ROI_end)

%--------------------------------------------------------------------------
% Performs pulse shape discrimination using Spiking Cortical Model (SCM).
%
% Uses a biologically-inspired neural network that mimics cortical neuron
% dynamics to transform signals into activation patterns. The discrimination
% factor is computed as the sum of neural activations in a region of
% interest after the pulse peak.
%
% Input:
%   Pulse_signal: Matrix of pulse signals to analyze
%   ROI_end: Length of region of interest after peak for feature extraction
%           (default: 120 points)
%
% Reference:
% Liu, Bing-Qi, et al. "Discrimination of neutrons and gamma-rays in plastic 
% scintillator based on spiking cortical model." Nuclear Engineering and 
% Technology 55.9 (2023): 3359-3366.
%--------------------------------------------------------------------------

% Set default ROI length
if nargin < 2
    ROI_end = 120;
end

% Initialize output array
Num_signal = size(Pulse_signal, 1);
Pulse_shape_discrimination_factor = zeros(Num_signal, 1);

% Process signals in parallel
parfor i = 1:Num_signal
    Pulse_signal_vector = Pulse_signal(i, :);

    % Generate neural activation map
    Ignition_map = SCM(Pulse_signal_vector);

    % Find signal peak position
    [~, Maxposition] = max(Pulse_signal_vector);

    % Define ROI bounds
    End_index = min(Maxposition + ROI_end, length(Ignition_map));

    % Calculate discrimination factor as sum of activations in ROI
    SUM = sum(Ignition_map(Maxposition:End_index));
    Pulse_shape_discrimination_factor(i) = SUM;
end

end