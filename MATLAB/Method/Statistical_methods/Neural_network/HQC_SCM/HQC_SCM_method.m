function [Pulse_shape_discrimination_factor] = HQC_SCM_method(Pulse_signal, ROI_end)

%--------------------------------------------------------------------------
% Performs pulse shape discrimination using Heterogeneous Quasi-Continuous 
% Spiking Cortical Model (HQC-SCM).
%
% Uses two specialized neural networks to process different parts of the
% pulse signal. The signal is automatically segmented based on amplitude
% characteristics, and each segment is processed by a network optimized
% for its features.
%
% Input:
%   Pulse_signal: Matrix of pulse signals to analyze
%   ROI_end: Length of region of interest after peak for feature extraction
%           (default: 120 points)
%
% Reference:
% Liu, Runxi, et al. "Heterogeneous Quasi-Continuous Spiking Cortical Model
% for Pulse Shape Discrimination." Electronics 12.10 (2023): 2234.
%--------------------------------------------------------------------------

% Set default ROI length
if nargin < 2
    ROI_end = 120;
end

% Initialize variables
Num_signal = size(Pulse_signal, 1);
Pulse_shape_discrimination_factor = zeros(Num_signal, 1);

% Find segmentation point using mean signal profile
Mean_signal = sum(Pulse_signal, 1) / Num_signal;
[Max_mean_signal_value, ~] = max(Mean_signal);

% Locate segment boundary at ~5% of peak amplitude
% This point separates early and late pulse features
Segment_point = find((Mean_signal >= (0.0491 * Max_mean_signal_value)) & ...
                     (Mean_signal <= (0.0501 * Max_mean_signal_value)), 1);

% Process signals in parallel
parfor i = 1:Num_signal
    Pulse_signal_vector = Pulse_signal(i, :);

    % Generate neural activation maps
    Ignition_map_combined = HQC_SCM(Pulse_signal_vector, Segment_point);

    % Find signal peak
    [~, Maxposition] = max(Pulse_signal_vector);

    % Extract features from ROI after peak
    end_index = min(Maxposition + ROI_end, length(Ignition_map_combined));
    
    % Calculate discrimination factor as sum of neural activations in ROI
    SUM = sum(Ignition_map_combined(Maxposition:end_index));
    Pulse_shape_discrimination_factor(i) = SUM;
end

end
