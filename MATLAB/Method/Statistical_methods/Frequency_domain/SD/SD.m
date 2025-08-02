function [Pulse_shape_discrimination_factor] = SD(pulse_signal)

%--------------------------------------------------------------------------
% Performs pulse shape discrimination using Scalogram-based Discrimination (SD).
% 
% Uses continuous wavelet transform (CWT) to generate scalograms and applies
% a trained discrimination mask to classify mixed neutron/gamma signals.
% The mask is created from labeled training data of both signal types.
% 
% Reference:
% Abdelhakim, Assem, and Ehab Elshazly. "Efficient pulse shape discrimination
% using scalogram image masking and decision tree." Nuclear Instruments and
% Methods in Physics Research Section A: Accelerators, Spectrometers, 
% Detectors and Associated Equipment 1050 (2023): 168140.
%--------------------------------------------------------------------------
    % Load and normalize reference signals
    class1_signals = load('Data/Reference_signal/EJ299_33_AmBe_9414_neutron_ref.txt');
    class2_signals = load('Data/Reference_signal/EJ299_33_AmBe_9414_gamma_ref.txt');
    class1_signals = normalize_signals(class1_signals);
    class2_signals = normalize_signals(class2_signals);

    % Process input signals
    mixed_signals = normalize_signals(pulse_signal);

    % Create discrimination mask from training data
    mask = generate_discrimination_mask(class1_signals, class2_signals, 'mexh', 1:50, 127);

    % Apply mask to get discrimination factors
    Pulse_shape_discrimination_factor = discriminate_signals(mixed_signals, mask, 'mexh', 1:50, 127);
end

function normalized_signals = normalize_signals(signals)
    % Normalize signals by their maximum absolute values
    max_abs_value = max(abs(signals), [], 2);  % Row-wise maximum
    normalized_signals = signals ./ max_abs_value;
end

function grayscale_matrix = convert_to_grayscale(matrix)
    % Convert matrix values to grayscale range [0, 255]
    % 
    % Args:
    %   matrix: Input data matrix
    % Returns:
    %   Normalized matrix scaled to [0, 255]
    
    min_val = min(matrix(:));
    max_val = max(matrix(:));
    grayscale_matrix = 255 * (matrix - min_val) / (max_val - min_val + 1e-9);
end

function mask = generate_discrimination_mask(class1_signals, class2_signals, wavelet, scales, threshold)
    % Generate binary mask highlighting regions of maximum discrimination
    % between two signal classes using CWT scalograms
    
    % Default parameters
    if nargin < 3
        wavelet = 'mexh';     % Mexican hat wavelet
    end
    if nargin < 4
        scales = 1:50;        % Wavelet scales
    end
    if nargin < 5
        threshold = 127;      % Grayscale threshold
    end

    % Select training subset
    num_samples = min(size(class1_signals, 1), size(class2_signals, 1));
    training_size = max(1, floor(num_samples / 15));
    difference_count = zeros(length(scales), size(class1_signals, 2));

    % Accumulate differences between class scalograms
    for sample_idx = 1:training_size
        % Get sample pair
        pulse1 = class1_signals(sample_idx, :);
        pulse2 = class2_signals(sample_idx, :);

        % Calculate scalograms
        [coeffs1, ~] = cwt(pulse1, scales, wavelet, 'scal');
        [coeffs2, ~] = cwt(pulse2, scales, wavelet, 'scal');
        energy1 = abs(coeffs1) .^ 2;
        energy2 = abs(coeffs2) .^ 2;

        % Convert to binary ROIs
        gray1 = convert_to_grayscale(energy1);
        gray2 = convert_to_grayscale(energy2);
        roi1 = gray1 >= threshold;
        roi2 = gray2 >= threshold;

        % Add to difference map
        difference_count = difference_count + abs(roi1 - roi2);
    end

    % Validate and create mask
    nonzero_count = nnz(difference_count);
    if nonzero_count == 0
        error('No discrimination features found between signal classes');
    end

    % Create mask using average difference as threshold
    difference_threshold = sum(difference_count(:)) / nonzero_count;
    mask = difference_count >= difference_threshold;
end

function Pulse_shape_discrimination_factor = discriminate_signals(mixed_signals, mask, wavelet, scales, threshold)
    % Apply discrimination mask to mixed signals and compute discrimination factors
    
    % Default parameters
    if nargin < 4
        scales = 1:50;
    end
    if nargin < 5
        threshold = 127;
    end
    
    num_signals = size(mixed_signals, 1);
    Pulse_shape_discrimination_factor = zeros(num_signals, 1);
    
    % Process each signal
    for i = 1:num_signals
        signal = mixed_signals(i, :);
        [coeffs, ~] = cwt(signal, scales, wavelet, 'scal');
        scalogram = abs(coeffs).^2;
        gray_scalogram = convert_to_grayscale(scalogram);
        roi = gray_scalogram >= threshold;
        
        % Calculate discrimination factor as ratio of matching mask points
        masked_roi = roi(mask);
        Pulse_shape_discrimination_factor(i) = sum(masked_roi(:)) / sum(mask(:));
    end
end
