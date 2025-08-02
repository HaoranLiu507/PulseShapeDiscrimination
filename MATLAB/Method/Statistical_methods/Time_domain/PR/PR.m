function [Pulse_shape_discrimination_factor] = PR(Pulse_signal)

%--------------------------------------------------------------------------
% Pattern Recognition (PR) method for pulse shape discrimination.
%
% Discriminates particle types by computing the angular separation between
% pulse signals and a reference signal in vector space. Uses the geometric
% relationship between signals to capture shape differences.
%
% Input:
%   Pulse_signal: Matrix of pulse signals to analyze [n_signals × signal_length]
%
% Output:
%   Pulse_shape_discrimination_factor: Angular separation from reference (rad)
%                                    [n_signals × 1]
%
% Algorithm:
%   1. Select reference signal (second signal in matrix)
%   2. For each signal:
%      - Align with reference at peak position
%      - Compute angle θ between vectors:
%        θ = arccos((x·r)/(||x||·||r||))
%      where x is signal vector, r is reference vector
%
% Mathematical basis:
%   - Vectors: Signal amplitudes after peak
%   - Similarity: Cosine of angle between vectors
%   - Range: θ ∈ [0, π] radians
%
% Note:
%   - Requires at least 2 signals (one for reference)
%   - Zero-norm vectors will cause errors
%   - Cosine is clamped to [-1,1] for numerical stability
%
% Reference:
% Takaku, D., T. Oishi, and M. Baba. "Development of neutron-gamma
% discrimination technique using pattern-recognition method with digital
% signal processing." Prog. Nucl. Sci. Technol 1 (2011): 210-213.
%--------------------------------------------------------------------------

% Initialize output array
Num_signal_row = size(Pulse_signal, 1);
Pulse_shape_discrimination_factor = zeros(Num_signal_row, 1);

% Validate input size
if Num_signal_row < 2
    error('At least 2 signals required (1 for analysis, 1 for reference).');
end

% Select reference signal (second row of input matrix)
Reference_signal_original = Pulse_signal(2, :);

% Process signals in parallel
parfor i = 1:Num_signal_row
    % Get current signal
    Pulse_signal_vector = Pulse_signal(i, :);
    
    % Align signals at peak
    [~, Max_position] = max(Pulse_signal_vector);
    Pulse_signal_vector = Pulse_signal_vector(Max_position:end);  % Post-peak region
    Reference_signal = Reference_signal_original(Max_position:end);  % Aligned reference

    % Compute vector operations
    Scalar_product = dot(Pulse_signal_vector, Reference_signal);  % x·r
    Norm_Reference_signal = norm(Reference_signal);               % ||r||
    Norm_pulse_signal = norm(Pulse_signal_vector);               % ||x||

    % Validate vector norms
    if (Norm_pulse_signal == 0 || Norm_Reference_signal == 0)
        error('Zero-norm vector detected at signal %d. Valid signals required.', i);
    end
    
    % Calculate cosine similarity
    Cos_angle = Scalar_product / (Norm_pulse_signal * Norm_Reference_signal);

    % Ensure numerical stability for arccos
    Cos_angle = max(min(Cos_angle, 1), -1);  % Clamp to [-1,1]
    
    % Compute angular separation
    Pulse_shape_discrimination_factor(i) = abs(acos(Cos_angle));  % θ in radians
end

end