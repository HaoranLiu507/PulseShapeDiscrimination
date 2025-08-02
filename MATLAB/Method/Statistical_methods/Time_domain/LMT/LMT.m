function [Pulse_shape_discrimination_factor] = LMT(Pulse_signal)

%--------------------------------------------------------------------------
% Log Mean Time (LMT) method for pulse shape discrimination.
%
% Discriminates particle types by analyzing the temporal characteristics
% of pulse signals. Computes the natural logarithm of the amplitude-weighted
% mean time to capture differences in pulse decay patterns.
%
% Input:
%   Pulse_signal: Matrix of pulse signals to analyze [n_signals × signal_length]
%
% Output:
%   Pulse_shape_discrimination_factor: Log of mean arrival time
%                                    [n_signals × 1]
%
% Algorithm:
%   1. Create time series t = [1, 2, ..., signal_length]
%   2. For each signal S(t):
%      - Calculate weighted mean time: μt = Σ(t·S(t)) / Σ(S(t))
%      - Compute discrimination factor: ln(μt)
%
% Note:
%   - Returns NaN for signals with zero total amplitude
%   - Time series assumes uniform sampling (1-indexed)
%
% References:
% [1] Dutta, S., et al. "Pulse shape simulation and discrimination using
%     machine learning techniques." Journal of Instrumentation 18.03 (2023):
%     P03038.
% [2] Lee, H. S., et al. "Neutron calibration facility with an Am-Be source
%     for pulse shape discrimination measurement of CsI (Tl) crystals."
%     Journal of Instrumentation 9.11 (2014): P11015.
%--------------------------------------------------------------------------

% Generate time series (1-indexed sample points)
Time_series = 1:size(Pulse_signal, 2);

% Initialize output array
Num_signals = size(Pulse_signal, 1);
Pulse_shape_discrimination_factor = NaN(Num_signals, 1);

% Process each pulse signal
for i = 1:Num_signals
    Signal = Pulse_signal(i, :);
    
    % Calculate amplitude-weighted temporal moments
    Weighted_sum = sum(Signal .* Time_series);  % First moment (Σ t·S(t))
    Total_sum = sum(Signal);                    % Zero moment (Σ S(t))
    
    % Compute log mean time if signal is non-zero
    if Total_sum ~= 0
        Mean_time = Weighted_sum / Total_sum;   % μt = Σ(t·S(t)) / Σ(S(t))
        Pulse_shape_discrimination_factor(i) = log(Mean_time);
    end
    % NaN is maintained for zero-amplitude signals
end

end
