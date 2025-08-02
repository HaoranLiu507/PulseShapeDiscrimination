function [Pulse_shape_discrimination_factor] = GP(Pulse_signal)

%--------------------------------------------------------------------------
% Gatti Parameter (GP) method for pulse shape discrimination.
%
% Discriminates particle types using a weighted linear operation based on
% reference signals from known particle classes. The method computes optimal
% weights that maximize separation between particle types.
%
% Input:
%   Pulse_signal: Matrix of pulse signals to analyze [n_signals × signal_length]
%
% Output:
%   Pulse_shape_discrimination_factor: Weighted sum using Gatti parameters
%                                    [n_signals × 1]
%
% Algorithm:
%   1. Load normalized reference signals for two particle classes
%   2. Compute Gatti parameter (P) from reference signals:
%      P = (S1 - S2)/(S1 + S2 + ε)
%      where S1, S2 are class references, ε is numerical stability term
%   3. Calculate discrimination factor as dot product of signal with P
%
% Required files:
%   - EJ299_33_AmBe_9414_neutron_ref.txt (Class 1 reference)
%   - EJ299_33_AmBe_9414_gamma_ref.txt (Class 2 reference)
%
% Reference:
% Gatti, E., et al. "A new linear method of discrimination between elementary
% particles in scintillation counters." Nuclear Electronics II. Proceedings
% of the Conference on Nuclear Electronics. V. II. 1962.
%--------------------------------------------------------------------------

% Initialize output array
Num_signals = size(Pulse_signal, 1);
Pulse_shape_discrimination_factor = zeros(Num_signals, 1);

% Load reference signals for both particle classes
try
    class1_signal = load('Data/Reference_signal/EJ299_33_AmBe_9414_neutron_ref.txt');
catch
    error('Class 1 (neutron) reference signal file not found. Check path: Data/Reference_signal/');
end

try
    class2_signal = load('Data/Reference_signal/EJ299_33_AmBe_9414_gamma_ref.txt');
catch
    error('Class 2 (gamma) reference signal file not found. Check path: Data/Reference_signal/');
end

% Validate signal dimensions
if length(class1_signal) ~= size(Pulse_signal, 2) || ...
   length(class2_signal) ~= size(Pulse_signal, 2)
    error('Reference signals must match pulse signal length.');
end

if Num_signals < 2
    error('At least two pulse signals required for analysis.');
end

% Normalize reference signals to [0,1] range
class1_signal = (class1_signal - min(class1_signal)) / ...
                (max(class1_signal) - min(class1_signal));
class2_signal = (class2_signal - min(class2_signal)) / ...
                (max(class2_signal) - min(class2_signal));

% Compute Gatti parameter (P)
% Add small epsilon (1e-10) to prevent division by zero
P = (class1_signal - class2_signal) ./ (class1_signal + class2_signal + 1e-10);

% Calculate discrimination factors using weighted sum
for i = 1:Num_signals
    Pulse_shape_discrimination_factor(i) = dot(Pulse_signal(i, :), P);
end

end