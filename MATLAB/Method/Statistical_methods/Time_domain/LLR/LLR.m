function [Pulse_shape_discrimination_factor] = LLR(Pulse_signal)

%--------------------------------------------------------------------------
% Log-Likelihood Ratio (LLR) method for pulse shape discrimination.
%
% Discriminates particle types using probability-based analysis of pulse
% shapes. Computes discrimination factors using probability mass functions
% (PMF) as an efficient approximation of the photon detection time
% probability density function (PDF).
%
% Input:
%   Pulse_signal: Matrix of pulse signals to analyze [n_signals × signal_length]
%
% Output:
%   Pulse_shape_discrimination_factor: LLR-based discrimination factors
%                                    [n_signals × 1]
%
% Algorithm:
%   1. Load and normalize reference signals for both particle classes
%   2. Compute PMFs for both reference signals
%   3. Calculate LLR values: -log(PMF1/PMF2)
%   4. Compute discrimination factor as weighted sum with LLR values
%
% Advantages over traditional PDF methods:
%   - Avoids complex singlet/triplet component mixing
%   - No detector resolution convolution needed
%   - Direct PMF calculation from amplitudes
%   - Computationally efficient
%
% Required files:
%   - EJ299_33_AmBe_9414_neutron_ref.txt (Class 1 reference)
%   - EJ299_33_AmBe_9414_gamma_ref.txt (Class 2 reference)
%
% References:
% [1] Adhikari, P., et al. "Pulse-shape discrimination against low-energy
%     Ar-39 beta decays in liquid argon with 4.5 tonne-years of DEAP-3600
%     data." The European Physical Journal C 81 (2021): 1-13.
% [2] Akashi-Ronquest, M., et al. "Improving photoelectron counting and
%     particle identification in scintillation detectors with Bayesian
%     techniques." Astroparticle physics 65 (2015): 40-54.
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

% Normalize reference signals to [0,1] range
class1_signal = (class1_signal - min(class1_signal)) / ...
                (max(class1_signal) - min(class1_signal));
class2_signal = (class2_signal - min(class2_signal)) / ...
                (max(class2_signal) - min(class2_signal));

% Compute probability mass functions for reference signals
pmf1 = calculate_pmf(class1_signal);
pmf2 = calculate_pmf(class2_signal);

% Calculate log-likelihood ratio values
epsilon = 1e-10;  % Numerical stability term
LLR_values = -log((pmf1 + epsilon) ./ (pmf2 + epsilon));

% Compute discrimination factors using LLR weights
for i = 1:Num_signals
    Pulse_shape_discrimination_factor(i) = dot(Pulse_signal(i, :), LLR_values);
end

    function pmf = calculate_pmf(data)
        %----------------------------------------------------------------------
        % Calculates the probability mass function (PMF) for a signal sequence.
        %
        % The PMF represents the discrete probability distribution of signal
        % amplitudes, providing an efficient alternative to continuous PDFs.
        %
        % Input:
        %   data: Signal sequence [1 × signal_length]
        %
        % Output:
        %   pmf: Probability for each amplitude value [1 × signal_length]
        %----------------------------------------------------------------------
        
        if isempty(data)
            pmf = [];
            return;
        end
        
        % Find unique amplitude values and their occurrences
        [~, ~, idx] = unique(data);
        counts = accumarray(idx, 1);
        
        % Convert counts to probabilities
        total_samples = length(data);
        probs = counts / total_samples;
        
        % Map probabilities back to original sequence
        pmf = probs(idx);
    end
end

