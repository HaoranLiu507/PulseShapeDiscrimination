function [Pulse_shape_discrimination_factor] = FGA(Pulse_signal, Sample_frequency)

%--------------------------------------------------------------------------
% Performs pulse shape discrimination using Frequency Gradient Analysis (FGA).
% 
% Calculates discrimination factor based on the gradient between first two
% frequency components of the signal's frequency spectrum.
%
% Input:
%   Sample_frequency: Sampling rate in GSa/s (Giga Samples per Second)
%
% Reference:
% Liu, Guofu, et al. "A digital method for the discrimination of neutrons 
% and gamma rays with organic scintillation detectors using frequency
% gradient analysis." IEEE Transactions on Nuclear Science 57.3 (2010):
% 1682-1691.
%--------------------------------------------------------------------------

% Set default sampling frequency if not provided
if nargin < 2
    Sample_frequency = 1;
end

% Initialize variables
Num_signal = size(Pulse_signal, 1);
Pulse_shape_discrimination_factor = zeros(Num_signal, 1);

% Setup signal parameters
Index_signal = 1:1:length(Pulse_signal(1, :));
Length_signal = length(Pulse_signal(1, :));

% Process each signal
for i = 1:Num_signal
    Pulse_signal_vector = Pulse_signal(i, :);
    
    % Calculate DC component (zero frequency)
    X_0 = Length_signal * abs(mean(Pulse_signal_vector));
    
    % Calculate first harmonic component
    X_1 = abs(sum(Pulse_signal_vector .* cos(2 * pi * Index_signal / Length_signal))) - ...
           sum(Pulse_signal_vector .* sin(2 * pi * Index_signal / Length_signal));

    % Calculate discrimination factor:
    % (Signal length * |X_0 - X_1|) / Sample frequency
    Pulse_shape_discrimination_factor(i) = Length_signal * (abs(X_0 - X_1)) / Sample_frequency;
end

end
