function [Pulse_shape_discrimination_factor] = WT1(Pulse_signal, Scale, Shift)

%--------------------------------------------------------------------------
% Performs pulse shape discrimination using Haar Wavelet Transform (WT1).
% 
% Uses multi-scale wavelet decomposition to analyze signal features at 
% different frequencies. Discrimination factor is computed as the ratio of
% wavelet energies at specific scales where neutron and gamma signals show
% maximum differences.
%
% Inputs:
%   Scale: Scaling factor for frequency analysis (default: 100)
%   Shift: Time-domain translation parameter (default: 50)
%
% Reference:
% Yousefi, S., L. Lucchese, and M. D. Aspinall. "Digital discrimination of 
% neutrons and gamma-rays in liquid scintillators using wavelets." 
% Nuclear Instruments and Methods in Physics Research Section A: 
% Accelerators, Spectrometers, Detectors and Associated Equipment 598.2 
% (2009): 551-555.
%--------------------------------------------------------------------------
    
% Set default parameters
if nargin < 2
    Scale = 100;   % Number of frequency scales
end
if nargin < 3
    Shift = 50;    % Time shift window
end

% Initialize output array
Num_signal = size(Pulse_signal, 1);
Pulse_shape_discrimination_factor = zeros(Num_signal, 1);

% Process signals in parallel
parfor i = 1:Num_signal
    Pulse_signal_vector = Pulse_signal(i, :);

    % Compute wavelet scale decomposition
    Scale_function = Scale_Haar(Pulse_signal_vector, Scale, Shift);

    % Calculate discrimination factor:
    % Ratio of wavelet energies at scales 28 and 40
    % These scales show maximum neutron/gamma separation
    Pulse_shape_discrimination_factor(i) = Scale_function(28) / Scale_function(40);
end

end
