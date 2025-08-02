function [Pulse_shape_discrimination_factor] = DFT(Pulse_signal)

%--------------------------------------------------------------------------
% Performs pulse shape discrimination using Discrete Fourier Transform (DFT).
% 
% Transforms signals to frequency domain and uses the ratio of total DFT power
% to the product of first DST and DCT components as discrimination factor.
%
% Reference:
% Safari, M. J., et al. "Discrete fourier transform method for 
% discrimination of digital scintillation pulses in mixed neutron-gamma 
% fields." IEEE Transactions on Nuclear Science 63.1 (2016): 325-332.
%--------------------------------------------------------------------------
    
% Get total number of input signals
Num_signal = size(Pulse_signal, 1);
Pulse_shape_discrimination_factor = zeros(Num_signal, 1);

% Process each signal in parallel
parfor i = 1:Num_signal
    Pulse_signal_vector = Pulse_signal(i, :);

    % Get signal peak position
    [~, Max_position] = max(Pulse_signal_vector);

    % Extract signal from peak onwards
    Pulse_signal_vector = Pulse_signal_vector(Max_position:end);

    % Calculate transforms
    DCT = dct(Pulse_signal_vector);
    DST = dst(Pulse_signal_vector);
    DFT = abs(fft(Pulse_signal_vector));

    % Calculate discrimination factor:
    % (Total DFT power) / (DST[0] * DCT[0] * Signal sum)
    Pulse_shape_discrimination_factor(i) = ...
        (sum(DFT.^2) / (DST(1) * DCT(1))) * (1 / sum(Pulse_signal_vector));
end

end
