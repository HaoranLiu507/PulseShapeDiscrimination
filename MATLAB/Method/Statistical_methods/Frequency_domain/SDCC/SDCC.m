function [Pulse_shape_discrimination_factor] = SDCC(Pulse_signal, ROI_left_endpoint)

%--------------------------------------------------------------------------
% Performs pulse shape discrimination using Simplified Digital Charge 
% Collection (SDCC).
% 
% Calculates discrimination factor as the log of squared sum over a region
% of interest (ROI) in the pulse tail, where neutron and gamma signals
% show maximum decay rate differences.
%
% Input:
%   ROI_left_endpoint: Starting point of the ROI in the pulse tail
%
% Reference:
% Shippen, David I., Malcolm J. Joyce, and Michael D. Aspinall. 
% "A wavelet packet transform inspired method of neutron-gamma discrimination."
% IEEE Transactions on Nuclear Science 57.5 (2010): 2617-2624. 325-332.
%--------------------------------------------------------------------------
    
% Set default ROI start point
if nargin < 2
    ROI_left_endpoint = 70;  % Default position in pulse tail
end

% Initialize output array
Num_signal = size(Pulse_signal, 1);
Pulse_shape_discrimination_factor = zeros(Num_signal, 1);

% Process signals in parallel
parfor i = 1:Num_signal
    Pulse_signal_vector = Pulse_signal(i, :);

    % Extract region of interest from pulse tail
    ROI = Pulse_signal_vector(ROI_left_endpoint:end);

    % Calculate discrimination factor:
    % log(sum(ROI^2))
    Pulse_shape_discrimination_factor(i) = log(sum(ROI.^2));
end

end
