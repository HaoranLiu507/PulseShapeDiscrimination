function wavelet = Wavelet_Haar(scale, shift)

%--------------------------------------------------------------------------
% Generates scaled and shifted Haar wavelet.
%
% Creates a normalized Haar wavelet function with specified scaling and
% translation parameters for time-frequency analysis.
% 
% Inputs:
%   scale: Dilation factor for frequency scaling
%   shift: Translation factor for time shifting
%
% Returns:
%   wavelet: Normalized Haar wavelet values
%--------------------------------------------------------------------------
    
    % Create time vector with 0.1 resolution
    t = -shift:0.1:shift;

    % Generate scaled wavelet
    wavelet = Haar(t / scale);
end

function w = Haar(t)

%--------------------------------------------------------------------------
% Implements the Haar wavelet basis function.
%
% Defines the standard Haar wavelet: 
%   1  for 0 <= t < 0.5
%   -1 for 0.5 <= t < 1
%   0  otherwise
%
% Input:
%   t: Time points for wavelet evaluation
%
% Returns:
%   w: Normalized Haar wavelet values
%--------------------------------------------------------------------------

    % Initialize wavelet vector
    w = zeros(size(t));

    % Define wavelet segments
    w(0 <= t & t < 0.5) = 1;   % Positive half
    w(0.5 <= t & t < 1) = -1;  % Negative half

    % L2 normalization
    w = w / norm(w);
end
