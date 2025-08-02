function Scale_function = Scale_Haar(data, Scale, Shift)

%--------------------------------------------------------------------------
% Computes scale function using Haar wavelet transform.
%
% Calculates wavelet transform energies at multiple scales to capture
% signal features at different frequency resolutions.
%
% Inputs:
%   data: Input signal to analyze
%   Scale: Number of scale levels to compute
%   Shift: Time translation window size
%
% Returns:
%   Scale_function: Vector of normalized wavelet energies at each scale
%--------------------------------------------------------------------------

% Set window size for normalization
n = Shift * 2;
Scale_function = zeros(Scale, 1);

% Compute energy at each scale level
for scale = 1:Scale
    % Apply wavelet transform with scale-dependent normalization
    W = ((1:length(data)) ./ sqrt(scale)) .* conv(data, Wavelet_Haar(scale, Shift), 'same');
    
    % Calculate total wavelet energy at this scale
    Scale_function(scale) = sum(abs(W));
    
    % Normalize by window size
    Scale_function(scale) = 1 / (1 + n) * Scale_function(scale);
end

end
