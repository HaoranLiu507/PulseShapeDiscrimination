function D = Rand_matrix(dimension, probability, flag, sigma)

%--------------------------------------------------------------------------
% Generates stochastic connectivity matrices for neural networks.
%
% Creates random weight matrices using either uniform or Gaussian-modulated
% probability distributions. Used to introduce controlled randomness in
% neural network connectivity patterns.
%
% Inputs:
%   dimension: Size of square matrix
%   probability: Threshold for random connections
%   flag: Distribution type ('norm' for Gaussian)
%   sigma: Standard deviation for Gaussian distribution
%
% Output:
%   D: Binary random weight matrix
%--------------------------------------------------------------------------

% Initialize base matrix
D = ones(dimension, dimension);

% Generate connectivity pattern
if strcmp(flag, 'norm') 
    % Create normalized Gaussian kernel
    D = fspecial('gaussian', dimension, sigma);
    
    % Scale to unit center weight
    S = 1 / D((dimension + 1) / 2, (dimension + 1) / 2);
    D = D .* S;
    
    % Convert to binary using Gaussian probabilities
    D = rand(dimension) < D;
else 
    % Generate uniform random connectivity
    D = rand(dimension) < (D .* probability);
end

end