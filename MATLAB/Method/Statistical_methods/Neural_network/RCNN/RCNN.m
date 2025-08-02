function [Ignition_map] = RCNN(Pulse_signal_vector)

%--------------------------------------------------------------------------
% Random-Coupled Neural Network implementation for signal analysis.
%
% Implements a neural network with stochastic connectivity patterns using
% Gaussian-modulated random weights. Features dynamic thresholds and
% adaptive neural responses for robust signal processing.
%
% Input:
%   Pulse_signal_vector: Signal to process
%
% Output:
%   Ignition_map: Accumulated neural activation history
%--------------------------------------------------------------------------

% Get signal dimensions
[Rows, Column] = size(Pulse_signal_vector);

% Neural dynamics parameters
B = 0.4;      % Coupling strength
V = 1;        % Input scaling
aT = 0.709;   % Threshold decay
vT = 0.101;   % Threshold coupling
aF = 0.205;   % Feedback decay
t = 20;       % Simulation iterations

% Gaussian kernel parameters
dimension = 9;  % Kernel size
sigma1 = 4;    % Primary Gaussian spread
sigma2 = 6;    % Secondary Gaussian spread

% Create base connectivity pattern
% Zero-centered Gaussian kernel for lateral connections
Gaussian_kernel = fspecial('gaussian', dimension, sigma1);
Gaussian_kernel((dimension + 1) / 2, (dimension + 1) / 2) = 0;

% Initialize neural states
Y = zeros(Rows, Column);  % Neural output
U = Y;                    % Membrane potential
E = Y + 1;               % Adaptive threshold
Ignition_map = Y;        % Activation history

% Simulate neural dynamics
for i = 1:t
    % Generate stochastic connectivity
    % Modulate Gaussian kernel with random weights
    Weight_matrix_random = Gaussian_kernel .* Rand_matrix(dimension, 0.1, 'norm', sigma2);

    % Compute lateral interactions
    L = conv2(Y, Weight_matrix_random, 'same');

    % Update membrane potential
    % U(n) = exp(-aF)*U(n-1) + S*(1 + V*B*L)
    U = U .* exp(-aF) + Pulse_signal_vector .* (1 + V * B * L);
    
    % Generate neural activation
    Y = im2double(U > E);  % Threshold crossing
    
    % Update adaptive threshold
    % E(n) = exp(-aT)*E(n-1) + vT*Y(n)
    E = exp(-aT) * E + vT * Y;

    % Accumulate activation history
    Ignition_map = Ignition_map + Y;
end

end
