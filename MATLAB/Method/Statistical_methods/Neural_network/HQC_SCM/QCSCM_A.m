function Ignition_map_A = QCSCM_A(Pulse_signal_vector)

%--------------------------------------------------------------------------
% Neural processor for early pulse features using QCSCM type A.
%
% Implements a quasi-continuous spiking cortical model optimized for
% processing the initial segment of pulse signals. Uses genetically
% optimized parameters for early feature detection.
%
% Input:
%   Pulse_signal_vector: Signal segment to process
%
% Output:
%   Ignition_map_A: Neural activation map for early features
%--------------------------------------------------------------------------

% Get signal dimensions
[Rows, Column] = size(Pulse_signal_vector);

% Define lateral connection weights
Weight_matrix = [0 0 0;
                0.5 0 0.5;
                0 0 0];

% Model parameters
Iterations = 50;
t = 0.5;  % Time step

% Genetically optimized parameters for type A processing
f = 0.3351;  % Membrane potential decay
g = 0.8359;  % Refractory decay
h = 7.9872;  % Activation strength
g = power(g, t);
f = power(f, t);

% Initialize neural states
Y = zeros(Rows, Column);  % Neural output
U = Y;                    % Membrane potential
Ignition_map_A = Y;       % Activation accumulator
E = Y + 1;               % Refractory state

% Simulate neural dynamics
for Iterations = 1:t:Iterations
    % Update membrane potential
    % U = decay * U + input * lateral_connections + direct_input
    U = f .* U + Pulse_signal_vector .* conv2(Y, Weight_matrix, 'same') + Pulse_signal_vector;
    
    % Update refractory state
    E = g .* E + h .* Y;
    
    % Compute neural activation
    X = 1 ./ (1 + exp(E - U));     % Sigmoid activation
    Y = double(X > 0.5);           % Binary threshold
    
    % Accumulate activation history
    Ignition_map_A = Ignition_map_A + Y;
end

end
