function Ignition_map_B = QCSCM_B(Pulse_signal_vector)

%--------------------------------------------------------------------------
% Neural processor for late pulse features using QCSCM type B.
%
% Implements a quasi-continuous spiking cortical model optimized for
% processing the tail segment of pulse signals. Uses genetically
% optimized parameters for late feature detection.
%
% Input:
%   Pulse_signal_vector: Signal segment to process
%
% Output:
%   Ignition_map_B: Neural activation map for late features
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

% Genetically optimized parameters for type B processing
f = 0.3389;  % Membrane potential decay
g = 0.7831;  % Refractory decay
h = 8.6316;  % Activation strength
g = power(g, t);
f = power(f, t);

% Initialize neural states
Y = zeros(Rows, Column);  % Neural output
U = Y;                    % Membrane potential
Ignition_map_B = Y;       % Activation accumulator
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
    Ignition_map_B = Ignition_map_B + Y;
end

end
