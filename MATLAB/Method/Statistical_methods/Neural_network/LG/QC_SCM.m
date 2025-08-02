function [Ignition_map] = QC_SCM(Pulse_signal_vector)

%--------------------------------------------------------------------------
% Quasi-Continuous Spiking Cortical Model for neural signal processing.
%
% Implements a spiking neural network with dynamic thresholds and lateral
% connections to generate an activation map (ignition map) from input pulses.
%
% Input:
%   Pulse_signal_vector: Signal to process
%
% Output:
%   Ignition_map: Accumulated neural activation history
%--------------------------------------------------------------------------

% Get signal dimensions
[Rows, Column] = size(Pulse_signal_vector);

% Define lateral connection weights
Matrix = [0 0 0;
          0.44 0 0.44;
          0 0 0];

% Initialize neural states
Time_pluse_sequence = zeros(Rows, Column);  % Neural output
Neural_potential = Time_pluse_sequence;      % Membrane potential
Ignition_map = Time_pluse_sequence;         % Activation accumulator
Dynamic_threshold = Time_pluse_sequence + 1; % Adaptive threshold

% Neural dynamics parameters
f = 0.38;  % Membrane potential decay
g = 0.8;   % Threshold decay
h = 8.45;  % Activation strength
t = 0.5;   % Time step
Iterations = 50;

% Apply time step scaling
g = power(g, t);
f = power(f, t);

% Simulate neural dynamics
for t = 1:t:Iterations
    % Update membrane potential
    % V = decay * V + input * lateral_connections + direct_input
    Neural_potential = f .* Neural_potential + Pulse_signal_vector .*...
    conv2(Time_pluse_sequence, Matrix, 'same') + Pulse_signal_vector;
    
    % Update adaptive threshold
    Dynamic_threshold = g .* Dynamic_threshold + h .* Time_pluse_sequence;

    % Compute neural activation
    Pulse_generator = 1 ./ (1 + exp(Dynamic_threshold - Neural_potential));  % Sigmoid
    Time_pluse_sequence = double(Pulse_generator > 0.5);                     % Threshold

    % Accumulate activation history
    Ignition_map = Ignition_map + Time_pluse_sequence;
end

end

