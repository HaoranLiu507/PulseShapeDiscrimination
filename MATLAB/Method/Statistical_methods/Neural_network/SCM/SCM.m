function Ignition_map = SCM(Pulse_signal_vector)

%--------------------------------------------------------------------------
% Spiking Cortical Model implementation for signal analysis.
%
% Implements a neural network that simulates cortical neuron dynamics with
% membrane potential integration, adaptive thresholds, and refractory
% periods. Features local lateral connections for spatial signal processing.
%
% Input:
%   Pulse_signal_vector: Signal to process
%
% Output:
%   Ignition_map: Accumulated neural activation history
%--------------------------------------------------------------------------

% Get signal dimensions
[Rows, Column] = size(Pulse_signal_vector);

% Define local connectivity pattern
% 3x3 kernel with center-surround organization
Weight_matrix = [0.1091 0.1409 0.1091;
                 0.1409   0    0.1409;
                 0.1091 0.1409 0.1091];

% Initialize neural states
Output_action_potential = zeros(Rows, Column);  % Spike output
U = Output_action_potential;                    % Membrane potential
Ignition_map = Output_action_potential;         % Activation history
E = Output_action_potential + 1;                % Dynamic threshold

% Neural dynamics parameters
Iterations = 50;                                % Simulation length
Membrane_potential_attenuation_constant = 0.8;  % Membrane decay
Threshold_attenuation_constant = 0.704;         % Threshold decay
Absolute_refractory_period = 18.3;              % Post-spike inhibition

% Simulate neural dynamics
for t = 1:Iterations
    % Update membrane potential
    % V(n) = α*V(n-1) + S*(W*Y(n-1) + 1)
    % where α is decay, S is input, W is connectivity, Y is output
    U = Membrane_potential_attenuation_constant .* U + ...
        Pulse_signal_vector .* conv2(Output_action_potential, Weight_matrix, 'same') + ...
        Pulse_signal_vector;

    % Update adaptive threshold
    % E(n) = β*E(n-1) + γ*Y(n-1)
    % where β is decay, γ is refractory strength
    E = Threshold_attenuation_constant .* E + ...
        Absolute_refractory_period .* Output_action_potential;

    % Compute neural activation
    X = 1 ./ (1 + exp(E - U));                 % Sigmoid activation
    Output_action_potential = double(X > 0.5);  % Threshold crossing

    % Accumulate activation history
    Ignition_map = Ignition_map + Output_action_potential;
end

end