function [Ignition_map] = PCNN(Pulse_signal_vector)

%--------------------------------------------------------------------------
% Pulse-Coupled Neural Network implementation for signal analysis.
%
% Implements a biologically-inspired neural network with feedback and lateral
% connections. Features dynamic thresholds, linking fields, and feedback
% mechanisms inspired by the visual cortex.
%
% Input:
%   Pulse_signal_vector: Signal to process
%
% Output:
%   Ignition_map: Accumulated neural activation history
%--------------------------------------------------------------------------

% Get signal dimensions
[Rows, Column] = size(Pulse_signal_vector);

% Define network parameters
% Lateral connection weights
l = 0.1091;  % Local weight
r = 0.1409;  % Radial weight
Matrix = [l r l;  % 3x3 connection kernel
         r 0 r;
         l r l];

% Neural dynamics parameters
al = 0.356;   % Linking decay
vl = 0.0005;  % Linking coupling
ve = 15.5;    % Threshold magnitude
ae = 0.081;   % Threshold decay
af = 0.325;   % Feedback decay
vf = 0.0005;  % Feedback coupling
beta = 0.67;  % Linking strength
Iterations = 180;

% Initialize neural states
Time_pluse_sequence = zeros(Rows, Column);  % Neural output
Ignition_map = Time_pluse_sequence;         % Activation history
Feedback_input = Time_pluse_sequence;       % Feedback pathway
Link_input = Time_pluse_sequence;           % Linking field
Dynamic_threshold = Time_pluse_sequence;     % Adaptive threshold

% Simulate neural dynamics
for T = 1:Iterations
    % Update modulation pathways
    % F(n) = exp(-af)*F(n-1) + vf*W*Y(n-1) + S
    Feedback_input = exp(-af) * Feedback_input + vf * conv2(Time_pluse_sequence, Matrix, 'same') + Pulse_signal_vector;
    
    % L(n) = exp(-al)*L(n-1) + vl*W*Y(n-1)
    Link_input = exp(-al) * Link_input + vl * conv2(Time_pluse_sequence, Matrix, 'same');
    
    % Compute total neural input
    % U(n) = F(n) * (1 + β*L(n))
    Total_input = Feedback_input .* (1 + beta * Link_input);
    
    % Generate neural activation
    Pulse_generator = 1 ./ (1 + exp(Dynamic_threshold - Total_input));
    Time_pluse_sequence = double(Pulse_generator > 0.5);
    
    % Update adaptive threshold
    % θ(n) = exp(-ae)*θ(n-1) + ve*Y(n)
    Dynamic_threshold = exp(-ae) * Dynamic_threshold + ve * Time_pluse_sequence;
    
    % Accumulate activation history
    Ignition_map = Ignition_map + Time_pluse_sequence;
end

end
