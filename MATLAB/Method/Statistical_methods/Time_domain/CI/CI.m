function [Pulse_shape_discrimination_factor] = CI(Pulse_signal, Gate_rang, dt)

%--------------------------------------------------------------------------
% Charge Integration (CI) method for pulse shape discrimination.
%
% Discriminates particle types by computing the ratio between delayed and
% total charge integrations. Uses trapezoidal numerical integration to
% account for temporal characteristics of the pulse decay.
%
% Input:
%   Pulse_signal: Matrix of pulse signals to analyze [n_signals × signal_length]
%   Gate_rang: Integration gate parameters [pre_peak, gate_width, total_width]
%             Default: [14, 35, 65] samples relative to peak
%   dt: Time interval between samples (default: 1 ms)
%
% Output:
%   Pulse_shape_discrimination_factor: Ratio of delayed to total charge
%                                    [n_signals × 1]
%
% Integration gates:
%   - Total gate: [peak-pre_peak : peak+total_width]
%   - Delay gate: [peak+(total_width-gate_width) : peak+total_width]
%
% Reference:
% Pawełczak, I. A., et al. "Studies of neutron–γ pulse shape discrimination
% in EJ-309 liquid scintillator using charge integration method." Nuclear
% Instruments and Methods in Physics Research Section A: Accelerators,
% Spectrometers, Detectors and Associated Equipment 711 (2013): 21-26.
%--------------------------------------------------------------------------

% Set default parameters
if nargin < 2
    Gate_rang = [14, 35, 65];  % [pre_peak, gate_width, total_width]
end

if nargin < 3
    dt = 1;  % Sampling interval (ms)
end

% Initialize output array
Num_signal = size(Pulse_signal, 1);
Pulse_shape_discrimination_factor = zeros(Num_signal, 1);

% Process signals in parallel
parfor i = 1:Num_signal
    Pulse_signal_vector = Pulse_signal(i, :);
    
    % Locate pulse peak
    [~, Maxposition] = max(Pulse_signal_vector);

    % Calculate gate boundaries relative to peak
    Delay_gate_start = Maxposition + Gate_rang(3) - Gate_rang(2);  % Start of delayed region
    Delay_gate_end = Maxposition + Gate_rang(3);                   % End of both gates
    Total_gate_start = Maxposition - Gate_rang(1);                 % Start of total gate
    Total_gate_end = Maxposition + Gate_rang(3);                   % Same as delay end

    % Validate gate boundaries
    if Delay_gate_start < 1 || Delay_gate_end > length(Pulse_signal_vector) || ...
       Total_gate_start < 1 || Total_gate_end > length(Pulse_signal_vector)
        Pulse_shape_discrimination_factor(i) = NaN;  % Invalid gate range
        continue;
    end

    % Define integration ranges
    Delay_gate = Delay_gate_start:Delay_gate_end;
    Total_gate = Total_gate_start:Total_gate_end;

    % Extract signal segments
    Delay_component = Pulse_signal_vector(Delay_gate);
    Total_component = Pulse_signal_vector(Total_gate);

    % Compute charge integrals using trapezoidal method
    Delay_integration = trapz(Delay_gate, Delay_component) * dt;
    Total_integration = trapz(Total_gate, Total_component) * dt;

    % Calculate discrimination factor as charge ratio
    Pulse_shape_discrimination_factor(i) = Delay_integration / Total_integration;
end

end
