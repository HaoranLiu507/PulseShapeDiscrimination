function [Pulse_shape_discrimination_factor] = CC(Pulse_signal, Gate_rang)

%--------------------------------------------------------------------------
% Charge Comparison (CC) method for pulse shape discrimination.
%
% Discriminates particle types by comparing the slow component charge to the
% total charge of the pulse signal. The method exploits differences in decay
% characteristics between particle interactions.
%
% Input:
%   Pulse_signal: Matrix of pulse signals to analyze [n_signals × signal_length]
%   Gate_rang: Integration gate parameters [pre_peak, post_peak_start, post_peak_end]
%             Default: [10, 27, 90] samples relative to peak
%
% Output:
%   Pulse_shape_discrimination_factor: Ratio of slow component to total charge
%                                    [n_signals × 1]
%
% Integration gates:
%   - Long gate: [peak-pre_peak : peak+post_peak_end]
%   - Short gate: [peak+post_peak_start : peak+post_peak_end]
%
% Reference:
% Moszynski, Marek, et al. "Study of n-γ discrimination by digital charge 
% comparison method for a large volume liquid scintillator." Nuclear 
% Instruments and Methods in Physics Research Section A: Accelerators, 
% Spectrometers, Detectors and Associated Equipment 317.1-2 (1992): 262-272.
%--------------------------------------------------------------------------

% Set default integration gates
if nargin < 2
    Gate_rang = [10, 27, 90];  % [pre_peak, post_peak_start, post_peak_end]
end

% Initialize output array
Num_signal = size(Pulse_signal, 1);
Pulse_shape_discrimination_factor = zeros(Num_signal, 1);

% Process signals in parallel
parfor i = 1:Num_signal
    Pulse_signal_vector = Pulse_signal(i, :);
    
    % Locate pulse peak
    [~, Maxposition] = max(Pulse_signal_vector);
    
    % Define integration gates relative to peak
    Short_gate = Maxposition + Gate_rang(2) : Maxposition + Gate_rang(3);    % Slow component
    Long_gate = (Maxposition - Gate_rang(1)) : (Maxposition + Gate_rang(3)); % Total signal
    
    % Verify gate boundaries are within signal
    if (Maxposition + Gate_rang(2) <= length(Pulse_signal_vector)) && ...
       (Maxposition + Gate_rang(3) <= length(Pulse_signal_vector)) && ...
       (Maxposition - Gate_rang(1) > 0)
       
        % Extract components and compute charge integrals
        Slow_component = Pulse_signal_vector(Short_gate);
        Total_component = Pulse_signal_vector(Long_gate);
        
        % Calculate discrimination factor as charge ratio
        Pulse_shape_discrimination_factor(i) = sum(Slow_component) / sum(Total_component);
    else
        % Mark invalid gates with NaN
        Pulse_shape_discrimination_factor(i) = NaN;
    end
end

end
