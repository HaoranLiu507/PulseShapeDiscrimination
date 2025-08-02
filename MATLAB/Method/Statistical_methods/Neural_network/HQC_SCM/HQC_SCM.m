function Ignition_map_combined = HQC_SCM(Pulse_signal_vector, Segment_point)

%--------------------------------------------------------------------------
% This function computes the combined Ignition Map for a given pulse signal 
% vector by dividing the signal into two parts and applying separate 
% processing methods (QCSCM_A and QCSCM_B) to each part.
%
% Param Segment_point: The index where the signal is divided into two parts.
%
% Returns:
%   Ignition_map_combined: Combined ignition map generated from both parts 
%                          of the signal after processing.
%--------------------------------------------------------------------------

% Divide the signal into two parts based on the segment point
Pulse_signal_vector_A = Pulse_signal_vector(1:Segment_point);
Pulse_signal_vector_B = Pulse_signal_vector(Segment_point+1:end);

% Apply QCSCM_A and QCSCM_B functions to the two parts of the signal
Ignition_map_A = QCSCM_A(Pulse_signal_vector_A);
Ignition_map_B = QCSCM_B(Pulse_signal_vector_B);

% Combine the ignition maps from the two parts
Ignition_map_combined = [Ignition_map_A, Ignition_map_B]; 

end
