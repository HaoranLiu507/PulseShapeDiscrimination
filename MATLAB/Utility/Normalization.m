function [Processed_data] = Normalization(Pulse_signal)

%--------------------------------------------------------------------------
% Signal normalization for pulse shape discrimination preprocessing.
%
% Normalizes pulse signals to the range [0,1] if necessary, using min-max
% scaling. Includes random sampling verification to check if normalization
% is needed, improving computational efficiency.
%
% Input:
%   Pulse_signal: Matrix of pulse signals [n_signals × signal_length]
%
% Output:
%   Processed_data: Normalized signals [n_signals × signal_length]
%                  Range: [0,1] if normalized, original if already in range
%
% Algorithm:
%   1. Sample random points to check signal range
%   2. If samples outside [0,1]:
%      - Normalize each signal: x' = (x - min(x))/(max(x) - min(x))
%   3. Save normalized data if modified
%
% Note:
%   - Preserves signal shape while standardizing amplitude
%   - Handles constant signals (max = min) without modification
%   - Saves normalized data to 'Normalized_DATA.txt' if changed
%--------------------------------------------------------------------------

% Store input data
Data = Pulse_signal;

% Random sampling for range check
num_samples = 10;
Random_indices = randperm(length(Data), num_samples);  % Random positions
Random_samples = Data(Random_indices);                 % Sample values

% Check if normalization needed
if all(Random_samples >= 0) && all(Random_samples <= 1)
    Processed_data = Data;  % Already in [0,1] range
else
    % Initialize normalized array
    Normalized_data = zeros(size(Data));
    
    % Process each signal
    for i = 1:size(Data, 1)
        % Get signal range
        Min_value = min(Data(i, :));
        Max_value = max(Data(i, :));
        
        % Apply min-max normalization
        if Max_value - Min_value == 0
            Normalized_data(i, :) = Data(i, :);  % Constant signal
        else
            Normalized_data(i, :) = (Data(i, :) - Min_value) / ...
                                  (Max_value - Min_value);
        end
    end
    
    % Store normalized result
    Processed_data = Normalized_data;
    
    % Save to file for reference
    save('Normalized_DATA.txt', 'Normalized_data', '-ascii');
end

end
