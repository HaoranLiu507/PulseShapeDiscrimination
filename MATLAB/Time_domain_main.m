clc;
clear;

CurrentFolder = pwd;  % Consistent naming convention
addpath(genpath(CurrentFolder));

% Read data from txt file
Dataset = importdata('EJ299_33_AmBe_9414.txt');

if isempty(Dataset)
    error('Data import failed. Please check the file path and content.');
end

Pulse_signal_original = Dataset;

% Normalize the pulse signals if needed
Processed_data = Normalization(Pulse_signal_original);

% Available time domain methods
Available_methods = {
    'PR - Pattern Recognition';
    'PGA - Pulse Gradient Analysis';
    'PCA - Principal Component Analysis';
    'FEPS - Falling-Edge Percentage Slope';
    'ZC - Zero Crossing';
    'CC - Charge Comparison';
    'CI - Charge Integration';
    'GP - Gatti Parameter'; 
    'LLR - Log-Likelihood Ratio';
    'LMT - Log Mean Time'
};

% Initialize a flag for valid input
Valid_input = false;

while ~Valid_input
    % Prompt the user to select a method name
    Method_name = input(sprintf(['Select a PSD method from the following options:\n%s\n\n' ...
                                  'Example: Type "PR" for Pattern Recognition and press "Enter".\n\n'], ...
                                  strjoin(Available_methods, '\n')), 's');
    
    % Convert the input to uppercase
    Method_name = upper(Method_name);
    
    % Function mapping
    Methods = containers.Map(...
        {'PR', 'PGA', 'PCA', 'FEPS', 'ZC', 'CC', 'CI', 'GP', 'LLR', 'LMT'}, ...
        {@PR, @PGA, @PCA, @FEPS, @ZC, @CC, @CI, @GP, @LLR, @LMT});
    
    % Check if method name is valid
    if isKey(Methods, Method_name)
        % Retrieve the function handle from the map
        Func_handle = Methods(Method_name);
        
        % Compute Pulse_shape_discrimination_factor
        Pulse_shape_discrimination_factor = Func_handle(Processed_data);
        
        % Mark the input as valid to exit the loop
        Valid_input = true;
    else
        % Raise an error with a user-friendly message
        fprintf('Invalid method name: %s. Please enter a valid option.\n', Method_name);
        fprintf('Available methods: %s\n\n', strjoin(keys(Methods), ', '));
    end
end

% Compute FOM
FOM = Histogram_fitting_compute_FOM(Pulse_shape_discrimination_factor, Method_name);