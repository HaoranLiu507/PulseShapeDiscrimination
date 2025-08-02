clc;
clear;

CurrentFolder = pwd; % Consistent naming convention
addpath(genpath(CurrentFolder));

% Read data from txt file
Dataset = importdata('EJ299_33_AmBe_9414.txt');

if isempty(Dataset)
    error('Data import failed. Please check the file path and content.');
end

Pulse_signal_original = Dataset;

% Normalize the pulse signals if needed
Processed_data = Normalization(Pulse_signal_original);

% Available frequency domain methods
Available_methods = {
    'FGA - Frequency Gradient Analysis';
    'SDCC - Simplified Digital Charge Collection';
    'DFT - Discrete Fourier Transform';
    'WT1 - Wavelet Transform';
    'FS - Fractal Spectrum';
    'SD - Scalogram-based Discrimination';
    'WT2 - Marr Wavelet Transform';
};

% Initialize a flag for valid input
Valid_input = false;

while ~Valid_input
    % Prompt the user to select a method name
    Method_name = input(sprintf(['Select a frequency domain method from the following options:\n%s\n\n' ...
                                  'Example: Type "FGA" for Frequency Gradient Analysis and press "Enter".\n\n'], ...
                                  strjoin(Available_methods, '\n')), 's');
    
    % Convert the input to uppercase
    Method_name = upper(Method_name);

    % Function mapping
    Methods = containers.Map(...
        {'FGA', 'SDCC', 'DFT', 'WT1', 'FS','SD', 'WT2'}, ...
        {@FGA, @SDCC, @DFT, @WT1, @FS,@SD, @WT2});
    
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