function [miu, sigma, FOM] = Histogram_fitting_compute_FOM(Pulse_shape_discrimination_factor, Method_name)

%--------------------------------------------------------------------------
% Histogram fitting and Figure of Merit (FOM) calculation for PSD evaluation.
%
% Evaluates discrimination performance by fitting a double Gaussian model
% to the distribution of discrimination factors and computing the Figure
% of Merit, which quantifies the separation between particle types.
%
% Input:
%   Pulse_shape_discrimination_factor: Discrimination values [n_signals × 1]
%   Method_name: Name of PSD method for plot title
%
% Output:
%   miu: Mean values of fitted Gaussians [1 × 2]
%        miu(1): First peak (typically gamma)
%        miu(2): Second peak (typically neutron)
%   sigma: Standard deviations of fitted Gaussians [1 × 2]
%   FOM: Figure of Merit = |μ₂ - μ₁|/(2.355(σ₁ + σ₂))
%
% Algorithm:
%   1. Normalize discrimination factors to [0,1]
%   2. Fit double Gaussian mixture model
%   3. Compute FOM from fitted parameters
%   4. Visualize results with histogram and fit
%
% Note:
%   - FOM > 1.5 typically indicates good discrimination
%   - 2.355 factor converts σ to FWHM
%   - Higher FOM indicates better particle separation
%--------------------------------------------------------------------------

% Normalize discrimination factors to [0,1] range
min_val = min(Pulse_shape_discrimination_factor);
max_val = max(Pulse_shape_discrimination_factor);

if min_val == max_val
    R = zeros(size(Pulse_shape_discrimination_factor));  % Handle constant input
else
    R = (Pulse_shape_discrimination_factor - min_val) / (max_val - min_val);
end

% Fit Gaussian mixture model
num_components = 2;                                      % Two particle types
options = statset('MaxIter', 1000);                     % Maximum iterations
gmModel = fitgmdist(R, num_components, 'Options', options);

% Generate points for plotting fitted distribution
x = linspace(min(R), max(R), 1000);                     % Evaluation points
y = pdf(gmModel, x');                                   % Probability density

% Create visualization
figure;
hold on;

% Plot normalized histogram with transparency
histogram(R, 'Normalization', 'pdf', ...                % Probability density
          'FaceColor', 'b', ...                         % Blue bars
          'FaceAlpha', 0.5, ...                         % 50% transparent
          'EdgeColor', 'black');                        % Black edges

% Plot fitted distribution
plot(x, y, 'LineWidth', 2, 'Color', 'r');              % Red fit line

% Add labels and title
title(['Double Gaussian Fitting with Histogram of ', Method_name]);
xlabel('Pulse Shape Discrimination Factor');
ylabel('Probability Density');

% Extract distribution parameters
miu = gmModel.mu;                                       % Mean values
sigma = sqrt(gmModel.Sigma);                            % Standard deviations

% Calculate Figure of Merit
FOM = abs((miu(2) - miu(1)) / (2.355 * (sigma(2) + sigma(1))));

% Display FOM on plot
str_FOM = sprintf('FOM = %.4f', FOM);                   % Format with 4 decimals
dim = [.75 .60 .3 .3];                                  % Position on plot
annotation('textbox', dim, 'String', str_FOM, ...
          'FontSize', 12, 'FitBoxToText', 'on');

end
