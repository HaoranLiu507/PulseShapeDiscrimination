function [Pulse_shape_discrimination_factor] = PCA(Pulse_signal, NumRows, Signal_cut)

%--------------------------------------------------------------------------
% Principal Component Analysis (PCA) method for pulse shape discrimination.
%
% Discriminates particle types by projecting pulse signals onto their
% principal components. Uses dimensionality reduction to identify the most
% discriminating features in the pulse shapes.
%
% Input:
%   Pulse_signal: Matrix of pulse signals to analyze [n_signals × signal_length]
%   NumRows: Number of signals to use for PCA training (default: 3000)
%   Signal_cut: Start index for region of interest (default: 79)
%
% Output:
%   Pulse_shape_discrimination_factor: Projection magnitudes onto first PC
%                                    [n_signals × 1]
%
% Algorithm:
%   1. Select training subset and ROI from signals
%   2. Compute covariance matrix of training data
%   3. Extract principal components (eigenvectors)
%   4. Project all signals onto first PC
%
% Mathematical basis:
%   - Covariance matrix: C = X'X / (n-1)
%   - Eigendecomposition: CV = VΛ
%   - Discrimination factor: |Xv₁|
%   where X is the signal matrix, v₁ is the first eigenvector
%
% Note:
%   - Training subset should be representative of all signal types
%   - ROI selection affects discrimination performance
%   - NaN values in input data will cause errors
%
% Reference:
% Alharbi, Thamer. "Principal Component Analysis for pulse-shape
% discrimination of scintillation radiation detectors." Nuclear Instruments
% and Methods in Physics Research Section A: Accelerators, Spectrometers,
% Detectors and Associated Equipment 806 (2016): 240-243.
%--------------------------------------------------------------------------

% Set default parameters
if nargin < 2
    NumRows = 3000;  % Number of signals for PCA training
end

if nargin < 3
    Signal_cut = 79;  % Start of region of interest
end

% Validate input dimensions
if NumRows > size(Pulse_signal, 1)
    error('NumRows (%d) exceeds available signals (%d).', NumRows, size(Pulse_signal, 1));
end

if Signal_cut > size(Pulse_signal, 2)
    error('Signal_cut (%d) exceeds signal length (%d).', Signal_cut, size(Pulse_signal, 2));
end

% Extract training data from ROI
X = Pulse_signal(1:NumRows, Signal_cut:end);  % [NumRows × ROI_length]

% Validate data quality
if any(isnan(X(:)))
    error('Training data contains NaN values. Clean data required for PCA.');
end

% Perform PCA
Covariance = cov(X);                          % Compute covariance matrix
[Eigenvector, Eigenvalue] = eig(Covariance);  % Eigendecomposition

% Extract principal component
[~, Max_index] = max(diag(Eigenvalue));       % Find largest eigenvalue
Max_eigenvector = Eigenvector(:, Max_index);  % First principal component

% Project all signals onto first principal component
Y = Pulse_signal(:, Signal_cut:end);          % Full dataset ROI
Pulse_shape_discrimination_factor = abs(Y * Max_eigenvector);  % Projection magnitudes

end
