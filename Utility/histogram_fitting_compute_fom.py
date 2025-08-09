import numpy as np
import matplotlib
# Prefer a headless-safe backend by default, but try TkAgg first when available
try:
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt  # noqa: E402
except Exception:
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt  # noqa: E402
from sklearn.mixture import GaussianMixture
from typing import Tuple

# Torch is optional – handle absence gracefully
try:  # pragma: no cover - environment dependent
    import torch as _torch
except Exception:  # torch not installed or unavailable
    _torch = None

def histogram_fitting_compute_fom(
    pulse_shape_discrimination_factor: np.ndarray,
    method_name,
    show_plot: bool = True
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Performs histogram fitting on pulse shape discrimination factors and computes the Figure of Merit (FOM).

    Parameters:
        pulse_shape_discrimination_factor: numpy array or pytorch tensor of discrimination factors.
        show_plot: Whether to display the plot (True) or save it to 'fom_plot.jpg' (False).

    Returns:
        tuple: (mu, sigma, fom)
            - mu: Means of the fitted Gaussian components.
            - sigma: Standard deviations of the fitted Gaussian components.
            - fom: Figure of Merit calculated from the fitted Gaussian parameters.
    """
    # Convert torch tensor to numpy if needed (only if torch is available)
    if _torch is not None and isinstance(pulse_shape_discrimination_factor, _torch.Tensor):
        pulse_shape_discrimination_factor = pulse_shape_discrimination_factor.detach().cpu().numpy()

    # Check for NaN values
    nan_count = np.sum(np.isnan(pulse_shape_discrimination_factor))
    total_signals = len(pulse_shape_discrimination_factor)
    if nan_count > 0:
        nan_percentage = (nan_count / total_signals) * 100
        print(f"Warning: {nan_count} signals ({nan_percentage:.2f}% of total {total_signals} signals) have NaN discrimination factor values. Please check if there are error signals in the dataset or if the discrimination algorithm parameters match the dataset. NaN values will be removed.")
        # Remove NaN values
        pulse_shape_discrimination_factor = pulse_shape_discrimination_factor[~np.isnan(pulse_shape_discrimination_factor)]

    # Check for all same values
    if np.all(pulse_shape_discrimination_factor == pulse_shape_discrimination_factor[0]):
        print(f"Warning: All {total_signals} signals have the same discrimination factor value ({pulse_shape_discrimination_factor[0]}). This indicates that the discrimination algorithm may not be working properly or all signals have identical characteristics. Please check the discrimination algorithm parameters and dataset.")
        return np.array([0, 0]), np.array([0, 0]), 0.0
    
    def normalize_psd(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values)
        min_val = np.min(values)
        max_val = np.max(values)
        if min_val == max_val:
            return np.zeros_like(values)
        return (values - min_val) / (max_val - min_val)

    # Normalize the discrimination factors
    r = normalize_psd(pulse_shape_discrimination_factor)

    # Fit a double Gaussian model to the data
    num_components = 2
    gm_model = GaussianMixture(
        n_components=num_components,
        max_iter=1000,
        random_state=42  # for reproducibility
    )
    r_reshaped = r.reshape(-1, 1)
    gm_model.fit(r_reshaped)

    # Generate x values and compute the probability density function (PDF)
    x = np.linspace(np.min(r), np.max(r), 1000)
    y = np.exp(gm_model.score_samples(x.reshape(-1, 1)))

    # Plot histogram and fitted Gaussian
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Primary y-axis (Signal Count)
    counts, bins, _ = ax1.hist(r, bins='auto', alpha=0.5, color='b', edgecolor='black', label='Histogram')
    ax1.set_xlabel('Pulse Shape Discrimination Factor')
    ax1.set_ylabel('Signal Count', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_ylim(bottom=0)
    
    # Secondary y-axis (Probability Density)
    ax2 = ax1.twinx()
    ax2.plot(x, y, 'r-', linewidth=2, label='Gaussian Fit')
    ax2.set_ylabel('Probability Density', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_ylim(bottom=0)
    
    plt.title(f'Double Gaussian Fitting with Histogram of {method_name}')
    
    # Create combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # Retrieve Gaussian parameters
    mu = gm_model.means_.flatten()
    sigma = np.sqrt(gm_model.covariances_.flatten())
    fom = abs((mu[1] - mu[0]) / (2.355 * (sigma[1] + sigma[0])))

    # Annotate FOM on the plot
    ax2.text(0.75, 0.60, f'FOM = {fom:.4f}', transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8))

    # Decide whether to display or save, depending on backend/headless state
    backend = matplotlib.get_backend().lower()
    interactive_backend = ('tkagg' in backend) or ('qt' in backend) or ('macosx' in backend)
    if show_plot and interactive_backend:
        plt.show()
    else:
        plt.savefig(f'fom_plot_{method_name}.jpg', format='jpg', dpi=300)
        print(f"Figure of Merit plot saved as 'fom_plot_{method_name}.jpg'.")

    plt.close()

    return mu, sigma, fom

# Example usage (for testing purposes)
if __name__ == "__main__":
    np.random.seed(42)
    sample_data = np.concatenate([
        np.random.normal(0.3, 0.1, 1000),
        np.random.normal(0.7, 0.1, 1000)
    ])
    mu, sigma, fom = histogram_fitting_compute_fom(sample_data, "Test Method")
    print(f"Means: {mu}")
    print(f"Standard Deviations: {sigma}")
    print(f"FOM: {fom}")
