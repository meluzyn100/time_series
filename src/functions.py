import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

def moving_average(data, window_size):
    """
    Calculate the moving average of a given data array.

    Parameters:
    - data: numpy.ndarray, the input data array
    - window_size: int, the size of the moving average window

    Returns:
    - numpy.ndarray, the moving average of the input data
    """
    n = len(data)
    if window_size > n:
        raise ValueError("Window size must be less than or equal to the length of the data.")
    if window_size % 2 == 0:
        raise ValueError("Window size must be an odd number.")

    result = np.zeros(n - window_size + 1)
    for i in range(n - window_size + 1):
        result[i] = np.mean(data[i:i + window_size])
    return result


def calculate_point_distances(x, y, slope, intercept):
    """
    Calculate the perpendicular distances of points from the regression line.

    Parameters:
    - x: numpy.ndarray, independent variable values
    - y: numpy.ndarray, dependent variable values
    - slope: float, slope of the regression line
    - intercept: float, intercept of the regression line

    Returns:
    - numpy.ndarray, distances of points from the regression line
    """
    a = -slope
    b = 1
    c = -intercept
    distances = np.abs(a * x + b * y + c) / np.sqrt(a**2 + b**2)
    return distances



def estimate_coefficients(x_values, y_values):
    """Estimate coefficients b0 and b1 using OLS."""
    b1_estimate = np.sum((x_values - np.mean(x_values)) * y_values) / np.sum((x_values - np.mean(x_values)) ** 2)
    b0_estimate = np.mean(y_values) - b1_estimate * np.mean(x_values)
    return b0_estimate, b1_estimate

def calculate_theoretical_distributions(b0_true, b1_true, sigma, x_values, n_samples):
    """Calculate theoretical distributions for b0 and b1."""
    b0_variance = sigma**2 / n_samples + np.mean(x_values)**2 * sigma**2 / np.sum((x_values - np.mean(x_values))**2)
    b1_variance = sigma**2 / np.sum((x_values - np.mean(x_values))**2)
    b0_theoretical = norm(loc=b0_true, scale=np.sqrt(b0_variance))
    b1_theoretical = norm(loc=b1_true, scale=np.sqrt(b1_variance))
    return b0_theoretical, b1_theoretical

def generate_simulation_data(x_values, b0, b1, sigma, n, mcs):
    """Generate simulation data for Monte Carlo simulations."""
    b0_estimates = np.zeros(mcs)
    b1_estimates = np.zeros(mcs)
    residual_variances = np.zeros(mcs)
    for i in range(mcs):
        noise = np.random.normal(0, sigma, n)
        y_values = b0 + b1 * x_values + noise
        b1_estimate = np.sum((x_values - np.mean(x_values)) * y_values) / np.sum((x_values - np.mean(x_values)) ** 2)
        b0_estimate = np.mean(y_values) - b1_estimate * np.mean(x_values)
        b0_estimates[i] = b0_estimate
        b1_estimates[i] = b1_estimate
        residual_variances[i] = np.sum((y_values - (b0_estimate + b1_estimate * x_values)) ** 2) / (n - 2)
    return b0_estimates, b1_estimates, residual_variances

def calculate_t_statistics(b_estimates, true_value, residual_variances, x_values, n, b_type):
    """Calculate t-statistics for B0 or B1."""
    if b_type == "b1":
        denominator = np.sqrt(residual_variances) / np.sqrt(np.sum((x_values - np.mean(x_values)) ** 2))
    elif b_type == "b0":
        denominator = np.sqrt(residual_variances) * np.sqrt(1 / n + np.mean(x_values) ** 2 / np.sum((x_values - np.mean(x_values)) ** 2))
    else:
        raise ValueError("Invalid b_type. Use 'b0' or 'b1'.")
    return (b_estimates - true_value) / denominator

def plot_distributions(t_statistics, theoretical_dist, title_prefix):
    """Plot empirical and theoretical distributions."""
    ecdf = ECDF(t_statistics)
    xs = np.linspace(-4, 4, 100)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot PDF
    axs[0].hist(t_statistics, density=True, bins=20, alpha=0.5, label="Empirical PDF")
    axs[0].plot(xs, theoretical_dist.pdf(xs), label="Theoretical PDF", color="red", alpha=0.7)
    axs[0].set_title(f"{title_prefix} PDF")
    axs[0].legend()

    # Plot CDF
    axs[1].plot(ecdf.x, ecdf.y, label="Empirical CDF")
    axs[1].plot(xs, theoretical_dist.cdf(xs), label="Theoretical CDF", color="red", alpha=0.7)
    axs[1].set_title(f"{title_prefix} CDF")
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def generate_dependent_variable(x_values, b0, b1, sigma):
    """
    Generate the dependent variable with added Gaussian noise.

    Args:
        x_values (np.ndarray): Independent variable values.
        b0 (float): True intercept.
        b1 (float): True slope.
        sigma (float): Standard deviation of noise.
        num_samples (int): Number of samples.

    Returns:
        np.ndarray: Dependent variable values.
    """
    return b1 * x_values + b0 + np.random.normal(0, scale=sigma, size=len(x_values))



def estimate_coefficients(x_values, y_values):
    """
    Estimate coefficients using Ordinary Least Squares (OLS).

    Args:
        x_values (np.ndarray): Independent variable values.
        y_values (np.ndarray): Dependent variable values.

    Returns:
        tuple: Estimated intercept (b0) and slope (b1).
    """
    b1 = np.sum((x_values - np.mean(x_values)) * y_values) / np.sum((x_values - np.mean(x_values)) ** 2)
    b0 = np.mean(y_values) - b1 * np.mean(x_values)
    return b0, b1

def calculate_residual_variance(x_values, y_values, b0, b1, num_samples):
    """
    Calculate the residual variance.

    Args:
        x_values (np.ndarray): Independent variable values.
        y_values (np.ndarray): Dependent variable values.
        b0 (float): Estimated intercept.
        b1 (float): Estimated slope.
        num_samples (int): Number of samples.

    Returns:
        float: Residual variance.
    """
    residuals = y_values - (b0 + b1 * x_values)
    return np.sum(residuals ** 2) / (num_samples - 2)

def calculate_confidence_intervals(estimate, true_value, std_error, critical_value):
    """
    Calculate confidence intervals and check if the true value is within the interval.

    Args:
        estimate (np.ndarray): Estimated parameter values.
        true_value (float): True parameter value.
        std_error (float or np.ndarray): Standard error of the estimate.
        critical_value (float): Critical value for the confidence interval.

    Returns:
        np.ndarray: Boolean array indicating whether the true value is within the interval.
    """
    lower_bound = estimate - critical_value * std_error
    upper_bound = estimate + critical_value * std_error
    return (lower_bound <= true_value) & (upper_bound >= true_value)

def calculate_b1_std_error_known(sigma, x_values):
    """
    Calculate the standard error for B1 with known variance.

    Args:
        sigma (float): Standard deviation of noise.
        x_values (np.ndarray): Independent variable values.

    Returns:
        float: Standard error for B1.
    """
    return (sigma**2 / np.sum((x_values - np.mean(x_values)) ** 2)) ** 0.5

def calculate_b1_std_error_estimated(residual_variances, x_values):
    """
    Calculate the standard error for B1 with estimated variance.

    Args:
        residual_variances (np.ndarray): Residual variances from simulations.
        x_values (np.ndarray): Independent variable values.

    Returns:
        np.ndarray: Standard error for B1.
    """
    return (residual_variances / np.sum((x_values - np.mean(x_values)) ** 2)) ** 0.5

def calculate_b0_std_error_known(sigma, x_values, num_samples):
    """
    Calculate the standard error for B0 with known variance.

    Args:
        sigma (float): Standard deviation of noise.
        x_values (np.ndarray): Independent variable values.
        num_samples (int): Number of samples.

    Returns:
        float: Standard error for B0.
    """
    return (sigma**2 * (1 / num_samples + np.mean(x_values)**2 / np.sum((x_values - np.mean(x_values)) ** 2))) ** 0.5

def calculate_b0_std_error_estimated(residual_variances, x_values, num_samples):
    """
    Calculate the standard error for B0 with estimated variance.

    Args:
        residual_variances (np.ndarray): Residual variances from simulations.
        x_values (np.ndarray): Independent variable values.
        num_samples (int): Number of samples.

    Returns:
        np.ndarray: Standard error for B0.
    """
    return (residual_variances * (1 / num_samples + np.mean(x_values)**2 / np.sum((x_values - np.mean(x_values)) ** 2))) ** 0.5


def calculate_prediction_intervals(x_full, x_train, b0, b1, residual_variance, z_critical):
    """Calculate prediction intervals."""
    y_pred = b0 + b1 * x_full

    n_train = len(x_train)
    x_muean = np.mean(x_train)
    x_minus_mean_squared = (x_train - x_muean)**2
    se = np.sqrt(residual_variance * (1 + 1/n_train + x_minus_mean_squared / np.sum(x_minus_mean_squared)))
    y_pred_lower = y_pred - z_critical * se
    y_pred_upper = y_pred + z_critical * se
    return y_pred, y_pred_lower, y_pred_upper