import os
from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

beta = 0.05  # Nominal reproducibility error 5%
z_1_beta_2 = 1.96  # Approximate value for 95% confidence


def calculate_variance(mse_values: list[float]) -> float:
    """Compute the variance of the MSE aggregated estimate"""
    g = len(mse_values)
    if g < 2:  # Avoid division by zero
        return np.nan
    mean_mse = np.mean(mse_values)
    variance = np.sum((mse_values - mean_mse) ** 2) / (g * (g - 1))
    return variance


def get_thresholds(avg_mse: float, z_1_beta_2: float) -> Tuple[list[float], float, float]:
    xi = 0.001 * avg_mse  # 0.1% of avg MSE as the error threshold
    variance_threshold = (xi ** 2) / (
                2 * (z_1_beta_2 ** 2))  # Compute variance threshold (v-hat) based on 5% error prob
    all_variances = [calculate_variance(df["MSE"][:i]) for i in range(2, len(df) + 1)]  # variance for each g iteration
    variance_percentages = [(var / avg_mse) * 100 for var in all_variances if var is not None]  # normalize to avg MSE %
    threshold_percentage = (variance_threshold / avg_mse) * 100  # Convert threshold to percentage
    dataset_max_variance = max(variance_percentages)
    y_max = max(dataset_max_variance, threshold_percentage) * 1.1  # Allow 10% padding
    return variance_percentages, threshold_percentage, y_max


def plot_variance(df: pd.DataFrame, cumulative_variances: list, label: str, threshold_percentage: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(df["g_iteration"][:len(cumulative_variances)], cumulative_variances, label=label, color='b')
    # graph stopping point if found
    if cumulative_variances and min(cumulative_variances) <= threshold_percentage:
        stop_iteration = next(i for i, v in enumerate(cumulative_variances) if (v <= threshold_percentage) & (i > 10))
        plt.axvline(x=df["g_iteration"][stop_iteration], color='r', linestyle='--', alpha=0.7)
        plt.text(df["g_iteration"][stop_iteration], min(cumulative_variances), "Stop", color="red")

    plt.axhline(y=threshold_percentage, color='r', linestyle='dashed', label="Variance Threshold (%)")
    plt.ylim(0, y_max)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.2f}%"))
    plt.xlabel("g_iteration")
    plt.ylabel("Variance (% of Average MSE)")
    plt.title(f"Variance of Aggregated MSE vs g_iteration ({label})")
    plt.legend()
    plt.grid(True)
    graph_label = label.replace("+", "_").replace(" ", "").lower()
    plt.savefig(f'graphs/{graph_label}.png')
    plt.show()


files_label_dict = {
    "cb_cv_shuffle__random__g_100.csv": "CatBoost + Cross Validation + Shuffle",
    "cb_random__g_100.csv": "CatBoost",
    "cb_shuffle__random__g_100.csv": "CatBoost + Shuffle",
    "cv_cb__random__g_100.csv": "CatBoost + Cross Validation",
    "cv_random__g_100.csv": "Cross Validation",
    "cv_shuffle__random__g_100.csv": "Cross Validation + Shuffle",
    "shuffle_random__g_100.csv": "Shuffle",
}

if __name__ == "__main__":
    os.makedirs("graphs", exist_ok=True)
    for file, label in files_label_dict.items():
        df = pd.read_csv(f"results/{file}", header=None, names=["g_iteration", "MSE"])
        avg_mse = df["MSE"].mean()  # Compute average MSE for a model
        cumulative_variances, threshold_percentage, y_max = get_thresholds(avg_mse, z_1_beta_2)
        plot_variance(df, cumulative_variances, label, threshold_percentage)
