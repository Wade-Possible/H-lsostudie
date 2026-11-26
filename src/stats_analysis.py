import numpy as np
import pandas as pd
from scipy import stats 

def decript_stat(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Ska beräknar medelvärde, median, min och max för valda kolumner  
    """
    summary = df[columns].agg(["mean", "median", "min", "max"]).T
    summary.columns = ["Medelvärde", "Median", "Min", "Max"]
    return summary.round(2)

def simulate_disease_prevalance(df: pd.DataFrame, n_sim: int = 1000, seed: int = 42) -> dict[str, float]:
    """
    Ska simulerar n_sim individer utifrån datasetets verkliga sjukdomsandel.
    """
    rng = np.random.default_rng(seed)
    true_rate = float(df["disease"].mean())
    simulated = rng.binomial(n=1, p=true_rate, size=n_sim)
    sim_rate = float(simulated.mean())
    return {
        "true_rate": true_rate,
        "sim_rate": sim_rate,
        "diff": sim_rate - true_rate
    }

def calc_normal_ci(data: pd.Series | np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """
    Ska beräkna konfidensintervall för medelvärdet med normalapproximation.
    """
    arr = np.asarray(data.dropna() if isinstance(data, pd.Series) else data)
    n = len(arr)
    mean = np.mean(arr)
    se = np.std(arr, ddof=1) / np.sqrt(n)
    z_crit = stats.norm.ppf(1 - (1 - confidence) / 2)
    return float(mean - z_crit * se), float(mean + z_crit * se)

def ci_bootstrap(data: pd.Series | np.ndarray, n_boot: int = 10_000, confidence: float = 0.95, seed: int = 42) -> tuple[float, float]:
    """Beräknar konfidensintervall för medelvärdet med non-parametrisk bootstrap"""
    rng = np.random.default_rng(seed)
    arr = np.asarray(data.dropna() if isinstance(data, pd.Series) else data)
    n = len(arr)

    boot_means = np.array([rng.choice(arr, size=n, replace=True).mean() for _ in range(n_boot)])
    alpha = (1 - confidence) * 100
    low, high = np.percentile(boot_means, [alpha / 2, 100 - alpha / 2])
    return float(low), float(high)
