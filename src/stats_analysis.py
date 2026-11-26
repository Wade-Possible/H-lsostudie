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

def test_smoker_blood_pressure(df: pd.DataFrame, alpha: float = 0.05) -> dict[str, float | bool]:
    """Utför ensidigt Welchs t-test för att se om rökare har högre systoliskt bloodtryck."""
    smokers = df.loc[df["smoker"] == "Yes", "systolic_bp"].dropna()
    nonsmokers = df.loc[df["smoker"] == "No", "systolic_bp"].dropna()

    t_stat, p_val = stats.ttest_ind(smokers, nonsmokers, equal_var=False, alternative="greater")

    return {
        "smoker_mean": float(smokers.mean()),
        "nonsmoker_mean": float(nonsmokers.mean()),
        "diff": float(smokers.mean() - nonsmokers.mean()),
        "t_stat": float(t_stat),
        "p_val": float(p_val),
        "reject_h0": bool(p_val < alpha)

    }

def simulate_power(df: pd.DataFrame, true_effect: float, alpha: float = 0.05, n_sims: int = 2000, seed: int = 42) -> float:
    """Simulerar statistikens styrka för en ensidig Welchs t-test."""
    rng = np.random.default_rng(seed)
    smokers = df.loc[df["smoker"] == "Yes", "systolic_bp"].dropna()
    nonsmokers = df.loc[df["smoker"] == "No", "systolic_bp"].dropna()

    n1, n2 = len(smokers), len(nonsmokers)
    sd1, sd2 = smokers.std(ddof=1), nonsmokers.std(ddof=1)
    base_mean = nonsmokers.mean()

    rejections = 0
    for _ in range(n_sims):
        sim_smokers = rng.normal(loc=base_mean + true_effect, scale=sd1, size=n1)
        sim_nonsmokers = rng.normal(loc=base_mean, scale=sd2, size=n2)
        _, p = stats.ttest_ind(sim_smokers, sim_nonsmokers, equal_var=False, alternative="greater")
        if p < alpha:
            rejections += 1
    return rejections / n_sims