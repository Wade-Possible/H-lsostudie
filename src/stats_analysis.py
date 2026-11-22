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



