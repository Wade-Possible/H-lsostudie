import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

def plot_bp_hist(df: pd.DataFrame) -> None:
    """Ritar histogram med täthetskurva över systoliskt bloodtryck"""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(df["systolic_bp"], kde=True, bins=20, color="#31828B", ax=ax)
    ax.axvline(df["systolic_bp"].mean(), color="red", linestyle="--", label="Medelvärde")
    ax.set_title("Fördelning av systoliskt bloodtryck")
    ax.set_xlabel("Systoliskt bloodtryck (mmHg)")
    ax.set_ylabel("Antal deltagare")
    ax.legend()
    plt.tight_layout()
    plt.show()