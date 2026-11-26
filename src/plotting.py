import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

def plot_bp_hist(df: pd.DataFrame) -> None:
    """Ritar histogram med täthetskurva över systoliskt bloodtryck"""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(df["systolic_bp"], kde=True, bins=20, color="#31828B", ax=ax)
    ax.axvline(df["systolic_bp"].mean(), color="red", linestyle="--", label="Medelvärde")
    ax.set_title("Fördelning av systoliskt bloodtryck", fontweight="bold")
    ax.set_xlabel("Systoliskt bloodtryck (mmHg)", fontweight="bold")
    ax.set_ylabel("Antal deltagare", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_weight_by_sex(df: pd.DataFrame) -> None:
    """Ritar boxplot över vikt uppdelat på kön."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.boxplot(data=df, x="sex", y="weight", hue="sex", palette="Set2", legend=False, ax=ax)
    ax.set_title("Viktfördelning per kön", fontweight="bold")
    ax.set_xlabel("Kön (F/M)", fontweight="bold")
    ax.set_ylabel("Vikt (Kg)", fontweight="bold")
    plt.tight_layout()
    plt.show()

def plot_smoker_share(df: pd.DataFrame) -> None:
    """Ritar stapeldiagram över andelen rökare i precent"""
    smoker_prop = df["smoker"].value_counts(normalize=True) * 100
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(x=smoker_prop.index, 
                y=smoker_prop.values, 
                hue=smoker_prop.index, 
                palette="pastel", 
                legend=False, 
                ax=ax
                )
    ax.set_title("Andel rökare i studien", fontweight="bold")
    ax.set_xlabel("Rökare", fontweight="bold")
    ax.set_ylabel("Andel (%)", fontweight="bold")
    ax.set_ylim(0, max(smoker_prop.values) + 12)

    for i, val in enumerate(smoker_prop.values):
        ax.text(i, val + 1.5, f"{val:.1f}%", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.show()
