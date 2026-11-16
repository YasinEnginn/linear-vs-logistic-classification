import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.ticker import PercentFormatter
from textwrap import fill

sns.set_theme(
    style="whitegrid",
    context="paper",
    font_scale=1.2,
)

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.titleweight": "bold",
    }
)

dataset_path = "data/raw/raisin/Raisin_Dataset.xlsx"
output_dir = Path("experiments/01_dataset_candidate_analysis/outputs")
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(dataset_path)


fig, ax = plt.subplots(figsize=(6, 4))

sns.countplot(
    data=df,
    x="Class",
    hue="Class",
    palette="colorblind",
    legend=False,
    ax=ax,
)

ax.set_title("Class Distribution of the Raisin Dataset")
ax.set_xlabel("Raisin Class")
ax.set_ylabel("Number of Samples")

for container in ax.containers:
    ax.bar_label(container, padding=3)  # type: ignore

sns.despine()
fig.tight_layout()
fig.savefig(output_dir / "raisin_class_distribution.pdf")
fig.savefig(
    output_dir / "raisin_class_distribution.png",
    dpi=300,
)
plt.close(fig)


# Mushroom Dataset: categorical visual analysis
mushroom_path = "datasets/raw/mushroom/mushrooms.csv"
mushroom_df = pd.read_csv(mushroom_path)
mushroom_df["ClassLabel"] = mushroom_df["class"].map(
    {
        "e": "Edible",
        "p": "Poisonous",
    }
)

fig, ax = plt.subplots(figsize=(6, 4))

sns.countplot(
    data=mushroom_df,
    x="ClassLabel",
    hue="ClassLabel",
    palette="colorblind",
    legend=False,
    ax=ax,
)

ax.set_title("Class Distribution of the Mushroom Dataset")
ax.set_xlabel("Mushroom Class")
ax.set_ylabel("Number of Samples")

for container in ax.containers:
    ax.bar_label(container, padding=3)

sns.despine()
fig.tight_layout()
fig.savefig(output_dir / "mushroom_class_distribution.pdf")
fig.savefig(
    output_dir / "mushroom_class_distribution.png",
    dpi=300,
)
plt.close(fig)


mushroom_analysis_df = mushroom_df.assign(
    Poisonous=(mushroom_df["class"] == "p").astype(int)
)

odor_labels = {
    "a": "Almond",
    "c": "Creosote",
    "f": "Foul",
    "l": "Anise",
    "m": "Musty",
    "n": "None",
    "p": "Pungent",
    "s": "Spicy",
    "y": "Fishy",
}

habitat_labels = {
    "d": "Woods",
    "g": "Grasses",
    "l": "Leaves",
    "m": "Meadows",
    "p": "Paths",
    "u": "Urban",
    "w": "Waste",
}

odor_summary = (
    mushroom_analysis_df.groupby("odor", as_index=False)
    .agg(
        PoisonRate=("Poisonous", "mean"),
        SampleCount=("Poisonous", "size"),
    )
    .sort_values("PoisonRate")
)
odor_summary["OdorLabel"] = odor_summary["odor"].map(odor_labels)

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    odor_summary["OdorLabel"],
    odor_summary["PoisonRate"],
    color=sns.color_palette("viridis", len(odor_summary)),
)

ax.set_title("Poisonous Rate by Odor Category")
ax.set_xlabel("Odor")
ax.set_ylabel("Poisonous Rate")
ax.set_ylim(0, 1.08)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.tick_params(axis="x", rotation=25)
ax.bar_label(
    bars,
    labels=[f"{value:.0%}" for value in odor_summary["PoisonRate"]],
    padding=3,
    fontsize=8,
)

sns.despine()
fig.tight_layout()
fig.savefig(output_dir / "mushroom_poison_rate_by_odor.pdf")
fig.savefig(
    output_dir / "mushroom_poison_rate_by_odor.png",
    dpi=300,
)
plt.close(fig)


odor_habitat_rate = mushroom_analysis_df.pivot_table(
    index="odor",
    columns="habitat",
    values="Poisonous",
    aggfunc="mean",
)
odor_habitat_rate = odor_habitat_rate.rename(
    index=odor_labels,
    columns=habitat_labels,
)

fig, ax = plt.subplots(figsize=(9, 6))

sns.heatmap(
    odor_habitat_rate,
    cmap="RdYlGn_r",
    vmin=0,
    vmax=1,
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    cbar_kws={"label": "Poisonous Rate"},
    ax=ax,
)

ax.set_title("Poisonous Rate by Odor and Habitat")
ax.set_xlabel("Habitat")
ax.set_ylabel("Odor")

fig.tight_layout()
fig.savefig(output_dir / "mushroom_odor_habitat_heatmap.pdf")
fig.savefig(
    output_dir / "mushroom_odor_habitat_heatmap.png",
    dpi=300,
)
plt.close(fig)


# Credit Card Fraud Dataset: imbalance-aware visual analysis
credit_card_path = "datasets/raw/credit_card_fraud/creditcard.csv"
credit_features = ["V10", "V12", "V14", "V17", "Class"]
credit_df = pd.read_csv(credit_card_path, usecols=credit_features)
credit_df["ClassLabel"] = credit_df["Class"].map(
    {
        0: "Normal",
        1: "Fraud",
    }
)

credit_class_counts = credit_df["ClassLabel"].value_counts().reindex(
    ["Normal", "Fraud"]
)

fig, ax = plt.subplots(figsize=(6, 4))

bars = ax.bar(
    credit_class_counts.index,
    credit_class_counts.values,
    color=sns.color_palette("colorblind", 2),
)

ax.set_title("Credit Card Fraud Class Distribution")
ax.set_xlabel("Transaction Class")
ax.set_ylabel("Number of Transactions — Log Scale")
ax.set_yscale("log")
ax.bar_label(
    bars,
    labels=[f"{value:,}" for value in credit_class_counts.values],
    padding=3,
)

sns.despine()
fig.tight_layout()
fig.savefig(output_dir / "credit_card_class_distribution.pdf")
fig.savefig(
    output_dir / "credit_card_class_distribution.png",
    dpi=300,
)
plt.close(fig)


normal_transactions = credit_df[credit_df["Class"] == 0]
fraud_transactions = credit_df[credit_df["Class"] == 1]

normal_sample = normal_transactions.sample(
    n=min(6_000, len(normal_transactions)),
    random_state=42,
)

credit_plot_df = pd.concat(
    [normal_sample, fraud_transactions],
    ignore_index=True,
)

fig, ax = plt.subplots(figsize=(7, 5))

sns.scatterplot(
    data=credit_plot_df,
    x="V10",
    y="V14",
    hue="ClassLabel",
    style="ClassLabel",
    palette="colorblind",
    alpha=0.55,
    s=28,
    edgecolor="white",
    linewidth=0.2,
    ax=ax,
)

ax.set_title("Credit Card Transactions by V10 and V14")
ax.set_xlabel("V10 — PCA Component")
ax.set_ylabel("V14 — PCA Component")
ax.legend(title="Transaction Class")

sns.despine()
fig.tight_layout()
fig.savefig(output_dir / "credit_card_v10_v14_scatter.pdf")
fig.savefig(
    output_dir / "credit_card_v10_v14_scatter.png",
    dpi=300,
)
plt.close(fig)


credit_pair_features = ["V10", "V12", "V14", "V17"]

credit_pair_grid = sns.pairplot(
    data=credit_plot_df[credit_pair_features + ["ClassLabel"]],
    hue="ClassLabel",
    corner=True,
    diag_kind="hist",
    palette="colorblind",
    plot_kws={
        "alpha": 0.40,
        "s": 14,
        "rasterized": True,
    },
)

credit_pair_grid.fig.suptitle(
    "Selected PCA Components in Credit Card Transactions",
    y=1.02,
)
credit_pair_grid._legend.set_title("Transaction Class")

credit_pair_grid.savefig(output_dir / "credit_card_pairplot.pdf")
credit_pair_grid.savefig(
    output_dir / "credit_card_pairplot.png",
    dpi=300,
)
plt.close(credit_pair_grid.fig)


# Report-ready comparison table
table_columns = [
    "Dataset",
    "Source",
    "Samples",
    "Inputs",
    "Classes",
    "Balance",
    "Missing",
    "Leakage Audit",
    "Initial Structure",
    "License",
    "Academic Use",
]

table_rows = [
    [
        "Raisin",
        "UCI 850",
        "900",
        "7 numerical",
        "Kecimen / Besni",
        "450 / 450",
        "None",
        "No obvious leakage",
        "Overlap and curved feature relations",
        "CC BY 4.0",
        "Yes, with attribution",
    ],
    [
        "Mushroom",
        "UCI 73",
        "8,124",
        "22 categorical",
        "Edible / Poisonous",
        "4,208 / 3,916",
        "stalk-root: ?",
        "No obvious leakage",
        "Strong categorical interactions",
        "CC BY 4.0",
        "Yes, with attribution",
    ],
    [
        "Credit Card Fraud",
        "Kaggle / ULB",
        "284,807",
        "30 numerical; 28 PCA",
        "Normal / Fraud",
        "284,315 / 492",
        "None",
        "Limited by anonymized features",
        "Overlapping PCA regions",
        "ODbL / DbCL",
        "Yes, follow license and citation",
    ],
]

wrap_widths = [14, 14, 10, 16, 16, 14, 12, 20, 22, 16, 18]
wrapped_rows = [
    [
        fill(str(value), width=width)
        for value, width in zip(row, wrap_widths)
    ]
    for row in table_rows
]

fig, ax = plt.subplots(figsize=(21, 5.2))
ax.axis("off")

comparison_table = ax.table(
    cellText=wrapped_rows,
    colLabels=table_columns,
    cellLoc="center",
    colLoc="center",
    loc="center",
    bbox=[0, 0, 1, 1],
    colWidths=[
        0.090,
        0.075,
        0.060,
        0.090,
        0.090,
        0.080,
        0.075,
        0.105,
        0.115,
        0.095,
        0.125,
    ],
)

comparison_table.auto_set_font_size(False)
comparison_table.set_fontsize(8.2)

for (row_index, column_index), cell in comparison_table.get_celld().items():
    cell.set_edgecolor("#D0D0D0")
    cell.set_linewidth(0.7)
    cell.PAD = 0.025
    cell.get_text().set_wrap(True)

    if row_index == 0:
        cell.set_facecolor("#8C1515")
        cell.get_text().set_color("white")
        cell.get_text().set_weight("bold")
    elif row_index % 2 == 0:
        cell.set_facecolor("#F3F4F6")
    else:
        cell.set_facecolor("white")

fig.suptitle(
    "Comparison of Binary Classification Dataset Candidates",
    fontsize=15,
    fontweight="bold",
    y=0.98,
)

fig.savefig(output_dir / "candidate_comparison_table.pdf")
fig.savefig(
    output_dir / "candidate_comparison_table.png",
    dpi=300,
)
plt.close(fig)


selected_features = [
    "Area",
    "MajorAxisLength",
    "Eccentricity",
    "Perimeter",
]

pair_grid = sns.pairplot(
    data=df[selected_features + ["Class"]],
    hue="Class",
    corner=True,
    diag_kind="hist",
    palette="colorblind",
    plot_kws={
        "alpha": 0.55,
        "s": 24,
    },
)

pair_grid.fig.suptitle(
    "Pairwise Feature Relationships in the Raisin Dataset",
    y=1.02,
)

pair_grid.savefig(output_dir / "raisin_pairplot.pdf")
pair_grid.savefig(
    output_dir / "raisin_pairplot.png",
    dpi=300,
)
plt.close(pair_grid.fig)


fig, ax = plt.subplots(figsize=(7, 5))

sns.scatterplot(
    data=df,
    x="Eccentricity",
    y="Perimeter",
    hue="Class",
    style="Class",
    palette="colorblind",
    alpha=0.65,
    s=40,
    edgecolor="white",
    linewidth=0.3,
    ax=ax,
)

ax.set_title("Raisin Classes by Eccentricity and Perimeter")
ax.set_xlabel("Eccentricity")
ax.set_ylabel("Perimeter")

sns.despine()
fig.tight_layout()

fig.savefig(output_dir / "raisin_eccentricity_perimeter.pdf")
fig.savefig(
    output_dir / "raisin_eccentricity_perimeter.png",
    dpi=300,
)

plt.close(fig)
