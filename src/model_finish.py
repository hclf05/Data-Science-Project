"""Run the baseline finish model and create simple outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from utils import DATA_PROCESSED, OUTPUT_FIGURES, ROOT, ensure_directories


INPUT_FILE = DATA_PROCESSED / "analytic_dataset.csv"
FILTERED_OUTPUT_FILE = DATA_PROCESSED / "analytic_dataset_2010plus.csv"
MODEL_TABLE_FILE = ROOT / "output" / "model_table.txt"
ROBUSTNESS_FILE = ROOT / "output" / "robustness_results.txt"
COEFFICIENT_FIGURE = OUTPUT_FIGURES / "logit_odds_ratios.png"
WEIGHT_CLASS_FIGURE = OUTPUT_FIGURES / "finish_rate_by_weight_class.png"
YEAR_FIGURE = OUTPUT_FIGURES / "finish_rate_by_year.png"
AGE_GAP_BIN_FIGURE = OUTPUT_FIGURES / "finish_rate_by_age_gap_bin.png"
ANALYSIS_START_DATE = pd.Timestamp("2010-01-01")
FULL_MODEL_FORMULA = (
    "finish ~ reach_diff + age_diff + experience_diff + "
    "C(weight_class_grouped) + inactivity_diff + stance_mismatch"
)
FULL_MODEL_REQUIRED_COLUMNS = [
    "finish",
    "reach_diff",
    "age_diff",
    "experience_diff",
    "weight_class_grouped",
    "inactivity_diff",
    "stance_mismatch",
]


def filter_analysis_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict the analytic sample to modern UFC fights from 2010 onward."""
    filtered = df.copy()
    filtered["event_date"] = pd.to_datetime(filtered["event_date"], errors="coerce")
    filtered = filtered.loc[filtered["event_date"] >= ANALYSIS_START_DATE].copy()

    if filtered.empty:
        raise ValueError(
            "No fights remain after filtering to the 2010+ analysis window."
        )

    return filtered


def prepare_model_frame(df: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
    """Restrict the dataframe to the columns required for the model."""
    model_df = df[required_columns].dropna().copy()
    model_df["finish"] = model_df["finish"].astype(int)
    return model_df


def save_model_table(results: list[dict[str, object]], filtered_rows: int) -> None:
    """Write all model summaries to a single text table."""
    lines = [
        "UFC finish models (2010+)",
        "==========================",
        "",
        f"Analysis window: fights on or after {ANALYSIS_START_DATE.date().isoformat()}",
        f"Filtered dataset rows: {filtered_rows}",
        "",
    ]

    for result in results:
        label = str(result["label"])
        formula = str(result["formula"])
        sample_size = int(result["sample_size"])
        model = result["model"]

        lines.append(label)
        lines.append("-" * len(label))
        lines.append(f"Formula: {formula}")
        lines.append(f"Sample size: {sample_size}")
        lines.append("")
        lines.append(str(model.summary()))
        lines.append("")

    lines.extend(
        [
            "Outcome coding:",
            "- finish = 1 for KO/TKO and Submission outcomes (including doctor stoppages)",
            "- finish = 0 for Decision outcomes (including draws decided on scorecards)",
            "- NC, Overturned, Could Not Continue, DQ, and Other outcomes are excluded",
            "",
            "Interpretation note:",
            "- reach_diff, age_diff, experience_diff, and inactivity_diff are absolute gaps between fighters.",
            "- experience is prior UFC fights, not total MMA record.",
        ]
    )
    Path(MODEL_TABLE_FILE).write_text("\n".join(lines), encoding="utf-8")


def fit_logit_model(formula: str, data: pd.DataFrame):
    """Fit a logistic GLM model with a binomial family."""
    return smf.glm(
        formula=formula,
        data=data,
        family=sm.families.Binomial(),
    ).fit()


def save_robustness_results(results: list[dict[str, object]], filtered_rows: int) -> None:
    """Write robustness-test model summaries to disk."""
    lines = [
        "UFC finish model robustness checks (2010+)",
        "=========================================",
        "",
        f"Analysis window: fights on or after {ANALYSIS_START_DATE.date().isoformat()}",
        f"Filtered dataset rows: {filtered_rows}",
        "",
    ]

    for result in results:
        label = str(result["label"])
        sample_size = int(result["sample_size"])
        model = result["model"]

        lines.append(label)
        lines.append("-" * len(label))
        lines.append(f"Formula: {FULL_MODEL_FORMULA}")
        lines.append(f"Sample size: {sample_size}")
        lines.append("")
        lines.append(str(model.summary()))
        lines.append("")

    Path(ROBUSTNESS_FILE).write_text("\n".join(lines), encoding="utf-8")


def exclude_five_round_fights(df: pd.DataFrame) -> pd.DataFrame:
    """Remove fights scheduled for five rounds or more using time_format."""
    time_format = df["time_format"].fillna("").astype(str)
    scheduled_rounds = pd.to_numeric(
        time_format.str.extract(r"^\s*(\d+)")[0],
        errors="coerce",
    )
    return df.loc[~scheduled_rounds.ge(5).fillna(False)].copy()


def plot_coefficients(model) -> None:
    """Save a coefficient plot for the non-categorical predictors."""
    params = model.params
    conf_int = model.conf_int()

    rows = []
    for term, estimate in params.items():
        if term == "Intercept" or term.startswith("C(weight_class_grouped)"):
            continue
        rows.append(
            {
                "term": term,
                "odds_ratio": float(np.exp(estimate)),
                "ci_low": float(np.exp(conf_int.loc[term, 0])),
                "ci_high": float(np.exp(conf_int.loc[term, 1])),
            }
        )

    if not rows:
        return

    coef_df = pd.DataFrame(rows).sort_values(by="odds_ratio")
    labels = {
        "reach_diff": "Reach gap (inches)",
        "age_diff": "Age gap (years)",
        "experience_diff": "UFC experience gap",
        "inactivity_diff": "Inactivity gap (days)",
        "stance_mismatch": "Stance mismatch",
    }

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y_positions = np.arange(len(coef_df))
    xerr = np.vstack(
        [
            coef_df["odds_ratio"] - coef_df["ci_low"],
            coef_df["ci_high"] - coef_df["odds_ratio"],
        ]
    )

    ax.errorbar(
        coef_df["odds_ratio"],
        y_positions,
        xerr=xerr,
        fmt="o",
        color="#0f3b57",
        ecolor="#5d8aa8",
        capsize=4,
    )
    ax.axvline(1.0, color="#a61b29", linestyle="--", linewidth=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([labels.get(term, term) for term in coef_df["term"]])
    ax.set_xscale("log")
    ax.set_xlabel("Odds ratio (log scale)")
    ax.set_title("Logistic regression odds ratios")
    fig.tight_layout()
    fig.savefig(COEFFICIENT_FIGURE, dpi=300)
    plt.close(fig)


def plot_finish_rates(df: pd.DataFrame) -> None:
    """Save finish rates by weight class."""
    plot_df = (
        df.dropna(subset=["finish"])
        .groupby("weight_class_grouped", as_index=False)
        .agg(finish_rate=("finish", "mean"), fights=("finish", "size"))
        .sort_values(by="finish_rate", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(plot_df["weight_class_grouped"], plot_df["finish_rate"], color="#3a7d44")
    ax.set_xlabel("Finish rate")
    ax.set_ylabel("Weight class")
    ax.set_title("Finish rate by weight class")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(WEIGHT_CLASS_FIGURE, dpi=300)
    plt.close(fig)


def plot_finish_rate_by_year(df: pd.DataFrame) -> None:
    """Save yearly finish rates for the modern sample."""
    plot_df = df.dropna(subset=["finish"]).copy()
    plot_df["event_year"] = pd.to_datetime(plot_df["event_date"], errors="coerce").dt.year
    plot_df = (
        plot_df.dropna(subset=["event_year"])
        .groupby("event_year", as_index=False)
        .agg(finish_rate=("finish", "mean"), fights=("finish", "size"))
        .sort_values("event_year")
    )

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(
        plot_df["event_year"],
        plot_df["finish_rate"],
        marker="o",
        linewidth=2,
        color="#1b5e20",
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Finish rate")
    ax.set_title("Finish rate by year (2010+ sample)")
    ax.set_ylim(0.35, 0.75)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(YEAR_FIGURE, dpi=300)
    plt.close(fig)


def plot_finish_rate_by_age_gap_bin(df: pd.DataFrame) -> None:
    """Save finish rates by binned age-gap groups."""
    plot_df = df.dropna(subset=["finish", "age_diff"]).copy()
    bins = [0, 1, 2, 3, 4, 5, 7, 10, np.inf]
    labels = ["0-1", "1-2", "2-3", "3-4", "4-5", "5-7", "7-10", "10+"]
    plot_df["age_gap_bin"] = pd.cut(
        plot_df["age_diff"],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )

    grouped = (
        plot_df.groupby("age_gap_bin", as_index=False, observed=False)
        .agg(finish_rate=("finish", "mean"), fights=("finish", "size"))
    )

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(grouped["age_gap_bin"].astype(str), grouped["finish_rate"], color="#3f7cac")
    ax.set_xlabel("Age-gap bin (years)")
    ax.set_ylabel("Finish rate")
    ax.set_title("Finish rate by age-gap bin (2010+ sample)")
    ax.set_ylim(0.35, 0.75)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(AGE_GAP_BIN_FIGURE, dpi=300)
    plt.close(fig)


def main() -> None:
    """Fit the requested logistic models and save outputs."""
    ensure_directories()

    if not INPUT_FILE.exists():
        raise FileNotFoundError("Missing analytic dataset. Run build_features.py first.")

    df = pd.read_csv(INPUT_FILE)
    filtered_df = filter_analysis_sample(df)
    filtered_df.to_csv(FILTERED_OUTPUT_FILE, index=False, date_format="%Y-%m-%d")

    model_specs = [
        {
            "label": "MODEL 1",
            "formula": "finish ~ reach_diff + age_diff + experience_diff",
            "required_columns": ["finish", "reach_diff", "age_diff", "experience_diff"],
        },
        {
            "label": "MODEL 2",
            "formula": (
                "finish ~ reach_diff + age_diff + experience_diff + "
                "C(weight_class_grouped)"
            ),
            "required_columns": [
                "finish",
                "reach_diff",
                "age_diff",
                "experience_diff",
                "weight_class_grouped",
            ],
        },
        {
            "label": "MODEL 3",
            "formula": FULL_MODEL_FORMULA,
            "required_columns": FULL_MODEL_REQUIRED_COLUMNS,
        },
    ]

    model_results: list[dict[str, object]] = []
    for spec in model_specs:
        label = str(spec["label"])
        formula = str(spec["formula"])
        required_columns = list(spec["required_columns"])

        model_df = prepare_model_frame(filtered_df, required_columns)
        if model_df.empty:
            raise ValueError(f"No complete cases available for {label}.")

        model = fit_logit_model(formula=formula, data=model_df)

        model_results.append(
            {
                "label": label,
                "formula": formula,
                "sample_size": len(model_df),
                "model": model,
            }
        )
        print(f"{label} sample size: {len(model_df)}")

    save_model_table(model_results, filtered_rows=len(filtered_df))
    full_model = model_results[-1]["model"]

    men_only_df = filtered_df.loc[
        ~filtered_df["weight_class_grouped"].fillna("").str.contains(
            "Women",
            case=False,
        )
    ].copy()
    non_five_round_df = exclude_five_round_fights(filtered_df)

    robustness_specs = [
        {
            "label": "ROBUSTNESS TEST 1 — MEN ONLY",
            "subset": men_only_df,
        },
        {
            "label": "ROBUSTNESS TEST 2 — NON-5-ROUND FIGHTS",
            "subset": non_five_round_df,
        },
    ]
    robustness_results: list[dict[str, object]] = []
    for spec in robustness_specs:
        label = str(spec["label"])
        subset = spec["subset"]
        model_df = prepare_model_frame(subset, FULL_MODEL_REQUIRED_COLUMNS)
        if model_df.empty:
            raise ValueError(f"No complete cases available for {label}.")

        model = fit_logit_model(formula=FULL_MODEL_FORMULA, data=model_df)
        robustness_results.append(
            {
                "label": label,
                "sample_size": len(model_df),
                "model": model,
            }
        )
        print(f"{label} sample size: {len(model_df)}")

    save_robustness_results(robustness_results, filtered_rows=len(filtered_df))
    plot_coefficients(full_model)
    plot_finish_rates(filtered_df)
    plot_finish_rate_by_year(filtered_df)
    plot_finish_rate_by_age_gap_bin(filtered_df)

    print(f"Saved filtered 2010+ dataset to {FILTERED_OUTPUT_FILE}")
    print(f"Wrote model table to {MODEL_TABLE_FILE}")
    print(f"Wrote robustness results to {ROBUSTNESS_FILE}")
    print(f"Wrote coefficient plot to {COEFFICIENT_FIGURE}")
    print(f"Wrote finish-rate figure to {WEIGHT_CLASS_FIGURE}")
    print(f"Wrote finish-rate-by-year figure to {YEAR_FIGURE}")
    print(f"Wrote age-gap-binned finish-rate figure to {AGE_GAP_BIN_FIGURE}")


if __name__ == "__main__":
    main()
