"""Build the analytic dataset for modelling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils import DATA_INTERIM, DATA_PROCESSED, ensure_directories


MERGED_FILE = DATA_INTERIM / "fights_merged.csv"
OUTPUT_FILE = DATA_PROCESSED / "analytic_dataset.csv"


def group_weight_class(value: str) -> str:
    """Collapse rare or irregular weight classes used for modelling."""
    cleaned = str(value).strip()
    if not cleaned:
        return "Unknown"

    standard_classes = {
        "Women's Strawweight",
        "Women's Flyweight",
        "Women's Bantamweight",
        "Women's Featherweight",
        "Strawweight",
        "Flyweight",
        "Bantamweight",
        "Featherweight",
        "Lightweight",
        "Welterweight",
        "Middleweight",
        "Light Heavyweight",
        "Heavyweight",
    }
    if cleaned in standard_classes:
        return cleaned

    return "Other"


def add_experience_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add prior UFC fights and inactivity for both fighters."""
    appearances = []
    for corner in (1, 2):
        fighter_url_col = f"fighter_{corner}_url"
        temp = df[["fight_url", "event_date", fighter_url_col]].copy()
        temp = temp.rename(columns={fighter_url_col: "fighter_url"})
        temp["corner"] = corner
        appearances.append(temp)

    history = pd.concat(appearances, ignore_index=True)
    history = history.dropna(subset=["fighter_url"]).copy()
    history = history[history["fighter_url"].astype(str).str.len() > 0].copy()
    history = history.sort_values(by=["fighter_url", "event_date", "fight_url"])
    history["ufc_fights_before"] = history.groupby("fighter_url").cumcount()
    history["previous_event_date"] = history.groupby("fighter_url")["event_date"].shift()
    history["days_since_last_fight"] = (
        history["event_date"] - history["previous_event_date"]
    ).dt.days

    output = df.copy()
    for corner in (1, 2):
        side_history = history.loc[history["corner"] == corner, [
            "fight_url",
            "ufc_fights_before",
            "days_since_last_fight",
        ]].rename(
            columns={
                "ufc_fights_before": f"fighter_{corner}_ufc_fights_before",
                "days_since_last_fight": f"fighter_{corner}_days_since_last_fight",
            }
        )
        output = output.merge(side_history, how="left", on="fight_url")

    return output


def main() -> None:
    """Run the feature-engineering step."""
    ensure_directories()

    if not MERGED_FILE.exists():
        raise FileNotFoundError("Missing merged file. Run clean_data.py first.")

    df = pd.read_csv(MERGED_FILE)
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    for corner in (1, 2):
        df[f"fighter_{corner}_profile_date_of_birth"] = pd.to_datetime(
            df[f"fighter_{corner}_profile_date_of_birth"], errors="coerce"
        )

    df = df.drop_duplicates(subset=["fight_url"]).sort_values(
        by=["event_date", "fight_url"]
    )
    df = add_experience_features(df)

    method_family = df["method_family"].fillna("").astype(str).str.strip()
    result_type = df["result_type"].fillna("").astype(str).str.strip()
    method_family_lower = method_family.str.casefold()
    result_type_lower = result_type.str.casefold()

    # Outcome coding for the empirical question:
    # - finish = 1 for KO/TKO and Submission outcomes (KO/TKO includes doctor stoppages)
    # - finish = 0 for Decision outcomes (including decision draws)
    # - exclude NC / Overturned / Could Not Continue / DQ / Other from the binary outcome
    finish_mask = method_family_lower.isin({"ko/tko", "submission"})
    decision_mask = method_family_lower.eq("decision") | result_type_lower.eq("draw")
    excluded_mask = (
        result_type_lower.eq("nc")
        | method_family_lower.isin({"overturned", "could not continue", "dq", "other"})
    )

    df["finish"] = np.nan
    df.loc[decision_mask, "finish"] = 0
    df.loc[finish_mask, "finish"] = 1
    df.loc[excluded_mask, "finish"] = np.nan

    for corner in (1, 2):
        df[f"fighter_{corner}_age"] = (
            df["event_date"] - df[f"fighter_{corner}_profile_date_of_birth"]
        ).dt.days / 365.25

    df["reach_diff"] = (
        df["fighter_1_profile_reach_in"] - df["fighter_2_profile_reach_in"]
    ).abs()
    df["age_diff"] = (df["fighter_1_age"] - df["fighter_2_age"]).abs()
    df["experience_diff"] = (
        df["fighter_1_ufc_fights_before"] - df["fighter_2_ufc_fights_before"]
    ).abs()
    df["inactivity_diff"] = (
        df["fighter_1_days_since_last_fight"]
        - df["fighter_2_days_since_last_fight"]
    ).abs()

    stance_1 = df["fighter_1_profile_stance"]
    stance_2 = df["fighter_2_profile_stance"]
    known_stances = stance_1.notna() & stance_2.notna()
    df["stance_mismatch"] = np.where(
        known_stances,
        (stance_1 != stance_2).astype(float),
        np.nan,
    )

    df["weight_class_grouped"] = df["weight_class"].map(group_weight_class)
    df["complete_case_baseline"] = df[
        ["finish", "reach_diff", "age_diff", "experience_diff", "weight_class_grouped"]
    ].notna().all(axis=1)

    df = df.sort_values(by=["event_date", "fight_url"]).reset_index(drop=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Wrote analytic dataset to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
