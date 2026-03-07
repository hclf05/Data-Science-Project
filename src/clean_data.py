"""Clean raw UFC fight and fighter data."""

from __future__ import annotations

import pandas as pd

from utils import (
    DATA_INTERIM,
    DATA_RAW,
    ensure_directories,
    normalise_whitespace,
    parse_height_to_inches,
    parse_reach_to_inches,
    parse_weight_to_lbs,
)


FIGHTS_RAW = DATA_RAW / "fights_raw.csv"
FIGHTERS_RAW = DATA_RAW / "fighters_raw.csv"
FIGHTS_CLEAN = DATA_INTERIM / "fights_clean.csv"
FIGHTERS_CLEAN = DATA_INTERIM / "fighters_clean.csv"
FIGHTS_MERGED = DATA_INTERIM / "fights_merged.csv"


def clean_method(value: str) -> str:
    """Standardise the method text."""
    return normalise_whitespace(str(value))


def classify_method_family(value: str) -> str:
    """Collapse detailed method labels into broad families."""
    method = clean_method(value).upper()
    if not method:
        return ""
    if "DECISION" in method:
        return "Decision"
    if "SUBMISSION" in method:
        return "Submission"
    if "KO/TKO" in method or method.startswith("TKO") or method.startswith("KO"):
        return "KO/TKO"
    if "DQ" in method or "DISQUAL" in method:
        return "DQ"
    return method.title()


def standardise_weight_class(value: str) -> str:
    """Standardise weight-class labels to a consistent set."""
    cleaned = normalise_whitespace(str(value))
    if not cleaned:
        return ""

    replacements = {
        "Women'S Strawweight": "Women's Strawweight",
        "Women'S Flyweight": "Women's Flyweight",
        "Women'S Bantamweight": "Women's Bantamweight",
        "Women'S Featherweight": "Women's Featherweight",
        "Catch Weight": "Catchweight",
    }
    title_cased = cleaned.title()
    return replacements.get(title_cased, title_cased)


def clean_stance(value: str) -> str | None:
    """Standardise stance labels."""
    cleaned = normalise_whitespace(str(value))
    if not cleaned:
        return None

    mapping = {
        "Orthodox": "Orthodox",
        "Southpaw": "Southpaw",
        "Switch": "Switch",
        "Open Stance": "Open Stance",
        "Sideways": "Sideways",
    }
    return mapping.get(cleaned.title(), cleaned.title())


def parse_time_to_seconds(value: str) -> float | None:
    """Convert a mm:ss fight time string into seconds."""
    cleaned = normalise_whitespace(str(value))
    if ":" not in cleaned:
        return None

    minutes, seconds = cleaned.split(":", maxsplit=1)
    try:
        return float(int(minutes) * 60 + int(seconds))
    except ValueError:
        return None


def clean_fights(fights: pd.DataFrame) -> pd.DataFrame:
    """Clean fight-level fields."""
    fights = fights.copy()
    fights = fights.drop_duplicates(subset=["fight_url"]).reset_index(drop=True)

    text_columns = [
        "event_name",
        "fight_url",
        "fight_title",
        "weight_class_raw",
        "method",
        "time",
        "time_format",
        "referee",
        "details",
        "fighter_1_name",
        "fighter_1_url",
        "fighter_1_status",
        "fighter_2_name",
        "fighter_2_url",
        "fighter_2_status",
        "winner_name",
        "result_type",
    ]
    for column in text_columns:
        fights[column] = fights[column].fillna("").astype(str).map(normalise_whitespace)

    fights["event_date"] = pd.to_datetime(fights["event_date"], errors="coerce")
    fights["round"] = pd.to_numeric(fights["round"], errors="coerce")
    fights["time_seconds"] = fights["time"].map(parse_time_to_seconds)
    fights["method"] = fights["method"].map(clean_method)
    fights["method_family"] = fights["method"].map(classify_method_family)
    fights["weight_class"] = fights["weight_class_raw"].map(standardise_weight_class)

    return fights.sort_values(by=["event_date", "fight_url"]).reset_index(drop=True)


def clean_fighters(fighters: pd.DataFrame) -> pd.DataFrame:
    """Clean fighter-profile fields."""
    fighters = fighters.copy()
    fighters = fighters.drop_duplicates(subset=["fighter_url"]).reset_index(drop=True)

    text_columns = [
        "fighter_name",
        "fighter_url",
        "height",
        "weight",
        "reach",
        "stance",
        "date_of_birth",
    ]
    for column in text_columns:
        fighters[column] = fighters[column].fillna("").astype(str).map(normalise_whitespace)

    fighters = fighters.rename(
        columns={
            "height": "height_raw",
            "weight": "weight_raw",
            "reach": "reach_raw",
            "stance": "stance_raw",
        }
    )

    fighters["height_in"] = fighters["height_raw"].map(parse_height_to_inches)
    fighters["weight_lbs"] = fighters["weight_raw"].map(parse_weight_to_lbs)
    fighters["reach_in"] = fighters["reach_raw"].map(parse_reach_to_inches)
    fighters["stance"] = fighters["stance_raw"].map(clean_stance)
    fighters["date_of_birth"] = fighters["date_of_birth"].replace({"--": pd.NA, "": pd.NA})
    fighters["date_of_birth"] = pd.to_datetime(
        fighters["date_of_birth"], errors="coerce", format="%b %d, %Y"
    )

    return fighters.sort_values(by=["fighter_name", "fighter_url"]).reset_index(drop=True)


def merge_fighter_profiles(
    fights: pd.DataFrame, fighters: pd.DataFrame
) -> pd.DataFrame:
    """Attach cleaned fighter profiles to both sides of each fight."""
    fighter_features = fighters.rename(
        columns={
            "fighter_name": "profile_name",
            "fighter_url": "profile_url",
            "height_raw": "profile_height_raw",
            "weight_raw": "profile_weight_raw",
            "reach_raw": "profile_reach_raw",
            "stance_raw": "profile_stance_raw",
            "height_in": "profile_height_in",
            "weight_lbs": "profile_weight_lbs",
            "reach_in": "profile_reach_in",
            "stance": "profile_stance",
            "date_of_birth": "profile_date_of_birth",
        }
    )

    fighter_1 = fighter_features.add_prefix("fighter_1_").rename(
        columns={"fighter_1_profile_url": "fighter_1_url"}
    )
    fighter_2 = fighter_features.add_prefix("fighter_2_").rename(
        columns={"fighter_2_profile_url": "fighter_2_url"}
    )

    merged = fights.merge(fighter_1, how="left", on="fighter_1_url")
    merged = merged.merge(fighter_2, how="left", on="fighter_2_url")
    return merged


def main() -> None:
    """Run the raw-data cleaning step."""
    ensure_directories()

    if not FIGHTS_RAW.exists() or not FIGHTERS_RAW.exists():
        raise FileNotFoundError("Missing raw files. Run scraping scripts first.")

    fights_raw = pd.read_csv(FIGHTS_RAW)
    fighters_raw = pd.read_csv(FIGHTERS_RAW)

    fights_clean = clean_fights(fights_raw)
    fighters_clean = clean_fighters(fighters_raw)
    fights_merged = merge_fighter_profiles(fights_clean, fighters_clean)

    fights_clean.to_csv(FIGHTS_CLEAN, index=False)
    fighters_clean.to_csv(FIGHTERS_CLEAN, index=False)
    fights_merged.to_csv(FIGHTS_MERGED, index=False)

    print(f"Wrote cleaned fight data to {FIGHTS_CLEAN}")
    print(f"Wrote cleaned fighter data to {FIGHTERS_CLEAN}")
    print(f"Wrote merged fight data to {FIGHTS_MERGED}")


if __name__ == "__main__":
    main()
