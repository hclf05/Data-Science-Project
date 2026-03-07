"""Scrape fight-level and fighter-level details from UFCStats."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from utils import (
    DATA_RAW,
    canonicalize_ufcstats_url,
    ensure_directories,
    get_soup,
    get_thread_session,
    parse_labelled_value,
    safe_text,
)


FIGHT_LINKS_FILE = DATA_RAW / "fight_links.csv"
FIGHTS_OUTPUT = DATA_RAW / "fights_raw.csv"
FIGHTERS_OUTPUT = DATA_RAW / "fighters_raw.csv"
DEFAULT_MAX_WORKERS = 8


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Number of concurrent workers to use (default: {DEFAULT_MAX_WORKERS}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of fight pages to scrape for a quick smoke test.",
    )
    return parser.parse_args()


def strip_label_prefix(value: str, label: str) -> str:
    """Remove a label prefix such as 'Method:' from text."""
    prefix = f"{label}:"
    return value[len(prefix) :].strip() if value.startswith(prefix) else value.strip()


def parse_weight_class(fight_title: str) -> str:
    """Convert page titles like 'Flyweight Bout' into a raw weight-class label."""
    cleaned = fight_title.replace("Bout", "").replace(" bout", "").strip()
    return " ".join(cleaned.split())


def parse_result_type(statuses: list[str]) -> str:
    """Map per-fighter status codes into a fight-level result label."""
    unique_statuses = {status.upper() for status in statuses if status}
    if unique_statuses == {"W", "L"}:
        return "Win/Loss"
    if unique_statuses == {"D"}:
        return "Draw"
    if unique_statuses == {"N"}:
        return "No Contest"
    return "/".join(sorted(unique_statuses)) or "Unknown"


def parse_fight_details(row: dict[str, str]) -> dict[str, str]:
    """Extract a structured fight record from a UFCStats fight page."""
    session = get_thread_session()
    soup = get_soup(session, row["fight_url"])

    people: list[dict[str, str]] = []
    for person in soup.select("div.b-fight-details__person"):
        link = person.select_one("a.b-link.b-fight-details__person-link")
        if link is None:
            continue

        people.append(
            {
                "fighter_name": safe_text(link),
                "fighter_url": canonicalize_ufcstats_url(link.get("href", "")),
                "status": safe_text(
                    person.select_one("i.b-fight-details__person-status")
                ).upper(),
            }
        )

    if len(people) != 2:
        raise ValueError(
            f"Expected two fighters on {row['fight_url']}, found {len(people)}"
        )

    metadata: dict[str, str] = {}
    for item in soup.select("i.b-fight-details__text-item_first, i.b-fight-details__text-item"):
        label_node = item.select_one("i.b-fight-details__label")
        if label_node is None:
            continue

        label = safe_text(label_node).rstrip(":")
        key = label.lower().replace(" ", "_")
        value = strip_label_prefix(safe_text(item), label)
        if key == "details":
            continue
        metadata[key] = value

    detail_blocks = soup.select("p.b-fight-details__text")
    details = ""
    if len(detail_blocks) > 1:
        details = safe_text(detail_blocks[1]).removeprefix("Details:").strip()

    fight_title = safe_text(soup.select_one("i.b-fight-details__fight-title"))
    winner_name = next(
        (person["fighter_name"] for person in people if person["status"] == "W"),
        "",
    )

    return {
        "event_name": row["event_name"],
        "event_date": row["event_date"],
        "fight_url": row["fight_url"],
        "fight_title": fight_title,
        "weight_class_raw": parse_weight_class(fight_title),
        "method": metadata.get("method", ""),
        "round": metadata.get("round", ""),
        "time": metadata.get("time", ""),
        "time_format": metadata.get("time_format", ""),
        "referee": metadata.get("referee", ""),
        "details": details,
        "fighter_1_name": people[0]["fighter_name"],
        "fighter_1_url": people[0]["fighter_url"],
        "fighter_1_status": people[0]["status"],
        "fighter_2_name": people[1]["fighter_name"],
        "fighter_2_url": people[1]["fighter_url"],
        "fighter_2_status": people[1]["status"],
        "winner_name": winner_name,
        "result_type": parse_result_type([people[0]["status"], people[1]["status"]]),
    }


def parse_fighter_profile(fighter: dict[str, str]) -> dict[str, str]:
    """Extract a fighter profile from a UFCStats fighter page."""
    session = get_thread_session()
    soup = get_soup(session, fighter["fighter_url"])

    values: dict[str, str] = {}
    for item in soup.select("li.b-list__box-list-item"):
        key, value = parse_labelled_value(safe_text(item))
        values[key] = value

    return {
        "fighter_name": safe_text(soup.select_one("span.b-content__title-highlight"))
        or fighter["fighter_name"],
        "fighter_url": fighter["fighter_url"],
        "height": values.get("height", ""),
        "weight": values.get("weight", ""),
        "reach": values.get("reach", ""),
        "stance": values.get("stance", ""),
        "date_of_birth": values.get("dob", ""),
    }


def deduplicate(df: pd.DataFrame, subset: list[str]) -> pd.DataFrame:
    """Drop duplicates and sort rows predictably."""
    if df.empty:
        return df
    return df.drop_duplicates(subset=subset).reset_index(drop=True)


def collect_fight_records(
    fight_links: pd.DataFrame, max_workers: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scrape fight pages and collect both fight rows and fighter URLs."""
    fight_records: list[dict[str, str]] = []
    fighter_records: list[dict[str, str]] = []
    failures: list[str] = []

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        futures = {
            executor.submit(parse_fight_details, row._asdict()): row.fight_url
            for row in fight_links.itertuples(index=False)
        }

        for index, future in enumerate(as_completed(futures), start=1):
            fight_url = futures[future]
            try:
                record = future.result()
                fight_records.append(record)
                fighter_records.extend(
                    [
                        {
                            "fighter_name": record["fighter_1_name"],
                            "fighter_url": record["fighter_1_url"],
                        },
                        {
                            "fighter_name": record["fighter_2_name"],
                            "fighter_url": record["fighter_2_url"],
                        },
                    ]
                )
            except requests.RequestException as exc:
                failures.append(fight_url)
                print(f"Request failed for {fight_url}: {exc}")
            except Exception as exc:
                failures.append(fight_url)
                print(f"Unexpected error for {fight_url}: {exc}")

            if index % 100 == 0 or index == len(futures):
                print(f"Processed {index}/{len(futures)} fight pages")

    fights_df = deduplicate(pd.DataFrame(fight_records), subset=["fight_url"])
    fighters_df = deduplicate(pd.DataFrame(fighter_records), subset=["fighter_url"])

    if failures:
        print(f"Skipped {len(failures)} fight pages due to errors")

    return fights_df, fighters_df


def collect_fighter_profiles(
    fighters: pd.DataFrame, max_workers: int
) -> pd.DataFrame:
    """Scrape fighter profile pages for unique fighters."""
    profile_records: list[dict[str, str]] = []
    failures: list[str] = []

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        futures = {
            executor.submit(parse_fighter_profile, row._asdict()): row.fighter_url
            for row in fighters.itertuples(index=False)
        }

        for index, future in enumerate(as_completed(futures), start=1):
            fighter_url = futures[future]
            try:
                profile_records.append(future.result())
            except requests.RequestException as exc:
                failures.append(fighter_url)
                print(f"Request failed for {fighter_url}: {exc}")
            except Exception as exc:
                failures.append(fighter_url)
                print(f"Unexpected error for {fighter_url}: {exc}")

            if index % 100 == 0 or index == len(futures):
                print(f"Processed {index}/{len(futures)} fighter pages")

    fighters_df = deduplicate(pd.DataFrame(profile_records), subset=["fighter_url"])

    if failures:
        print(f"Skipped {len(failures)} fighter pages due to errors")

    return fighters_df


def main() -> None:
    """Run the fight and fighter scraper."""
    args = parse_args()
    ensure_directories()

    if not FIGHT_LINKS_FILE.exists():
        raise FileNotFoundError(
            f"Missing {FIGHT_LINKS_FILE}. Run scrape_events.py first."
        )

    fight_links = pd.read_csv(FIGHT_LINKS_FILE)
    if args.limit is not None:
        fight_links = fight_links.head(args.limit).copy()

    if fight_links.empty:
        raise ValueError("fight_links.csv is empty. Run scrape_events.py again.")

    print(f"Scraping {len(fight_links)} fight pages from {FIGHT_LINKS_FILE}")
    fights_df, fighters_seed = collect_fight_records(
        fight_links=fight_links,
        max_workers=args.max_workers,
    )
    fighter_profiles_df = collect_fighter_profiles(
        fighters=fighters_seed,
        max_workers=args.max_workers,
    )

    fights_df.to_csv(FIGHTS_OUTPUT, index=False)
    fighter_profiles_df.to_csv(FIGHTERS_OUTPUT, index=False)

    print(f"Wrote {len(fights_df)} fights to {FIGHTS_OUTPUT}")
    print(f"Wrote {len(fighter_profiles_df)} fighter profiles to {FIGHTERS_OUTPUT}")


if __name__ == "__main__":
    main()
