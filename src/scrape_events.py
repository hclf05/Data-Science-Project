"""Scrape UFC event pages and collect fight links."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

import pandas as pd
import requests

from utils import (
    DATA_RAW,
    UFCSTATS_BASE_URL,
    canonicalize_ufcstats_url,
    ensure_directories,
    get_soup,
    get_thread_session,
    normalise_whitespace,
    safe_text,
)


EVENTS_URL = f"{UFCSTATS_BASE_URL}/statistics/events/completed?page=all"
OUTPUT_FILE = DATA_RAW / "fight_links.csv"
DEFAULT_MAX_WORKERS = 8


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-date",
        default=None,
        help="Optional lower bound for event_date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Number of concurrent workers to use (default: {DEFAULT_MAX_WORKERS}).",
    )
    return parser.parse_args()


def extract_event_links() -> list[dict[str, str]]:
    """Extract event names, dates, and event URLs from the completed events page."""
    session = get_thread_session()
    soup = get_soup(session, EVENTS_URL)

    rows = soup.select("tr.b-statistics__table-row")
    events: list[dict[str, str]] = []

    for row in rows:
        link = row.select_one("a.b-link.b-link_style_black")
        if link is None:
            continue

        href = link.get("href", "").strip()
        if not href or "event-details" not in href:
            continue

        event_name = safe_text(link)
        event_date = safe_text(row.select_one("span.b-statistics__date"))

        if not event_name or not event_date:
            continue

        parsed_date = pd.to_datetime(event_date, errors="coerce")
        if pd.isna(parsed_date):
            continue

        events.append(
            {
                "event_name": event_name,
                "event_date": parsed_date.date().isoformat(),
                "event_url": canonicalize_ufcstats_url(href),
            }
        )

    if not events:
        raise ValueError(
            "No event links found on the completed events page. "
            "The UFCStats HTML structure may have changed."
        )

    return events


def filter_events_by_date(
    events: list[dict[str, str]], min_date: str | None
) -> list[dict[str, str]]:
    """Filter events to a minimum ISO date if requested."""
    if min_date is None:
        return events

    threshold = pd.to_datetime(min_date, errors="coerce")
    if pd.isna(threshold):
        raise ValueError(f"Invalid --min-date value: {min_date}")

    return [
        event
        for event in events
        if pd.to_datetime(event["event_date"], errors="coerce") >= threshold
    ]


def extract_fight_links_from_event(event: dict[str, str]) -> list[dict[str, str]]:
    """Extract fight detail URLs from a single event page."""
    session = get_thread_session()
    soup = get_soup(session, event["event_url"])

    rows = soup.select("tr.b-fight-details__table-row")
    records: list[dict[str, str]] = []

    for row in rows:
        href = normalise_whitespace(row.get("data-link", ""))

        if not href:
            anchor = row.select_one('a[href*="fight-details"]')
            if anchor is not None:
                href = normalise_whitespace(anchor.get("href", ""))

        if not href or "fight-details" not in href:
            continue

        records.append(
            {
                "event_name": event["event_name"],
                "event_date": event["event_date"],
                "fight_url": canonicalize_ufcstats_url(href),
            }
        )

    return records


def deduplicate_records(records: Iterable[dict[str, str]]) -> pd.DataFrame:
    """Convert records to a de-duplicated dataframe."""
    df = pd.DataFrame(records, columns=["event_name", "event_date", "fight_url"])
    if df.empty:
        return df

    for column in ["event_name", "event_date", "fight_url"]:
        df[column] = df[column].astype(str).str.strip()

    df = df.drop_duplicates(subset=["fight_url"]).sort_values(
        by=["event_date", "event_name", "fight_url"],
        ascending=[False, True, True],
    )
    return df.reset_index(drop=True)


def main() -> None:
    """Run the event scraper."""
    args = parse_args()
    ensure_directories()

    print(f"Fetching completed events from: {EVENTS_URL}")
    all_events = extract_event_links()
    events = filter_events_by_date(all_events, args.min_date)
    print(
        f"Found {len(all_events)} events in total; "
        f"processing {len(events)} after date filtering"
    )

    all_records: list[dict[str, str]] = []
    failures: list[str] = []

    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        futures = {
            executor.submit(extract_fight_links_from_event, event): event
            for event in events
        }

        for index, future in enumerate(as_completed(futures), start=1):
            event = futures[future]
            try:
                all_records.extend(future.result())
            except requests.RequestException as exc:
                failures.append(event["event_url"])
                print(f"Request failed for {event['event_url']}: {exc}")
            except Exception as exc:
                failures.append(event["event_url"])
                print(f"Unexpected error for {event['event_url']}: {exc}")

            if index % 50 == 0 or index == len(futures):
                print(f"Processed {index}/{len(futures)} event pages")

    df = deduplicate_records(all_records)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Wrote {len(df)} fight links to {OUTPUT_FILE}")
    if failures:
        print(f"Skipped {len(failures)} event pages due to errors")


if __name__ == "__main__":
    main()
