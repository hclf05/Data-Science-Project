# Project Checklist

## Project anchor
- [ ] Title locked: What Predicts UFC Fight Outcomes?
- [ ] Main question locked: Which pre-fight characteristics are associated with a UFC fight ending in a finish rather than a decision?
- [ ] Outcome variable defined: `finish = 1` if not Decision, else `0`
- [ ] Core predictors agreed: reach difference, age difference, experience difference, inactivity difference if feasible, weight class

## Scope control
- [ ] Every chart helps answer the main question
- [ ] Every model helps answer the main question
- [ ] Every paragraph helps answer the main question
- [ ] Side quests postponed or cut

## Risk rules
- [ ] If old UFCStats pages are inconsistent, apply a date cutoff such as 2015 onward
- [ ] If `days_since_last_fight` still causes friction by Hour 9, drop it
- [ ] Protect a clean and reproducible dataset over a larger messy one

## File targets
- [ ] `data/raw/fight_links.csv`
- [ ] `data/raw/fights_raw.csv`
- [ ] `data/raw/fighters_raw.csv`
- [ ] `data/interim/fights_clean.csv`
- [ ] `data/interim/fighters_clean.csv`
- [ ] `data/interim/fights_merged.csv`
- [ ] `data/processed/analytic_dataset.csv`
- [ ] figures in `output/figures/`
- [ ] final blog output rendered from `blog.qmd`

## Build order
- [ ] `src/scrape_events.py`
- [ ] `src/scrape_fights.py`
- [ ] `src/clean_data.py`
- [ ] `src/build_features.py`
- [ ] `src/model_finish.py`

## Output targets
- [ ] Finish rate by weight class
- [ ] Reach difference vs finish rate
- [ ] Age difference vs finish probability
- [ ] Inactivity gap vs finish probability if feasible
- [ ] Regression output
- [ ] One model-based visual

## Repo hygiene
- [ ] Commit with clear messages
- [ ] Keep raw data untouched
- [ ] Keep processed data script-generated
- [ ] Keep README accurate
- [ ] Keep folder structure clean

## Final rule
- [ ] When lost, ask: does this help answer what predicts whether a UFC fight ends in a finish rather than a decision?
