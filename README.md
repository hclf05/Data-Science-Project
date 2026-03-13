# UFC Fight Outcomes: Empirical Analysis

This repository contains an undergraduate empirical data science project on UFC fight outcomes. It builds a full pipeline from UFCStats scraping to feature engineering, logistic regression, robustness checks, and a rendered Quarto report.

**Live project output:** [View the blog here](https://hclf05.github.io/Data-Science-Project/)

**Repository copy of the final report:** [`blog.html`](blog.html)

## Research Question
Which pre-fight characteristics are associated with a UFC fight ending in a finish rather than a decision?

## Motivation
Reach, age, inactivity, and experience are often discussed as practical predictors of fight outcomes. This project tests those claims in a structured way, while controlling for weight class and interpreting results as associations rather than causal effects.

## Data Source
Data are scraped from [UFCStats](http://ufcstats.com), using:
- completed event pages
- fight-level detail pages
- fighter profile pages

Outcome coding used in the analysis:
- `finish = 1` for KO/TKO and Submission outcomes (including doctor stoppages recorded as TKO)
- `finish = 0` for Decision outcomes (decision draws are coded as decisions)
- No Contest, Overturned, Could Not Continue, DQ, and Other outcomes are excluded from the binary outcome

## Final Analytical Sample
- Full scraped sample: `8,560` fights
- Full scraped fighter profiles: `2,662` fighters
- Modern sample (2010 onwards): `7,295` fights
- Model 1 sample: `6,773` fights
- Model 2 sample: `6,773` fights
- Model 3 sample: `5,450` fights

Time coverage:
- Full scraped period: `1994-03-11` to `2026-02-28`
- Modern analytical period: `2010-01-02` to `2026-02-28`

## Project Structure
- `src/`: scraping, cleaning, feature construction, modelling
- `data/raw/`: scraped raw files
- `data/interim/`: cleaned and merged intermediate files
- `data/processed/`: final modelling datasets
- `output/`: regression tables and figures
- `blog.qmd`: source for the final report
- `blog.html`: rendered report in the repository root

## Replication
Run all commands from the repository root.

**Tested environment:** Python 3.9.6.

### 1) Environment setup
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
quarto --version
