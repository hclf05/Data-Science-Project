# UFC Fight Outcomes — Empirical Analysis

This project builds a reproducible data pipeline to study which pre-fight characteristics are associated with UFC fights ending in a finish rather than a decision. Using fight and fighter data scraped from UFCStats, it constructs a fight-level analytical dataset and estimates logistic regression models of finish probability. The project is designed as an undergraduate empirical data science workflow: scrape, clean, engineer features, estimate models, and report results.

## Research Question
Which pre-fight characteristics are associated with a UFC fight ending in a finish rather than a decision?

## Data
The dataset is constructed by scraping publicly available fight statistics from UFCStats.

The scraping pipeline collects:
- UFC event pages
- individual fight pages
- fighter profile pages

Sample sizes used in the project:
- Full scraped sample: `8,560` fights
- Full scraped fighter profiles: `2,662` fighters
- Modern sample (2010 onwards): `7,295` fights
- Model 1 and Model 2 estimation sample: `6,773` fights
- Model 3 estimation sample (after missing-value filtering): `5,450` fights

Timeframes:
- Full scraped period: `1994-03-11` to `2026-02-28`
- Modern analytical period: `2010-01-02` to `2026-02-28`

## Key Variables
The analytical dataset contains the following variables used in the empirical model.

### finish
Binary outcome variable.

- `finish = 1` for KO/TKO and Submission outcomes (KO/TKO includes doctor stoppages)
- `finish = 0` for Decision outcomes
- Decision draws are coded as `finish = 0`
- No Contest, Overturned, Could Not Continue, DQ, and Other outcomes are excluded from the binary outcome

### reach_diff
Absolute difference in reach between fighters (inches).

- `reach_diff = |reach_A - reach_B|`

### age_diff
Absolute difference in age at fight date (years).

- `age_diff = |age_A - age_B|`

### experience_diff
Absolute difference in prior UFC fight experience.

- `experience_diff = |prior_UFC_fights_A - prior_UFC_fights_B|`

### inactivity_diff
Absolute difference in inactivity (days since previous UFC fight).

- `inactivity_diff = |days_since_last_fight_A - days_since_last_fight_B|`

### stance_mismatch
Binary indicator for stance mismatch.

- `stance_mismatch = 1` if fighters have different listed stances
- `stance_mismatch = 0` if fighters have the same listed stance

### weight_class_grouped
Categorical weight class variable used in regression controls (for example Lightweight, Welterweight, Heavyweight).

## Empirical Method
The project estimates logistic regression models (`statsmodels` GLM with binomial family) to analyse the probability that a fight ends in a finish. Because the dependent variable is binary, coefficients are interpreted in terms of log-odds (or odds ratios after transformation).

## Model Specifications
- Model 1: `finish ~ reach_diff + age_diff + experience_diff`
- Model 2: `finish ~ reach_diff + age_diff + experience_diff + C(weight_class_grouped)`
- Model 3: `finish ~ reach_diff + age_diff + experience_diff + C(weight_class_grouped) + inactivity_diff + stance_mismatch`

## Key Findings
- Heavier divisions (notably Heavyweight and Light Heavyweight) show higher finish odds relative to the omitted reference class (`Bantamweight`).
- Most women's divisions show lower finish odds than the omitted reference class in pooled models.
- Reach difference has weak evidence once controls are included in richer specifications.
- Age difference and inactivity difference are positively associated with finish probability in the full model.
- Overall fit is modest (low pseudo R-squared), so results are interpreted as associations rather than strong prediction.

## Robustness Checks
The full specification is re-estimated on two restricted samples:
- Men-only sample
- Non-5-round fights sample

Results are directionally similar for major patterns (for example heavier divisions and positive inactivity association), supporting baseline findings.

## Project Pipeline
The entire project can be reproduced with the following commands:

```bash
python src/scrape_events.py
python src/scrape_fights.py
python src/clean_data.py
python src/build_features.py
python src/model_finish.py
quarto render blog.qmd
```

Quarto is used for report rendering and should be installed on the system path (`quarto --version`).

## Outputs
Running the pipeline produces the following outputs:

- `data/raw/` - scraped fight and fighter datasets
- `data/interim/` - cleaned and merged datasets
- `data/processed/analytic_dataset.csv` - full processed analytical dataset
- `data/processed/analytic_dataset_2010plus.csv` - modern analytical dataset used for modelling
- `output/model_table.txt` - Model 1 to Model 3 regression output
- `output/robustness_results.txt` - robustness test results
- `output/figures/` - figures for model interpretation and descriptive summaries

## Repository Structure
- `src/` - Python scripts for scraping, cleaning, feature engineering, and modelling
- `data/raw/` - raw scraped datasets
- `data/interim/` - cleaned intermediate datasets
- `data/processed/` - analytical datasets for estimation
- `output/` - model tables, robustness tables, and figures

## Important Technical Note
On this machine, UFCStats is reachable only via HTTP and not HTTPS.

Working:
- `http://ufcstats.com`

Failing:
- `https://ufcstats.com`

The scraping utilities therefore prioritise HTTP connections when retrieving UFCStats pages.
