# Predicting Player Churn and Break Duration in Pokémon GO Using Behavioural Telemetry

Supporting code for my MSc Data Science dissertation (ECS750P MSc Project, Queen Mary University of London), supervised by Laurissa Tokarchuk.

This project investigates two aspects of player inactivity in Pokémon GO:

1. **Churn prediction** — can behavioural data predict long-term player inactivity?
2. **Break duration prediction** — can behavioural data predict how long a player's break lasts?

## Data

The data were voluntarily donated by 186 Pokémon GO players through the in-game **"Request My Data"** feature. Three files were used:

| File | Records | Players | Content |
|---|---|---|---|
| `App_Sessions.csv` | 714,857 | 186 | App session events |
| `InAppPurchases.csv` | 326,925 | 181 | In-game transactions |
| `Pokestop_spin.csv` | 3,710,550 | 180 | PokéStop spin events |

Pseudonymised player identifiers were used throughout. Device identifiers and precise GPS coordinates were not used for feature construction.

**The raw data is not included in this repository** to protect player privacy, and identifier columns have been removed from all notebook outputs.

## Method

### Features

More than 4.7 million event-level records were aggregated into a player-level feature matrix (186 players × 11 features):

- **Session:** `playCount`, `activeDuration`, `avg_interval`, `break_count` (gaps > 7 days), `max_gap`, `recency`
- **Purchase:** `total_transactions`, `unique_items`, `total_coins_spent`
- **PokéStop spin:** `spin_count`, `spin_per_day`

Players with no purchase (n = 5) or spin records (n = 6) were kept, with those features set to 0.

### Churn label

A player is labelled as churned if their last session was more than **60 days** before the latest session in the dataset. Thresholds of 7, 14, 30 and 60 days were compared:

| Threshold | Churn rate | RF AUC |
|---|---|---|
| 7 days | 88.2% | 0.575 |
| 14 days | 69.4% | 0.773 |
| 30 days | 68.3% | 0.805 |
| **60 days** | **42.5%** | **0.870** |

The 60-day threshold gave the highest AUC and the most balanced classes. No player had a recency between 60 and 90 days, so the threshold falls in a natural gap in the data.

`recency` was used only to define the label and was excluded from the predictors to avoid target leakage.

### Models

- **Churn:** Random Forest, Logistic Regression, Gradient Boosting — stratified 5-fold CV, evaluated with ROC-AUC
- **Break duration:** Random Forest Regressor, Linear Regression — 5-fold CV, evaluated with MAE

Ensemble models use 100 estimators; all models use `random_state = 42`. Feature scaling for the linear models is fitted inside each CV fold to avoid leakage.

## Results

### Churn prediction (AUC)

| Feature set | Random Forest | Logistic Regression | Gradient Boosting |
|---|---|---|---|
| Session only | 0.870 | 0.669 | 0.862 |
| Session + Spin | 0.872 | 0.670 | 0.873 |
| Session + Purchase | 0.903 | 0.651 | 0.903 |
| Session + Purchase + Spin | **0.913** | 0.667 | 0.904 |

Best model (Random Forest, all features): accuracy 0.84; churned class precision 0.80, recall 0.84, F1 0.81.

`activeDuration` was the most important feature (36.4%). Purchase features added the most predictive value on top of session features; PokéStop spin features gave a smaller additional improvement.

### Break duration prediction (MAE, days)

1,611 breaks longer than 7 days were identified (mean 46.9 days, median 14.6 days, max 1,117 days).

| Features | Random Forest | Linear Regression |
|---|---|---|
| All features | 44.0 | 46.2 |
| Without `max_gap` & `break_count` | 47.1 | 47.8 |

These results are **exploratory** — see Limitations.

## Limitations

- Small sample (186 players), limiting generalisability.
- Voluntary data donation may introduce selection bias.
- External factors such as game updates, seasonal events and social influence were not included.
- Breaks are defined with a fixed 7-day gap; other definitions could change the results.
- Break duration features were calculated from each player's full history rather than only data before each break, creating a risk of temporal leakage (especially for `max_gap` and `break_count`).

## How to run

The notebook was written for Google Colab.

1. Place the three CSV files in your Google Drive under `MyDrive/`.
2. Open the notebook in Colab and run all cells.

Requirements: `pandas`, `numpy`, `scikit-learn`, `matplotlib`
