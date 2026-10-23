# Model card

## Task and provenance
Predict the survey-derived Financial Health Index (Low, Medium, High) for SMEs in four Southern African countries. Author: Yader Alexander Ramírez Bonilla. Original experiments include multinomial regression, neural networks, XGBoost and blending. The archived scripts are preserved, including their dependencies on execution order and interactive objects; they are not advertised as a single clean pipeline.

## Serving candidate
The API candidate is `final_fit_total` from the supplied workspace, matching the no-ID XGBoost script. This is not proof that it is the best of every historical experiment. `country` remains a predictor. ID is excluded. The trained recipe, including its learned medians, rare-category mapping, dummy encoding and correlation filtering, must be exported with the booster; preprocessing must never be refitted on API requests.

## Feature engineering
Before the monetary logarithms: informal finance severity, owner age group, expenses/(turnover+1), personal income/(expenses+1), turnover/max(business age years,0.25), formal finance indicator, customer credit/cash-flow indicator and record discipline. Original raw fields mostly remain; the two informal-finance fields are removed after their summary is created. Log(1+x) is applied to income, expenses and turnover. Median imputation, rare categories at 2.5%, dummy variables, zero-variance removal and correlation filter at 0.8 follow the saved recipe.

The variable dictionary describes turnover as annual and expenses as monthly or annual in local currency; income is monthly. These engineered ratios reproduce the experiment but cannot automatically be interpreted as harmonized financial ratios across countries or reporting periods.

## Validation
Seed 2026; stratified random 80/20 labeled split: 7,694 training / 1,924 validation. Saved resampling has five folds, not seven despite historical output filenames. Original tuning considered 75 settings and used probability metrics. The competition specifies F1; the original code did not optimize F1 explicitly. Macro F1 here is explicitly an unweighted class average, without claiming the leaderboard's averaging convention.

Stored validation predictions were recovered and rescored. Hyperparameters: 989 trees, depth 6, learning rate 0.0106, min_n 2, loss reduction 0.000235, sample proportion 0.976, stop_iter 5. No new tuning was run. The total-data deployment fit sees all 9,618 labels; validation scores belong to the earlier training-only fit. Historical repeated experimentation may make these validation scores optimistic. No confidence intervals or leaderboard score are inferred from label-only submissions.

## Scope
Probabilities are model outputs, not verified calibrated confidence levels. High health is rare (470 training labels), and recall is lower than for Low. Classification is descriptive, not a causal explanation or a lending decision system. Nicaragua and other unseen countries have not been validated. Monetary units and question wording must match the training source. The API rejects unsupported countries. No public hosted service or uptime is implied by this repository.
