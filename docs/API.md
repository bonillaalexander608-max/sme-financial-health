# API contract

Start the server from the repository root with `Rscript scripts/run_api.R`.
The default binds to localhost port 8000. `HOST` and `PORT` override these values.
No external hosting has been configured.

- `GET /health`: reports model readiness and class order.
- `GET /schema`: returns all 37 required raw input fields and supported categories.
- `POST /predict`: JSON object with exactly one top-level key, `records`, containing 1–100 business objects.

All 37 field names must be present. Missing numeric/categorical observations may be `null`, except country, which must identify one of the four training countries. Numeric values must be finite and nonnegative. ID and Target are not accepted. Category spellings must match `/schema`. Use the questionnaire's original local-currency units and reporting periods; this API does not perform currency conversion or infer annual vs monthly expenses.

A successful response includes `model_id` and a `predictions` array. Each element gives its input row number, class and three probabilities. Classes use the saved model's Low, Medium, High ordering. Class selection is the first maximum probability; no fitted decision thresholds or probability calibration are added.

Invalid input returns HTTP 400; bodies over 1 MB return 413; internal model failures return a generic 500. For a publicly hosted deployment, configure transport security, authentication, rate and request-size limits at the gateway. The code is a local portfolio service and does not configure these external infrastructure services.

## Artifacts

`models/preprocessing.rds` contains the actual trained recipe, encoded feature names and class order. `models/booster.ubj` contains the native XGBoost model exported from `final_fit_total`. They are a matched pair. Neither incoming requests nor model startup re-estimate the medians or other learned recipe steps. The exported bundle avoids distributing the entire interactive session.

`tests/smoke.R` checks single/batch consistency, class probability totals, missing numeric values and rejected inputs. The stored `reports/tables/api_verification.json` describes any full-data equivalence check performed when exporting the supplied model.

The original prediction path uses sparse matrices, whose implicit zeros are treated as missing by XGBoost. Native inference preserves that behavior with `missing=0`; switching to a dense matrix with only NA as missing changes predictions substantially. This convention is checked against the stored validation probabilities and submission labels.

Tested export runtime: R 4.3.3, XGBoost 3.2.1.1, recipes 1.0.9 and Plumber 1.2.1. The exact export session is recorded in `models/export_session.txt`. No model was retrained during export. API category validation covers questionnaire values present in the supplied train/test files; unlabeled test categories were used only for this input contract, not for fitting transformations or selecting the model.
