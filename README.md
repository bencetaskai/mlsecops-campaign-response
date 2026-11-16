# mlsecops-campaign-response

MLSecOps semester project (Óbuda University, MSc Data Science, 2026/27/1).

Predicts which customers will accept the next marketing campaign offer (`Response`, binary classification)
on the Kaggle *Customer Personality Analysis* dataset (CC0). The project is an end-to-end MLOps system:

- data versioning with DVC
- experiment tracking and model registry with MLflow, using a champion/challenger quality gate
- a FastAPI REST API with `train` and `predict` endpoints
- drift monitoring with Evidently and Streamlit
- scheduled retraining with Airflow
- the whole system starts with a single `docker compose up`

> Work in progress. Full documentation will follow in this README.

## Known issues

- **Merge commit of PR #1 (`6770d78`) breaks the commit format.** GitHub's merge button re-wrapped the message
  body at 72 characters, splitting the `References:` and `Signed-off-by:` lines. The `commit-messages` CI job
  caught it on the push to `main`. `main` is protected against force pushes and was not rewritten. Since then,
  PRs are merged with `gh pr merge --merge --body-file`, using a message file checked by
  `tools/check_commit_msg.py` beforehand; the PR template describes the procedure.
