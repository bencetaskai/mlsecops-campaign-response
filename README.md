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
