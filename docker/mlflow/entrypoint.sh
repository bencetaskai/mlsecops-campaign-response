#!/bin/sh
# Reads the Docker secrets and starts the server. Credentials go into environment variables of this process only,
# so they appear neither in `docker inspect` nor in the process arguments.
set -eu

# The driver is named: SQLAlchemy 2.1 maps plain postgresql:// to psycopg 3, which the image does not have.
MLFLOW_BACKEND_STORE_URI="postgresql+psycopg2://mlflow:$(cat /run/secrets/mlflow_db_password)@postgres:5432/mlflow"
AWS_SECRET_ACCESS_KEY=$(cat /run/secrets/mlflow_s3_secret_key)
export MLFLOW_BACKEND_STORE_URI AWS_SECRET_ACCESS_KEY

exec mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --workers "${MLFLOW_WORKERS:-2}" \
    --serve-artifacts \
    --artifacts-destination s3://mlflow \
    --allowed-hosts "${MLFLOW_ALLOWED_HOSTS}"
