#!/bin/bash
# Runs once, on the first start with an empty data volume; later starts skip /docker-entrypoint-initdb.d.
# To re-run it after changing it: docker compose down, docker volume rm campaign-response_postgres-data.
#
# One database per service, each owned by its own login role, and PUBLIC may not connect: a leaked API password
# then reaches the prediction log only, not the MLflow registry or Airflow.
# No `set -u`: the entrypoint sources this file instead of running it when the executable bit is missing.
set -eo pipefail

create_database() {
    local name=$1
    local password
    password=$(cat "/run/secrets/${name}_db_password")
    # psql variables (:"name", :'password') quote the values, so a password with quotes cannot break the SQL.
    psql --set=ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres \
        --set=name="$name" --set=password="$password" <<'SQL'
CREATE ROLE :"name" LOGIN PASSWORD :'password';
CREATE DATABASE :"name" OWNER :"name";
REVOKE CONNECT, TEMPORARY ON DATABASE :"name" FROM PUBLIC;
SQL
}

for name in mlflow airflow monitoring; do
    create_database "$name"
done
