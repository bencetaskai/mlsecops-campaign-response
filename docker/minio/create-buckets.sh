#!/bin/sh
# One-shot init: create the buckets and the MLflow user, then exit. Safe to re-run on every start.
set -eu

mc alias set local http://minio:9000 \
    "$(cat /run/secrets/minio_root_user)" "$(cat /run/secrets/minio_root_password)" >/dev/null

for bucket in mlflow dvc; do
    mc mb --ignore-existing "local/${bucket}"
done

# MLflow gets its own user limited to the mlflow bucket, so a compromised MLflow container cannot touch the DVC data.
# user add and policy create overwrite on re-run, so a changed secret or policy is applied on the next start.
mc admin user add local mlflow "$(cat /run/secrets/mlflow_s3_secret_key)"
mc admin policy create local mlflow-artifacts /scripts/mlflow-policy.json
# attach fails if the policy is already attached; the image has no grep, hence the case match.
case "$(mc admin policy entities local --user mlflow)" in
    *mlflow-artifacts*) ;;
    *) mc admin policy attach local mlflow-artifacts --user mlflow ;;
esac

mc ls local
