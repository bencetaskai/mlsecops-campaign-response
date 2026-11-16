#!/bin/sh
# One-shot init: create the buckets, then exit. Safe to re-run (existing buckets are kept).
set -eu

mc alias set local http://minio:9000 \
    "$(cat /run/secrets/minio_root_user)" "$(cat /run/secrets/minio_root_password)" >/dev/null

for bucket in mlflow dvc; do
    mc mb --ignore-existing "local/${bucket}"
done

mc ls local
