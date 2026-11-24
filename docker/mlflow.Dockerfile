# syntax=docker/dockerfile:1
# MLflow tracking server and model registry: Postgres backend store, artifacts proxied to MinIO. Build from the
# repo root: docker build -f docker/mlflow.Dockerfile -t campaign-response-mlflow .

# Pinned by digest; the tag is only for readability.
FROM ghcr.io/mlflow/mlflow:v3.18.0@sha256:257204c77980b0f3450a26eb6089e487603fcdd4509529eadaf88f90669f5c13

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY docker/mlflow/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --no-deps --requirement /tmp/requirements.txt \
    && pip check \
    && rm /tmp/requirements.txt

# The upstream image runs as root; the server needs no write access outside /tmp.
RUN groupadd --system --gid 10002 mlflow \
    && useradd --system --uid 10002 --gid mlflow --no-create-home --shell /usr/sbin/nologin mlflow

COPY --chmod=0555 docker/mlflow/entrypoint.sh /usr/local/bin/mlflow-entrypoint.sh

USER mlflow
WORKDIR /tmp
EXPOSE 5000

ENTRYPOINT ["/usr/local/bin/mlflow-entrypoint.sh"]
