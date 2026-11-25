# syntax=docker/dockerfile:1
# Shared image for api, model-bootstrap and streamlit. Build from the repo root:
#   docker build -f docker/app.Dockerfile -t campaign-response-app .

# Pinned by digest; the tag is only for readability.
FROM python:3.12.16-slim-trixie@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

# The app user has no home directory, so matplotlib's font cache goes to /tmp instead of a warning on every start.
# MLflow writes plain text beside the JSON logs: a "View run" URL at the end of every run (one per /predict request),
# a progress bar for every artifact download (the champion model) and a hint for coding agents. All three are off.
# DO_NOT_TRACK switches off Evidently's usage telemetry (only its UI service sends any; set anyway).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    MPLCONFIGDIR=/tmp/matplotlib \
    MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT=true \
    MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR=false \
    MLFLOW_DISABLE_AGENT_HINT=true \
    DO_NOT_TRACK=1

WORKDIR /app

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

COPY requirements.txt .
# The pip cache lives in a BuildKit cache mount, not in a layer: the image stays as small as with --no-cache-dir,
# but after a requirements change only new packages are downloaded.
RUN --mount=type=cache,target=/root/.cache/pip pip install --requirement requirements.txt

# Owned by root: the app user can read the code but not modify it. params.yaml travels with the code it was
# tested with; the raw data does not (DVC versions it, compose mounts it read-only).
COPY params.yaml .
COPY src/ src/

ARG GIT_SHA=unknown
# Also as an environment variable: training runs read it for their git_sha lineage tag.
ENV GIT_SHA=${GIT_SHA}
LABEL org.opencontainers.image.title="campaign-response" \
      org.opencontainers.image.source="https://github.com/bencetaskai/mlsecops-campaign-response" \
      org.opencontainers.image.revision="${GIT_SHA}"

USER app

CMD ["python", "-m", "campaign_response"]
