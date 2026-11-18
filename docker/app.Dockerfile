# syntax=docker/dockerfile:1
# Shared image for api, model-bootstrap and streamlit. Build from the repo root:
#   docker build -f docker/app.Dockerfile -t campaign-response-app .

# Pinned by digest; the tag is only for readability.
FROM python:3.12.16-slim-trixie@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

COPY requirements.txt .
# The pip cache lives in a BuildKit cache mount, not in a layer: the image stays as small as with --no-cache-dir,
# but after a requirements change only new packages are downloaded.
RUN --mount=type=cache,target=/root/.cache/pip pip install --requirement requirements.txt

# Owned by root: the app user can read the code but not modify it.
COPY src/ src/

ARG GIT_SHA=unknown
LABEL org.opencontainers.image.title="campaign-response" \
      org.opencontainers.image.source="https://github.com/bencetaskai/mlsecops-campaign-response" \
      org.opencontainers.image.revision="${GIT_SHA}"

USER app

CMD ["python", "-m", "campaign_response"]
