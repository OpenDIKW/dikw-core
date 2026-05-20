FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/OpenDIKW/dikw-core"
LABEL org.opencontainers.image.description="DIKW knowledge engine — server"
LABEL org.opencontainers.image.licenses="MIT"

# Pull the latest Debian security patches into the base layer so Trivy
# doesn't flag the python:3.12-slim snapshot. Combined with --no-install-recommends
# the apt cache is wiped in the same layer to keep the image small.
RUN apt-get update \
    && apt-get -y --no-install-recommends upgrade \
    && rm -rf /var/lib/apt/lists/*

# Bump pip ahead of the install step — the python:3.12-slim base ships
# an older pip with known CVEs (CVE-2025-8869, CVE-2026-6357).
RUN pip install --no-cache-dir --upgrade "pip>=26.1"

# Tracks the latest version already published to PyPI — bump only AFTER
# the release tag pushes a new wheel (the trusted-publishing workflow in
# .github/workflows/release.yml). Bumping it before publish wedges the
# Trivy scan job, which builds this image from scratch on every PR and
# would `pip install` a version that doesn't exist yet.
ARG DIKW_VERSION=0.2.0
RUN pip install --no-cache-dir "dikw-core[postgres]==${DIKW_VERSION}"

RUN useradd --create-home --uid 1000 dikw
USER dikw
WORKDIR /base
VOLUME ["/base"]

EXPOSE 8765

ENTRYPOINT ["dikw"]
CMD ["serve", "--base", "/base", "--host", "0.0.0.0", "--port", "8765"]
