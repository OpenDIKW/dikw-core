FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/OpenDIKW/dikw-core"
LABEL org.opencontainers.image.description="DIKW knowledge engine — server"
LABEL org.opencontainers.image.licenses="MIT"

# Bump in lockstep with each release tag.
ARG DIKW_VERSION=0.0.2
RUN pip install --no-cache-dir "dikw-core[postgres]==${DIKW_VERSION}"

RUN useradd --create-home --uid 1000 dikw
USER dikw
WORKDIR /base
VOLUME ["/base"]

EXPOSE 8765

ENTRYPOINT ["dikw"]
CMD ["serve", "--base", "/base", "--host", "0.0.0.0", "--port", "8765"]
