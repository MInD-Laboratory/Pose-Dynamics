# pose-dynamics — Tier 0 container.
# Starts JupyterLab and opens straight to the quickstart notebook, so a user with
# no Python installed can run the pipeline in a browser. User data mounts at /work/data.
FROM python:3.12-slim

# A C/C++ compiler is needed to build the rqa-analysis recurrence core.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

# 1) Pinned dependencies first (cached layer; reproducible builds).
COPY docker/requirements-lock.txt /tmp/requirements-lock.txt
RUN pip install --no-cache-dir -r /tmp/requirements-lock.txt

# 2) The package itself (deps already pinned above).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

# 3) The documents a user runs / reads.
COPY notebooks ./notebooks
COPY docs ./docs
COPY examples ./examples
COPY configs ./configs
RUN mkdir -p /work/data

# Config: land on the quickstart, no login token (local use). More robust than CLI
# flags, which the Lab extension overrides.
COPY docker/jupyter_server_config.py /root/.jupyter/jupyter_server_config.py

EXPOSE 8888

CMD ["jupyter", "lab"]
