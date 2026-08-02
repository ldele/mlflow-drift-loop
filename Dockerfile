# Serve one city's promoted champion.
#
#   docker build -t drift-serve .                    # Kraków
#   docker build -t drift-serve --build-arg CITY=delhi .
#   docker run -p 8000:8000 drift-serve              # -> http://localhost:8000/docs
#
# The registry is rebuilt inside the image rather than copied in, and that is
# forced rather than chosen: MLflow stores artifact locations as absolute URIs,
# so a backend generated on a developer's Windows drive resolves to a path that
# does not exist in the container. Replaying the loop here regenerates the same
# history against paths that do. It reads the committed data_cache parquet and
# makes no network calls, so the image is reproducible and builds offline.

FROM python:3.12-slim

WORKDIR /app

# Dependencies first, so editing source does not re-resolve the environment.
# Installed editable on purpose: tracking.REPO_ROOT is derived from the package
# file's location, so a non-editable install would put the MLflow backend
# somewhere inside site-packages. Editable keeps it at /app, next to the cache.
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir -e .

COPY scripts/ scripts/
COPY data_cache/ data_cache/

# One city per image. The same name selects the training run and the profile
# served, so they cannot disagree.
ARG CITY=krakow
ENV DRIFTLOOP_PROFILE=${CITY}
RUN python scripts/run_openmeteo.py --fresh --city ${CITY}

EXPOSE 8000

# No curl in the slim image, so probe with the interpreter that is already here.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys,json; \
b=json.load(urllib.request.urlopen('http://localhost:8000/health')); \
sys.exit(0 if b['model_loaded'] else 1)"

CMD ["uvicorn", "driftloop.serving:app", "--host", "0.0.0.0", "--port", "8000"]
