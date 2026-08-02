# Handoff

Last updated 2026-08-02.

## Where the project is

Six cities run the full loop, the static site and Streamlit dashboard both
publish, the weekly Action keeps the live profile advancing, and the promoted
champion is now served over HTTP. `README.md` is the real documentation; this
file only carries what a reader of the code could not work out for themselves.

## What landed on 2026-08-02

- **Serving** (`src/driftloop/serving.py`, `scripts/serve.py`, `Dockerfile`).
  FastAPI reading the `champion` alias, with `/health`, `/model`, `/predict`
  and `/reload`. Ten tests in `tests/test_serving.py`.
- **A CI gate** (`.github/workflows/ci.yml`). Nothing ran the tests before
  this; the two existing workflows only deployed.
- **A latent bug fixed.** `tracking.load_champion` returned `version` as an int
  while the dataclass declared `str` and `log_and_register` coerced. It only
  surfaced once serving compared the two.

## Things that will bite you

- **The venv must come from the python.org interpreter.** The uv-managed one
  has broken SSL on this machine. Open-Meteo calls go through
  `truststore.inject_into_ssl()` (already in `openmeteo.py`) for the corporate
  TLS proxy. CI on Ubuntu needs neither.
- **MLflow artifact locations are absolute URIs.** A backend generated here
  will not resolve in a container or on another machine, which is why the
  Dockerfile replays the loop from the committed parquet cache at build time
  rather than copying `mlflow_openmeteo.db` in. Do not "simplify" that away.
- **`tracking.REPO_ROOT` is derived from the package file's location**
  (`parents[2]`), so a non-editable install puts the MLflow backend inside
  site-packages. The Dockerfile installs with `-e` for that reason. It is also
  the hook the serving tests use: they monkeypatch it to a `tmp_path` to
  relocate the whole registry.
- **Rendering the site needs headless Edge**, not the MCP browser, which times
  out on live Plotly pages:
  `msedge --headless=new --disable-gpu --window-size=1240,1560 --screenshot=out.png <url>`

## Open

- `/reload` is called by hand. The loop promotes and nothing tells serving.
  Closing that is the next obvious piece, and is what the README's remaining
  serving limitation describes.
- The lead-time sweep (error against 1–7 day lead) is still not run.
