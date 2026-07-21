Add Delhi + LA air-quality datasets
In the mlflow-drift-loop project (C:\Users\LDELEZ\Documents\Claude\Projects\ML & Medtech\mlflow-drift-loop), add two more real air-quality datasets — Delhi and Los Angeles — alongside the existing Kraków one, so the deployed dashboard's source switcher shows three contrasting cities. All three use the same Open-Meteo source (free, no API key) behind the get_data(start, end) contract.

Context / current architecture:

src/driftloop/config.py: OpenMeteoConfig currently holds a single location (Kraków, lat 50.0647, lon 19.9450, origin 2025-05-01, horizon 2026-02-01). PROFILES dict has keys "openmeteo" (Kraków), "synthetic", "scheduled". Each Profile has its own experiment_name, registered_model_name, db_filename, meta_filename.
scripts/run_openmeteo.py trains a champion on summer (CHAMPION_TRAIN_START/END = Jun–Aug 2025) then replays weekly into winter. Kraków is WINTER-bad (heating-season smog), so a summer-trained model decays into winter.
scripts/build_site.py has DISPLAY_ORDER = ["openmeteo", "scheduled"] controlling which profiles the page shows, and publish_raw_data() writes site/krakow_hourly.csv from the Kraków cache.
The data source caches to data_cache/*.parquet (now committed, not gitignored).
What to implement:

Parameterize per-city configs: give each city its own OpenMeteoConfig (Delhi lat 28.6139 lon 77.2090; Los Angeles lat 34.0522 lon -118.2437) and its own Profile (e.g. keys "openmeteo_delhi", "openmeteo_la") with distinct experiment_name / registered_model_name / db_filename / meta_filename. Consider refactoring run_openmeteo.py to take a --city arg (or loop over cities) rather than hardcoding Kraków.
Mind the seasonality per city so the drift story works: Kraków & Delhi are WINTER-bad (train on summer, decay into winter). Los Angeles is SUMMER-bad (ozone/wildfire PM2.5) — so for LA train the champion on a clean-season window and replay into summer, i.e. pick train/replay windows that actually produce a regime shift. Verify each city actually shows rising PSI + perf-drift + retrains before shipping (Delhi should be the most dramatic).
Add all three cities to build_site.DISPLAY_ORDER (Kraków, Delhi, LA, then scheduled) and generalize publish_raw_data() to emit a CSV per city (or keep just Kraków). Regenerate site/data.json.
Update pages.yml to build each city run (they rebuild from committed data_cache parquet — no re-fetch), and .github/workflows/drift-loop.yml if the live "scheduled" profile should also gain cities (optional — currently scheduled = Kraków only).
Update README "The data" section to describe the three cities and their contrasting regimes.
Verify visually: serve site/ and render with headless Edge (msedge --headless=new --disable-gpu --window-size=1240,1560 --screenshot=out.png <url>) — the MCP browser screenshot times out on live Plotly pages. Confirm the switcher shows the three cities and charts fill their cards.
Notes: venv must be built from the python.org interpreter (uv-managed one has broken SSL on this machine); Open-Meteo calls need truststore.inject_into_ssl() (already in openmeteo.py) for the corporate TLS proxy. Nothing in this repo is a git repo yet / nothing is pushed — deploying to ldele.github.io still requires git init + push (raise this with the user).