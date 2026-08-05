# Handoff

Last updated 2026-08-05.

## Where the project is

Six cities run the full loop, the static site and Streamlit dashboard both
publish, and the promoted champion is served over HTTP. `README.md` is the real
documentation; this file only carries what a reader of the code could not work
out for themselves.

## What landed on 2026-08-05 (second pass): the trigger fix, and its result

The top open item was "build the trigger fix", which had been documented and not
implemented. It is now built, measured, and switched off — and the measurement is
the useful part.

- **`LoopConfig.skill_floor`** adds a second retrain trigger: skill against the
  30-day daily profile dropping below a floor. Scale-free, so one number works
  for every city, and nothing about promoting a model can move it. Strictly an
  additional way to fire, never a way to suppress the ratio. `retrain_reason` is
  tagged per run (`ratio` / `skill` / `both` / `none`).
- **It does not pay.** [`scripts/sweep_skill_floor.py`](scripts/sweep_skill_floor.py)
  replays all six cities at several floors — a real replay per arm, because a
  changed trigger changes which models exist. At `-0.5` the outcome is identical
  in all six. At `0.0` it fires 2-3× as often and is worse in two cities (Delhi
  +10% error, Kraków +7%), better in one (Los Angeles), unchanged in three.
  Default is `None`.
- **The absolute floor, the other proposed fix, cannot be built.** Waking Los
  Angeles needs a floor below 18 µg/m³; at 15 Delhi fires on every run. The gap
  is empty, so it is a per-city knob in disguise and the cities stop being
  comparable. That is now argued from the measurement rather than asserted.
- **What it points at instead:** firing more often only feeds more challengers to
  a gate that certifies for about five weeks. The next experiment is the length
  of the exam (`holdout_days`), not the sensitivity of the trigger.
- `DataSource` now declares `forecast_lead_days`, because the skill baseline's
  causality rule depends on it and the loop should ask the source rather than be
  handed a config that might disagree with the data.

## What landed on 2026-08-05

The Streamlit app had fallen behind the published site, and closing that gap
surfaced one accounting bug in numbers both UIs publish.

- **A first run that promotes was credited to the wrong model.**
  `bootstrap_champion` registers v1 before any monitoring cycle exists, so a
  first run that promotes has no earlier row to read the outgoing version off,
  and the run's own tag has already been overwritten with the winner. Kraków and
  Los Angeles both promote on their first run. `retrospect.build` now takes the
  outgoing version from the registry there (the lowest registered version), which
  moved Kraków's paired figure from +6.4% over 48 weeks to +6.5% over 47, Los
  Angeles's from 0.0% over 37 to −2.8% over 36, and recovered one real promotion
  per city into the gate calibration (27 → 29, conclusions unchanged). Pinned by
  `test_a_first_run_that_promotes_was_served_by_the_bootstrap`.
- **The dashboard reached parity on the analysis.** It gained the paired
  retraining figure (it had been publishing only the unpaired one the evaluation
  doc says lies), the gate calibration, the physical-factors chart with the
  training band, and per-1-sd feature importance. `CITY_STORY` now covers all six
  cities. The sidebar states what the app is for, since it and the site are no
  longer answering the same question.
- **`GATE_LONG_WEEKS` lives in `retrospect.py`** and is published through
  `data.json`, because both UIs draw that split and a threshold kept by hand in
  Python and JavaScript is how they end up telling different stories.
- **`docs/methodology.md` was expanded** into something that explains the Ridge
  rather than naming it — the objective, the closed form, why standardisation is
  a correctness requirement, how the coefficients get back into real units and
  why that is what makes `retrospect` possible — plus the PSI saturation
  arithmetic, the skill score, and a reference list.

## What landed on 2026-08-04

The pages were reporting what the machine *did* rather than whether any of it
worked. Six counters and a raw RMSE cannot answer "is the model healthy" or "did
the loop earn its keep", and two of the numbers on the page were wrong in ways
that flattered the system.

- **`src/driftloop/retrospect.py`.** Scores every registered version on every
  monitoring window by rebuilding it from its coefficient tags, so no model has
  to be re-fitted or pickled. That unlocks per-model decay curves, a skill score
  against a causally-available climatology, and the promotion-gate calibration.
  Eleven tests in `tests/test_retrospect.py`.
- **Two findings the page now carries.** The retrain trigger ratchets: its bar
  is reset by every promotion and promotions happen at the seasonal peak, so
  after a while nothing can cross it (Kraków spends 30 of 48 weeks that way).
  And the seven-day promotion exam is well calibrated to about five weeks and
  reverses sign beyond twenty. Both are in `docs/evaluation.md`.
- **A pooled baseline** (`benchmark.fit_pooled`): one Ridge over all six cities
  with a per-city intercept, never retrained. It loses to per-city models in five
  of six, which is the evidence for the layout the project already had.
- **`site/compare.html`** puts all six cities on shared axes. `site/shared.css`
  is now shared by both pages; `tests/test_site_assets.py` guards the links,
  because a mistyped `href` after that split would serve both pages unstyled with
  every other test still green.

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
- **`champion_version` means two different things.** On a run that promotes, the
  loop monitors with the *outgoing* champion and then overwrites the tag with the
  winner. So the MLflow tag is the incoming version while `simulation_*.csv`
  records the one that did the monitoring. `retrospect` keeps both:
  `champion_version` (the tag, correct for "when did this model start") and
  `serving_version` (correct for "what was in service"). Mixing them up credits a
  new model with a window it never served, on data it trained on, and it is worth
  a free point of apparent improvement.
  **The first run is the awkward case**: there is no earlier row, because the
  bootstrap champion is registered before any monitoring cycle. Its outgoing
  version comes from the registry instead, as `min(models)`. Anything else that
  wants "what was serving" needs the same two-part answer.
- **Do not compare a logged float against a reconstructed one for equality.**
  The coefficient tags carry six decimals, so the same model scored both ways
  agrees only to about 1e-4. An `==` test there silently marked every window as
  retrained and moved Johannesburg's paired result from +14.9% to zero.
- **Rendering the site needs headless Edge**, not the MCP browser, which cannot
  screenshot a page it is not compositing:
  `msedge --headless=new --disable-gpu --hide-scrollbars --window-size=1240,9300 --virtual-time-budget=13000 --screenshot=out.png "http://localhost:8123/index.html?theme=light"`
  `?theme=light` exists for this: it pins the palette so a capture does not
  depend on the runner's dark-mode setting. Section crops in `docs/images/`
  are cut from one full-height capture using offsets read off the live DOM at the
  same 1240px width.
- **Plotly falls back to a hard-coded 700px** when it cannot measure a container
  during `newPlot`, which overflows the card and scrolls the page sideways. The
  fix is `Plots.resize` after plotting. Writing a measured `layout.width` instead
  is ~20% quicker and wrong: an explicit width opts the chart out of
  `responsive`, so every plot then keeps a stale size through a viewport change.
- **The globe does not animate, on purpose.** Re-projecting it costs 25-120ms a
  frame whichever API drives it, so an eased spin ran at about 11fps and read as
  a stutter. It cuts to the new centre and stays draggable.

## Open

- `/reload` is called by hand. The loop promotes and nothing tells serving.
- The lead-time sweep (error against 1-7 day lead) is still not run.
- **The weekly Action has not been observed running.** The scheduled profile
  holds two cycles, timestamped 09:14 and 09:50 UTC against a 06:00 cron, and the
  repository contains no commits from the workflow's bot identity. That is
  consistent with both having been run by hand. Whether the Action has ever
  fired cannot be established from the repository, so the site claims only the
  count and the dates. Worth checking in the Actions tab.
- The retrain trigger is shipped with the ratchet intact. Not for want of a fix
  any more: the fix exists, is tested, and measures out at roughly zero. See the
  2026-08-05 second-pass notes above.
- **Sweep `holdout_days`.** This is now the top open experiment, and the trigger
  work is what promoted it: if the exam's shelf life is the real constraint, a
  longer holdout should move the gate calibration and the delivered margins.
  `sweep_skill_floor.py` is the pattern to copy — a full replay per arm, into a
  temp backend, scored with `retrospect`.
- `challenger_train_days` is still argued rather than swept.
- **`docs/images/dashboard.png` is the one screenshot still showing the old
  Streamlit app.** `gate.png` and `compare.png` were regenerated on 2026-08-05
  and the rest were unaffected. The headless-Edge recipe below does not work on
  Streamlit: it serves a skeleton and fills it over a websocket, which
  `--virtual-time-budget` does not wait for, so every capture comes back as the
  ~15 KB grey placeholder. It needs a real browser window, or a driver that can
  wait on a selector.
- The wireframes in `docs/wireframes/` describe the original page layout, and the
  compare page was never drawn.
- **The training band on "what changed in the world" is hourly and the line is a
  two-week mean**, on both UIs. That is fine for temperature, where the seasonal
  swing dwarfs the diurnal one, and near-useless for shortwave radiation, where
  the band spans night to noon and no mean could ever leave it. Both captions now
  say so. The fix is a band of training-window *means* over the same window
  length, and it has to land on both UIs at once or they start disagreeing.
