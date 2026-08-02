"""Serve a city's promoted champion over HTTP.

    python scripts/serve.py [--city <name>] [--host H] [--port P] [--reload]

The city names come from config.CITY_CLI_NAMES; --help lists the current set.
Whatever holds the `champion` alias in that city's registry is what answers, so
run scripts/run_openmeteo.py first if the registry is empty.

    GET  /health   is the process up, and does it have a model
    GET  /model    which version is answering, and how stale it is
    POST /predict  a batch of forecast hours -> PM2.5
    POST /reload   re-read the alias after a promotion

Interactive docs at /docs. In a container this is run as
`uvicorn driftloop.serving:app` with DRIFTLOOP_PROFILE set instead.
"""

from __future__ import annotations

import argparse
import os

import uvicorn

from driftloop.config import CITY_CLI_NAMES, PROFILES
from driftloop.serving import PROFILE_ENV_VAR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", choices=[*CITY_CLI_NAMES], default="krakow")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--reload", action="store_true", help="restart on code changes (development)"
    )
    args = parser.parse_args()

    profile_key = CITY_CLI_NAMES[args.city]
    profile = PROFILES[profile_key]

    # uvicorn's --reload re-imports the app in a child process, so the choice of
    # profile has to survive as environment rather than as a closure.
    os.environ[PROFILE_ENV_VAR] = profile_key

    print(f"serving {profile.label} · model {profile.loop.registered_model_name}")
    print(f"docs on http://{args.host}:{args.port}/docs")
    uvicorn.run(
        "driftloop.serving:app", host=args.host, port=args.port, reload=args.reload
    )


if __name__ == "__main__":
    main()
