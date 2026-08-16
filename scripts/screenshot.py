"""Capture the figures in docs/images, including the ones Edge cannot.

    # the published site, which fills itself from data.json
    python scripts/build_site.py
    python -m http.server 8123 --directory site &
    python scripts/screenshot.py http://localhost:8123/compare.html?theme=light \
        docs/images/compare.png --wait-for ".js-plotly-plot" --trim

    # the Streamlit app, which fills itself over a websocket
    streamlit run dashboard/app.py --server.headless true &
    python scripts/screenshot.py http://localhost:8501/ docs/images/dashboard.png \
        --wait-for '[data-testid="stMetric"]' --size 1240x1800

    # one section of the site, cropped by selector rather than by pixel offset
    python scripts/screenshot.py http://localhost:8123/index.html?theme=light \
        docs/images/method.png --size 1240x9900 --clip "section#method"

`--clip` measures the element inside the page being captured, which is the only
place the answer is right. A viewport tall enough to hold the whole document
lays out differently from a scrolling one, so an offset read in one browser and
applied to a capture in another lands somewhere else. Cropping by selector also
survives the section growing, which is what makes a figure safe to regenerate
after an edit rather than something to re-measure by hand.

This drives Edge over the DevTools protocol and polls until a selector matches,
rather than shelling out to `msedge --headless --screenshot`, which has two
failure modes that both save a plausible-looking file instead of erroring.

It fires on the load event, which suits a page whose charts are drawn from JSON
fetched during load and not Streamlit, which serves a skeleton and paints into
it over a websocket afterwards; the capture comes back empty.

And a second invocation attaches to an Edge already running rather than starting
a headless one, writing nothing while exiting zero.

A capture here that never sees its selector fails loudly.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import subprocess
import time
from pathlib import Path

import requests
import websockets

EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
DEBUG_PORT = 9333


def trim_to_content(path: Path) -> None:
    """Crop trailing background so a tall capture is not mostly empty.

    The page height is not known before rendering, so a capture asks for more
    room than it needs. The background colour is read from the bottom-right
    pixel rather than assumed, because the palette differs between themes.
    """
    from PIL import Image

    image = Image.open(path).convert("RGB")
    width, height = image.size
    background = image.getpixel((width - 2, height - 2))
    pixels = image.load()
    last = next(
        (y for y in range(height - 1, -1, -1)
         if any(pixels[x, y] != background for x in range(0, width, 4))),
        height - 1,
    )
    image.crop((0, 0, width, min(height, last + 32))).save(path)


async def capture(url: str, out: Path, selector: str, width: int, height: int,
                  settle: float, clip: str | None = None) -> None:
    profile = Path(f"{Path.home()}/.cache/driftloop-shot-{DEBUG_PORT}")
    browser = subprocess.Popen(
        [str(EDGE), "--headless=new", "--disable-gpu", "--hide-scrollbars",
         f"--remote-debugging-port={DEBUG_PORT}", f"--user-data-dir={profile}",
         f"--window-size={width},{height}", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        target = None
        for _ in range(40):
            try:
                tabs = requests.get(f"http://localhost:{DEBUG_PORT}/json", timeout=2).json()
                target = next((t for t in tabs if t["type"] == "page"), None)
                if target:
                    break
            except requests.RequestException:
                pass
            time.sleep(0.5)
        if target is None:
            raise SystemExit("could not reach the DevTools endpoint; is another Edge running?")

        async with websockets.connect(target["webSocketDebuggerUrl"],
                                      max_size=200 * 1024 * 1024) as socket:
            sent = 0

            async def call(method: str, params: dict | None = None) -> dict:
                nonlocal sent
                sent += 1
                mine = sent
                await socket.send(json.dumps({"id": mine, "method": method,
                                              "params": params or {}}))
                while True:
                    message = json.loads(await socket.recv())
                    if message.get("id") == mine:
                        return message.get("result", {})

            await call("Page.enable")
            await call("Emulation.setDeviceMetricsOverride", {
                "width": width, "height": height, "deviceScaleFactor": 1, "mobile": False,
            })
            await call("Page.navigate", {"url": url})

            # Polled rather than waited on a lifecycle event, because the content
            # worth capturing arrives after every lifecycle event has fired.
            matched = 0
            for second in range(90):
                await asyncio.sleep(1)
                result = await call("Runtime.evaluate", {
                    "expression": f"document.querySelectorAll({selector!r}).length",
                    "returnByValue": True,
                })
                matched = result.get("result", {}).get("value", 0) or 0
                if matched:
                    print(f"  {selector} matched {matched} node(s) after {second + 1}s")
                    break
            if not matched:
                raise SystemExit(f"{selector} never matched; nothing captured")

            # Charts keep drawing for a moment after their container exists.
            await asyncio.sleep(settle)

            params: dict = {"format": "png"}
            if clip:
                # Measured inside the page being captured, which is the only
                # place the answer is right. A viewport tall enough to hold the
                # whole document lays out differently from a scrolling one, so
                # offsets read in a browser at one height and applied to a
                # capture at another land somewhere else entirely.
                box = await call("Runtime.evaluate", {
                    "expression": (
                        "(() => { const e = document.querySelector(%r);"
                        " if (!e) return null; const r = e.getBoundingClientRect();"
                        " return {x: r.left + scrollX, y: r.top + scrollY,"
                        " width: r.width, height: r.height}; })()" % clip
                    ),
                    "returnByValue": True,
                })
                rect = box.get("result", {}).get("value")
                if not rect:
                    raise SystemExit(f"--clip {clip!r} matched nothing")
                params["clip"] = {**{k: float(v) for k, v in rect.items()}, "scale": 1}
                print(f"  clipped to {clip} "
                      f"({rect['width']:.0f}x{rect['height']:.0f} at y={rect['y']:.0f})")

            shot = await call("Page.captureScreenshot", params)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(base64.b64decode(shot["data"]))
            print(f"  wrote {out} ({out.stat().st_size} bytes)")
    finally:
        browser.terminate()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("out", type=Path)
    parser.add_argument("--wait-for", default=".js-plotly-plot",
                        help="CSS selector that has to match before capturing")
    parser.add_argument("--size", default="1240x9300", help="viewport, WxH")
    parser.add_argument("--settle", type=float, default=6.0,
                        help="seconds to wait after the selector matches")
    parser.add_argument("--trim", action="store_true",
                        help="crop trailing background after capturing")
    parser.add_argument("--clip", default=None,
                        help="CSS selector to crop to, measured inside the captured page")
    args = parser.parse_args()

    width, height = (int(n) for n in args.size.lower().split("x"))
    asyncio.run(capture(args.url, args.out, args.wait_for, width, height,
                        args.settle, args.clip))
    if args.trim:
        trim_to_content(args.out)
        print(f"  trimmed to {args.out.stat().st_size} bytes")


if __name__ == "__main__":
    main()
