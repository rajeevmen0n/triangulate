"""Real-browser check of the web UI (headless Chromium via Playwright).

Run:  nix develop .#browser --command python -m pytest -m browser tests/browser -s
Uses the project's real data dir (warm stage cache) and the repo-root audio.mp3 fixture; the created job is
deleted at the end. Basemap is disabled and container metadata is never read. Screenshots go to
data/tmp/ui-check/.
"""

from __future__ import annotations

import math
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.browser
pw = pytest.importorskip("playwright.sync_api")

ROOT = Path(__file__).resolve().parents[2]
AUDIO = ROOT / "audio.mp3"
SHOTS = ROOT / "data" / "tmp" / "ui-check"
TRUTH = (40.876722, -73.526444)


def _haversine_m(a, b):
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dl = math.radians(b[1] - a[1])
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(h))


@pytest.fixture(scope="module")
def server():
    if not AUDIO.exists():
        pytest.skip("audio.mp3 fixture missing")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = dict(os.environ, TRIANGULATE_BASEMAP_ENABLED="0", TRIANGULATE_SKIP_METADATA="1")
    proc = subprocess.Popen([sys.executable, "-m", "triangulate", "serve", "--host", "127.0.0.1",
                             "--port", str(port)], env=env, cwd=ROOT)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            try:
                if httpx.get(base + "/api/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.2)
        else:
            raise RuntimeError("server did not start")
        yield base
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_full_flow(server):
    SHOTS.mkdir(parents=True, exist_ok=True)
    console: list[str] = []
    errors: list[str] = []
    failed: list[str] = []
    job_id = None
    with pw.sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.on("console", lambda m: console.append(f"{m.type}: {m.text}"))
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("requestfailed", lambda r: failed.append(f"{r.url} {r.failure}"))
        page.on("response", lambda r: failed.append(f"{r.status} {r.url}") if r.status >= 400 else None)
        page.on("request", lambda r: failed.append(f"EXTERNAL {r.url}")
                if not r.url.startswith((server, "data:", "blob:")) else None)
        try:
            page.goto(server + "/")
            page.wait_for_selector("form")
            page.wait_for_function("document.getElementById('use-notice').textContent.length > 30")
            page.set_input_files("#file", str(AUDIO))
            for k, v in zip("swne", ("40.50", "-74.00", "41.20", "-73.00")):
                page.fill(f"#bb-{k}", v)
            page.fill("#d0", "2026-09-08")
            page.fill("#tw0", "15:30")
            page.fill("#tw1", "18:30")
            page.fill("#tz", "America/New_York")
            assert not page.is_checked("#visual")
            page.screenshot(path=str(SHOTS / "1-form.png"), full_page=True)
            page.click("button[type=submit]")
            page.wait_for_url(re.compile(r"#/job/"))
            job_id = page.url.split("#/job/")[1]
            page.wait_for_selector("ol.stages")
            page.wait_for_function("document.querySelector('.bar.big i').style.width !== ''", timeout=60000)
            page.screenshot(path=str(SHOTS / "2-progress-start.png"), full_page=True)
            seen_mid = False
            deadline = time.time() + 600
            while time.time() < deadline:
                if page.query_selector(".results"):
                    break
                if "Could not show results" in page.inner_text("main") or (
                        page.query_selector("section.card .err") and "failed" in page.inner_text("main")):
                    break
                w = page.evaluate("document.querySelector('.bar.big i')?.style.width || ''")
                if w not in ("", "0%", "100%") and not seen_mid:
                    seen_mid = True
                    page.screenshot(path=str(SHOTS / "2-progress.png"), full_page=True)
                time.sleep(1)
            assert page.query_selector(".results"), "results view never appeared:\n" + page.inner_text("main")
            page.wait_for_selector(".spec img")
            page.wait_for_function("document.querySelector('.spec img').complete")
            page.wait_for_selector(".leaflet-image-layer")
            page.wait_for_timeout(1500)
            page.screenshot(path=str(SHOTS / "3-results.png"), full_page=True)

            # --- results assertions
            assert page.query_selector(".leaflet-image-layer"), "no heatmap overlay"
            assert page.evaluate("document.querySelector('.leaflet-image-layer').naturalWidth") > 0
            assert page.query_selector_all(".leaflet-overlay-pane path"), "no HPD contours"
            rows = page.query_selector_all("tr.cand")
            assert rows, "no candidates"
            top = page.inner_text("tr.cand >> nth=0")
            assert "%" in top
            m = re.search(r"(-?\d+\.\d+), (-?\d+\.\d+)", top)
            d = _haversine_m((float(m[1]), float(m[2])), TRUTH)
            print(f"top candidate {m[0]} is {d:.0f} m from truth")
            assert d < 500
            assert page.query_selector_all(".spec svg polyline"), "no fitted curves on spectrogram"
            assert page.query_selector_all("text=Detected events") or "Detected events" in page.inner_text("main")
            body = page.inner_text("body")
            assert "Modules:" in body
            assert "Data and attribution" in body and "adsb.lol" in body
            assert page.query_selector(".warn")
            for bad in ("undefined", "NaN", "[object Object]"):
                assert bad not in body, f"{bad!r} shown in UI"
        finally:
            browser.close()
            if job_id:
                httpx.delete(f"{server}/api/jobs/{job_id}", timeout=30)
    print("console:", console)
    print("failed/external:", failed)
    assert not errors, errors
    assert not [c for c in console if c.startswith(("error", "warning"))], console
    assert not failed, failed
