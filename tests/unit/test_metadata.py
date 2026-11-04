"""Metadata module tests on synthetic media only (never video.mp4 / audio.mp3)."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime

import pytest

from triangulate.ingest.metadata import extract_metadata, parse_iso6709, time_components


def _run(*args: str) -> None:
    subprocess.run(args, check=True, capture_output=True)


@pytest.fixture(scope="module")
def tagged(tmp_path_factory):
    d = tmp_path_factory.mktemp("meta")
    p = d / "tagged.mp4"
    _run("ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=d=2", "-metadata",
         "creation_time=2025-09-08T14:30:00Z", "-metadata", "location=+40.8767-073.5264/", str(p))
    return p


@pytest.fixture(scope="module")
def plain(tmp_path_factory):
    p = tmp_path_factory.mktemp("meta2") / "plain.wav"
    _run("ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=d=1", "-map_metadata", "-1",
         "-bitexact", str(p))
    return p


@pytest.fixture(scope="module")
def jpeg(tmp_path_factory):
    p = tmp_path_factory.mktemp("meta3") / "j.jpg"
    _run("ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=red:s=16x16", "-frames:v", "1", str(p))
    _run("exiftool", "-overwrite_original", "-GPSLatitude=40.8767", "-GPSLatitudeRef=N",
         "-GPSLongitude=73.5264", "-GPSLongitudeRef=W", "-DateTimeOriginal=2025:09:08 10:30:00",
         "-OffsetTimeOriginal=-04:00", "-Make=Canon", "-Model=X1", str(p))
    return p


def test_iso6709():
    assert parse_iso6709("+40.8767-073.5264/") == (40.8767, -73.5264, None)
    assert parse_iso6709("+40.8767-073.5264+012.5/")[2] == 12.5
    assert parse_iso6709("garbage") is None


def test_video_tags(tagged):
    r = extract_metadata(tagged)
    assert r.gps and r.gps[0].lat == pytest.approx(40.8767, abs=1e-3)
    assert r.gps[0].lon == pytest.approx(-73.5264, abs=1e-3)
    assert r.gps[0].sigma_m == 30.0
    assert r.times[0].naive == datetime(2025, 9, 8, 14, 30)
    assert all(h.trust in ("low", "very_low") for h in [*r.gps, *r.times])
    assert r.reencode_markers  # Lavf encoder, no make/model
    assert all(h.trust == "very_low" for h in r.gps)
    comps = time_components(r, tz="America/New_York")
    assert sum(c.weight for c in comps) == pytest.approx(1.0)
    ts = {c.t_s for c in comps}
    assert datetime(2025, 9, 8, 14, 30, tzinfo=UTC) in ts  # as UTC
    assert datetime(2025, 9, 8, 18, 30, tzinfo=UTC) in ts  # as EDT local
    assert datetime(2025, 9, 8, 19, 30, tzinfo=UTC) in ts  # DST +1h
    assert any(abs((c.t_s - datetime(2025, 9, 8, 14, 29, 58, tzinfo=UTC)).total_seconds()) < 1
               for c in comps)  # end-of-recording (2 s)
    assert all(c.trust != "high" for c in comps)


def test_jpeg_exif(jpeg):
    r = extract_metadata(jpeg)
    assert r.gps[0].lat == pytest.approx(40.8767, abs=1e-4)
    assert r.gps[0].lon == pytest.approx(-73.5264, abs=1e-4)
    assert r.gps[0].trust == "low"
    assert (r.make, r.model) == ("Canon", "X1")
    assert r.times[0].offset_min == -240
    comps = time_components(r, tz="America/New_York")
    assert datetime(2025, 9, 8, 14, 30, tzinfo=UTC) in {c.t_s for c in comps}


def test_no_metadata(plain, tmp_path):
    r = extract_metadata(plain)
    assert not r.gps and not r.times and not r.has_hints
    assert time_components(r, tz="UTC") == []
    bad = tmp_path / "junk.bin"
    bad.write_bytes(b"not media")
    r2 = extract_metadata(bad)  # must not raise
    assert not r2.has_hints
    r3 = extract_metadata(tmp_path / "missing.mp4")
    assert not r3.has_hints
