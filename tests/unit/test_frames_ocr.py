import shutil
import subprocess
from datetime import datetime

import pytest

from triangulate.visual.frames import dhash64, extract_frames, hamming
from triangulate.visual.ocr import ocr_frames, parse_timestamps, timestamp_hint, video_timestamp_hint

T0 = 1788880544  # 2026-09-08 15:15:44 UTC as "gmtime" wall clock


def _font() -> str | None:
    r = subprocess.run(["fc-match", "-f", "%{file}", "sans"], capture_output=True, text=True)
    return r.stdout.strip() or None


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    font = _font()
    if not shutil.which("ffmpeg") or not font:
        pytest.skip("ffmpeg/font unavailable")
    p = tmp_path_factory.mktemp("vid") / "syn.mp4"
    # four distinct 30 s scenes, each static-ish (so duplicates abound), "MAIN ST" sign, ticking timestamp.
    vf = (
        "[0:v][1:v][2:v][3:v]concat=n=4:v=1:a=0,"
        f"drawtext=fontfile={font}:text='MAIN ST':fontsize=110:fontcolor=white:box=1:boxcolor=0x006030:"
        "boxborderw=20:x=(w-tw)/2:y=(h-th)/2,"
        f"drawtext=fontfile={font}:text='%{{pts\\:gmtime\\:{T0}\\:%Y-%m-%d %H\\\\\\:%M\\\\\\:%S}}':"
        "fontsize=44:fontcolor=white:box=1:boxcolor=black:x=20:y=20"
    )
    cmd = ["ffmpeg", "-hide_banner", "-y", "-f", "lavfi", "-i", "testsrc=s=1280x720:r=5:d=30",
           "-f", "lavfi", "-i", "smptebars=s=1280x720:r=5:d=30",
           "-f", "lavfi", "-i", "rgbtestsrc=s=1280x720:r=5:d=30",
           "-f", "lavfi", "-i", "color=c=0x3050a0:s=1280x720:r=5:d=30",
           "-filter_complex", vf,
           "-c:v", "libx264", "-pix_fmt", "yuv420p", str(p)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    return p


@pytest.fixture(scope="module")
def frames(video, tmp_path_factory):
    return extract_frames(video, tmp_path_factory.mktemp("fr"))


def test_frames_capped_and_deduped(frames):
    assert 1 <= len(frames) <= 200
    # ticking timestamp changes only a tiny area -> near-duplicates are collapsed
    assert 3 <= len(frames) <= 12
    hs = [int(f.dhash, 16) for f in frames]
    assert all(hamming(a, b) > 6 for i, a in enumerate(hs) for b in hs[i + 1 :])
    assert frames == sorted(frames, key=lambda f: f.t_s)


def test_cap(video, tmp_path):
    fr = extract_frames(video, tmp_path, max_frames=1, hash_dist=-1)
    assert len(fr) == 1


def test_dhash_identity():
    import numpy as np

    a = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
    assert hamming(dhash64(a), dhash64(a.copy())) == 0


def test_ocr_main_st(frames):
    res = ocr_frames(frames[:3], osd=False)
    assert any("MAIN" in r.text.upper() for r in res)


def test_timestamp_hint_on_frames(frames):
    h = timestamp_hint(frames)
    assert h is not None and h.tz_ambiguous and h.trust == "low"
    assert abs((h.start_naive - datetime(2026, 9, 8, 15, 15, 44)).total_seconds()) <= 3


def test_parse_timestamps():
    assert parse_timestamps("CAM1 2026-09-08 16:15:44")[0][0] == datetime(2026, 9, 8, 16, 15, 44)
    dt, amb = parse_timestamps("13/09/2026 07:01:02")[0]
    assert dt == datetime(2026, 9, 13, 7, 1, 2) and not amb
    assert parse_timestamps("02/03/2026 07:01:02")[0][1] is True
    assert parse_timestamps("hello") == []


def test_video_timestamp_hint(video):
    h = video_timestamp_hint(video)
    assert h is not None and h.n_consistent >= 3
    assert abs((h.start_naive - datetime(2026, 9, 8, 15, 15, 44)).total_seconds()) <= 3
