"""Local OCR (tesseract via tesserocr) and burned-in timestamp OCR (plan 3.6 step 2). No network.

Public API:
    ocr_frames(frames, ...) -> list[FrameOcr]
    ocr_text_block(results) -> str                 (grounding text for the LLM prompt)
    timestamp_hint(frames, ...) -> TimestampHint | None
"""

from __future__ import annotations

import os
import re
import statistics
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from .frames import Frame

MIN_WORD_CONF = 40.0


class OcrWord(BaseModel):
    text: str
    conf: float


class FrameOcr(BaseModel):
    frame_id: str
    text: str  # words joined with spaces, confident words only
    words: list[OcrWord] = Field(default_factory=list)
    script: str | None = None  # OSD script, e.g. "Latin"
    script_conf: float | None = None


class TimestampHint(BaseModel):
    """Naive (no tz) wall-clock time of video position 0, i.e. an untrusted, tz-ambiguous T_s hint."""

    start_naive: datetime
    tz_ambiguous: bool = True
    trust: str = "low"
    n_frames_read: int
    n_consistent: int
    spread_s: float
    date_order_ambiguous: bool = False  # DD/MM vs MM/DD could not be resolved
    samples: list[dict] = Field(default_factory=list)  # {frame_id, t_s, raw, naive}


_EPOCH = datetime(1970, 1, 1)


def _ep(dt: datetime) -> float:
    return (dt - _EPOCH).total_seconds()


def _tessdata() -> str | None:
    p = os.environ.get("TESSDATA_PREFIX")
    return p if p and Path(p).exists() else None


def _load(path: str | Path) -> Image.Image:
    return Image.open(path).convert("L")


def _read_words(api, img: Image.Image) -> list[OcrWord]:
    api.SetImage(img)
    api.Recognize()
    txt_words = api.GetUTF8Text().split()
    confs = api.AllWordConfidences()
    out: list[OcrWord] = []
    if len(txt_words) == len(confs):
        out = [OcrWord(text=w, conf=float(c)) for w, c in zip(txt_words, confs, strict=True)]
    else:  # fallback: line-level text with the mean confidence
        c = float(api.MeanTextConf())
        out = [OcrWord(text=w, conf=c) for w in txt_words]
    return out


def ocr_frames(
    frames: Sequence[Frame], *, lang: str = "eng", osd: bool = True, min_conf: float = MIN_WORD_CONF
) -> list[FrameOcr]:
    """PSM 11 (sparse text) on every frame, plus OSD script detection (best effort)."""
    from tesserocr import PSM, PyTessBaseAPI

    td = _tessdata()
    kw = {"path": td} if td else {}
    res: list[FrameOcr] = []
    with PyTessBaseAPI(psm=PSM.SPARSE_TEXT, lang=lang, **kw) as api:
        osd_api = None
        if osd:
            try:
                osd_api = PyTessBaseAPI(psm=PSM.OSD_ONLY, lang="osd", **kw)
            except Exception:
                osd_api = None
        try:
            for f in frames:
                img = _load(f.path)
                words = _read_words(api, img)
                keep = [w for w in words if w.conf >= min_conf and re.search(r"\w", w.text)]
                script = conf = None
                if osd_api is not None:
                    try:
                        osd_api.SetImage(img)
                        r = osd_api.DetectOrientationScript()
                        if r:
                            script, conf = r[2], float(r[3])
                    except Exception:
                        pass
                res.append(
                    FrameOcr(
                        frame_id=f.id, text=" ".join(w.text for w in keep), words=keep,
                        script=script, script_conf=conf,
                    )
                )
        finally:
            if osd_api is not None:
                osd_api.End()
    return res


def ocr_text_block(results: Sequence[FrameOcr], max_chars: int = 6000) -> str:
    """Compact per-frame text for prompting; frames with no text are omitted."""
    lines = [f"[{r.frame_id}] {r.text}" for r in results if r.text.strip()]
    return "\n".join(lines)[:max_chars]


# ---------------------------------------------------------------- timestamp OCR

_D = r"[0-9OoIl|]"
_TIME = rf"[\sT_,]*({_D}{{1,2}})\s*:\s*({_D}{{2}})\s*:\s*({_D}{{2}})"
_YMD = re.compile(rf"({_D}{{4}})\s*[-/.]\s*({_D}{{2}})\s*[-/.]\s*({_D}{{2}})" + _TIME)
_DMY = re.compile(rf"({_D}{{2}})\s*[-/.]\s*({_D}{{2}})\s*[-/.]\s*({_D}{{4}})" + _TIME)
_FIX = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "|": "1"})


def _ints(*g: str) -> list[int]:
    return [int(x.translate(_FIX)) for x in g]


def parse_timestamps(text: str) -> list[tuple[datetime, bool]]:
    """All (naive datetime, date_order_ambiguous) matches in `text`."""
    out: list[tuple[datetime, bool]] = []
    for m in _YMD.finditer(text):
        y, mo, d, h, mi, s = _ints(*m.groups())
        try:
            out.append((datetime(y, mo, d, h, mi, s), False))
        except ValueError:
            pass
    for m in _DMY.finditer(text):
        a, b, y, h, mi, s = _ints(*m.groups())
        amb = a <= 12 and b <= 12 and a != b
        for d, mo in ((a, b), (b, a)):  # prefer DD/MM when ambiguous
            try:
                out.append((datetime(y, mo, d, h, mi, s), amb))
                break
            except ValueError:
                continue
    return [(dt, a) for dt, a in out if 2000 <= dt.year <= 2100]


def _corners(img: np.ndarray, wf: float = 0.5, hf: float = 0.22) -> list[np.ndarray]:
    h, w = img.shape[:2]
    cw, ch = int(w * wf), int(h * hf)
    return [img[:ch, :cw], img[:ch, w - cw :], img[h - ch :, :cw], img[h - ch :, w - cw :]]


def _prep(c: np.ndarray, invert: bool) -> Image.Image:
    c = cv2.resize(c, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    c = cv2.copyMakeBorder(c, 20, 20, 20, 20, cv2.BORDER_REPLICATE)
    if invert:
        c = 255 - c
    return Image.fromarray(c)


def read_frame_timestamp(api, img: np.ndarray | str | Path) -> tuple[datetime, bool, str] | None:
    if not isinstance(img, np.ndarray):
        img = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    for c in _corners(img):
        for invert in (False, True):
            api.SetImage(_prep(c, invert))
            txt = api.GetUTF8Text()
            found = parse_timestamps(txt)
            if found:
                dt, amb = found[0]
                return dt, amb, txt.strip()
    return None


def timestamp_hint(
    frames: Sequence[Frame], *, n_probe: int = 5, tol_s: float = 3.0, lang: str = "eng"
) -> TimestampHint | None:
    """Timestamp OCR on `n_probe` spaced (already de-duplicated) frames. See `_hint`."""
    if not frames:
        return None
    idx = sorted({int(round(i)) for i in np.linspace(0, len(frames) - 1, min(n_probe, len(frames)))})
    items = [(frames[i].id, frames[i].t_s, frames[i].path) for i in idx]
    return _hint(items, tol_s, lang)


def video_timestamp_hint(
    video: str | Path, *, n_probe: int = 5, tol_s: float = 3.0, lang: str = "eng"
) -> TimestampHint | None:
    """Like `timestamp_hint` but samples `n_probe` evenly spaced positions straight from the video.

    Preferred: a static camera collapses to very few frames after de-duplication, which would leave
    too few readings to cross-check.
    """
    cap = cv2.VideoCapture(str(video))
    try:
        n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
        if not n or not fps:
            return None
        dur = n / fps
        items = []
        for k in range(n_probe):
            t = dur * (k + 0.5) / n_probe
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, img = cap.read()
            if ok:
                items.append((f"t{k}", int(t * fps) / fps, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
    finally:
        cap.release()
    return _hint(items, tol_s, lang)


def _hint(items: list, tol_s: float, lang: str) -> TimestampHint | None:
    """Each reading implies a start time `reading - t`; a hint needs >= 2 readings agreeing within
    `tol_s` and forming a majority of those read. Result is naive (tz unknown, untrusted)."""
    from tesserocr import PSM, PyTessBaseAPI

    td = _tessdata()
    kw = {"path": td} if td else {}
    samples: list[dict] = []
    with PyTessBaseAPI(psm=PSM.SPARSE_TEXT, lang=lang, **kw) as api:
        api.SetVariable("tessedit_char_whitelist", "0123456789-/.:TOolI| ")
        for fid, t, img in items:
            r = read_frame_timestamp(api, img)
            if r is None:
                continue
            dt, amb, raw = r
            samples.append(
                {"frame_id": fid, "t_s": t, "raw": raw, "naive": dt, "amb": amb,
                 "start": dt - timedelta(seconds=t)}
            )
    if len(samples) < 2:
        return None
    ref = statistics.median(_ep(s["start"]) for s in samples)
    good = [s for s in samples if abs(_ep(s["start"]) - ref) <= tol_s]
    if len(good) < 2 or len(good) * 2 <= len(samples):
        return None
    start = _EPOCH + timedelta(seconds=statistics.median(_ep(s["start"]) for s in good))
    spread = max(_ep(s["start"]) for s in good) - min(_ep(s["start"]) for s in good)
    return TimestampHint(
        start_naive=start,
        n_frames_read=len(samples),
        n_consistent=len(good),
        spread_s=spread,
        date_order_ambiguous=any(s["amb"] for s in good),
        samples=[
            {"frame_id": s["frame_id"], "t_s": s["t_s"], "raw": s["raw"], "naive": s["naive"].isoformat()}
            for s in samples
        ],
    )
