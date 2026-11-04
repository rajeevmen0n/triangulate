"""WP21: the single page is served and every static asset it references resolves."""

import re
from html.parser import HTMLParser
from pathlib import PurePosixPath
from urllib.parse import urlparse

import pytest
from fastapi.testclient import TestClient

from triangulate.api import create_app
from triangulate.config import Settings


@pytest.fixture
def client(tmp_path):
    app = create_app(Settings(data=tmp_path / "data"), start_runner=False)
    with TestClient(app) as c:
        yield c


class _Refs(HTMLParser):
    def __init__(self):
        super().__init__()
        self.refs = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        for k in ("src", "href"):
            if a.get(k) and not a[k].startswith("#"):
                self.refs.append(a[k])


def _local(ref):
    u = urlparse(ref)
    assert not u.scheme and not u.netloc, f"external reference: {ref}"
    return u.path


def test_index_served_and_assets_resolve(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    p = _Refs()
    p.feed(r.text)
    assert p.refs
    seen, todo = set(), [_local(x) for x in p.refs]
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        seen.add(path)
        url = "/" + path.lstrip("/")
        a = client.get(url)
        assert a.status_code == 200, f"{url} -> {a.status_code}"
        base = PurePosixPath(url).parent
        if url.endswith(".js"):  # relative ES module imports
            for m in re.findall(r"""from\s+["'](\.[^"']+)["']""", a.text):
                todo.append(str((base / m)).replace("//", "/"))
        if url.endswith(".css"):
            for m in re.findall(r"url\(['\"]?([^)'\"]+)", a.text):
                if not m.startswith(("data:", "http", "#")):
                    todo.append(str(base / m))
    assert any(s.endswith("leaflet.js") for s in seen) and any(s.endswith("app.js") for s in seen)


def test_page_has_notice_and_attribution_hooks(client):
    html = client.get("/").text
    assert 'id="use-notice"' in html and 'id="attribution"' in html
    cfg = client.get("/api/config").json()
    assert cfg["use_notice"] and any("OpenStreetMap" in a for a in cfg["attribution"])


def test_no_external_urls_in_static(client):
    from triangulate.api.app import STATIC_DIR

    for f in STATIC_DIR.rglob("*"):
        if f.is_file() and f.suffix in {".html", ".css"} or f.name in {"app.js", "job.js", "form.js", "results.js", "evidence.js", "map.js", "util.js"}:
            assert not re.search(r"""(src|href)=["']https?://""", f.read_text()), f
