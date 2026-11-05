# triangulate

Estimate where an audio/video recording was made by fusing aircraft-Doppler matching against
ADS-B tracks (acoustic), frame clues (visual, opt-in, Claude API) and untrusted file metadata.
v1 is region search: a region (bbox) and a date are required. Full design: `docs/plan.md`.

For OSINT verification, journalism and research; not for locating private individuals.
See `RESPONSIBLE_USE.md` for the use policy, attribution, retention (`triangulate purge`) and rate limits.

## Credits

The method automated here comes from the YouTube video
[**"I discovered a sound that leaks your exact location (and it's everywhere)"**](https://www.youtube.com/watch?v=UytYrhYLOKI)
by **colsto** ([@colsto](https://www.youtube.com/@colsto)). In that video, colsto geolocates a recording from the Doppler
curves of propeller aircraft heard in it: the curves are matched against flight tracks over a grid of candidate points,
using terrain elevation and filtering on aircraft that should have been heard but weren't. The demo recording was published
separately as ["Unknown location - raw audio"](https://www.youtube.com/watch?v=4CxVxuMVkMc).

colsto released no code. This project is an independent implementation built from the video, and it is not
affiliated with or endorsed by colsto. Notes on the video are in `docs/video-notes.md`. The video and
audio files used for development are not part of this repository.

## Dev shell and tests

Everything runs inside the Nix dev shell (Python env from nixpkgs; no pip/uv, no global installs).
The shell keeps HOME/XDG/TMPDIR inside `.devhome/`.

```sh
cd /home/rajeev/work/repos/triangulate
nix develop                                   # interactive shell (or `direnv allow`)
nix develop --command pytest -q               # run tests
nix develop --command pytest -m "not network and not slow and not llm"
nix develop --command python -m triangulate --help
nix develop --command ruff check src tests
```

Markers: `network`, `slow`, `acceptance`, `llm` (skipped without `ANTHROPIC_API_KEY`).

## Configuration

Copy `.env.example` to `.env` (gitignored). It is read by the app (pydantic-settings), never
sourced by the shell. Key settings: `ANTHROPIC_API_KEY` (visual module, opt-in per job, $2/job cap
via `TRIANGULATE_LLM_BUDGET_USD`), `GITHUB_TOKEN`, `TRIANGULATE_CONTACT` (appended to the
`triangulate/0.1` User-Agent for Nominatim/Overpass), `TRIANGULATE_BASEMAP_ENABLED`.
Settings live in `src/triangulate/config.py` (`get_settings()`). Data goes to `data/` (gitignored).

## Adding a dependency

Add it to the `python3.withPackages` list in `flake.nix`, then re-enter the shell. Never pip-install.
If nixpkgs lacks it, define it with `buildPythonPackage` + `fetchPypi` in `nix/python.nix` and
`git add` the file (flakes only see git-tracked files).
