{
  description = "triangulate dev environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in {
      devShells = forAllSystems (pkgs:
        let
          # All Python deps come from nixpkgs (never pip/uv). To add one, add it to this list.
          # For a library missing from nixpkgs, define it with ps.buildPythonPackage
          # (fetchPypi) in nix/python.nix and `git add` that file (flakes only see tracked files).
          pyPkgs = ps: with ps; [
            numpy scipy numba polars pyarrow duckdb orjson
            rasterio pyproj shapely h3 timezonefinder pvlib
            fastapi uvicorn python-multipart sse-starlette pydantic pydantic-settings python-dotenv httpx jinja2
            anthropic
            pillow opencv4 tesserocr soundfile matplotlib
            pytest pytest-asyncio hypothesis
          ];
          pythonEnv = pkgs.python3.withPackages pyPkgs;
          # Separate env for the real-browser UI check only (`nix develop .#browser`).
          pythonBrowserEnv = pkgs.python3.withPackages (ps: pyPkgs ps ++ [ ps.playwright ]);
          tools = [
            pkgs.tesseract pkgs.ffmpeg pkgs.exiftool pkgs.ruff pkgs.mypy
            pkgs.git pkgs.yt-dlp pkgs.nodejs
          ];
          baseHook = ''
              export PROJECT_ROOT="$PWD"
              export DEVHOME="$PROJECT_ROOT/.devhome"
              mkdir -p "$DEVHOME"/{home,config,cache,data,state,tmp}
              export HOME="$DEVHOME/home"
              export XDG_CONFIG_HOME="$DEVHOME/config"
              export XDG_CACHE_HOME="$DEVHOME/cache"
              export XDG_DATA_HOME="$DEVHOME/data"
              export XDG_STATE_HOME="$DEVHOME/state"
              export TMPDIR="$DEVHOME/tmp"

              export PYTHONPATH="$PROJECT_ROOT/src''${PYTHONPATH:+:$PYTHONPATH}"
              export TRIANGULATE_DATA="$PROJECT_ROOT/data"
              export NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
              export MPLCONFIGDIR="$XDG_CONFIG_HOME/matplotlib"
              export PROJ_USER_WRITABLE_DIRECTORY="$XDG_DATA_HOME/proj"
              export TESSDATA_PREFIX="${pkgs.tesseract}/share/tessdata"
              # .env is deliberately NOT sourced here; the app loads it itself.

              # Dev-only defaults for the video "watch" tool (not used by the app). WATCH_SKILL_DIR is
              # passed through from the caller's environment if set.
              export WATCH_ENGINE=local
              export WATCH_DETAIL=balanced
              export WATCH_WHISPER_BACKEND=none
            '';
        in {
          default = pkgs.mkShell {
            packages = [ pythonEnv ] ++ tools;
            shellHook = baseHook;
          };
          browser = pkgs.mkShell {
            packages = [ pythonBrowserEnv ] ++ tools;
            shellHook = baseHook + ''
              export PLAYWRIGHT_BROWSERS_PATH="${pkgs.playwright-driver.browsers}"
              export PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS=true
              export PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1
            '';
          };
        });
    };
}
