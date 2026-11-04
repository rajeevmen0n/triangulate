"""Settings. Loaded from process env (TRIANGULATE_*, ANTHROPIC_API_KEY, GITHUB_TOKEN) and `.env`.

`.env` is looked up at $TRIANGULATE_ENV_FILE or <project root>/.env (project root = the directory
holding pyproject.toml above this file, else the cwd). The dev shell never sources it.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from . import USER_AGENT_BASE


def project_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / "pyproject.toml").exists():
            return p
    return Path.cwd()


def _env_file() -> Path:
    return Path(os.environ.get("TRIANGULATE_ENV_FILE", project_root() / ".env"))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TRIANGULATE_",
        env_file=None,  # set per instance in get_settings()
        extra="ignore",
        populate_by_name=True,
    )

    data: Path = Field(default_factory=lambda: project_root() / "data")

    # Secrets (unprefixed names, as in .env.example). Never logged.
    anthropic_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("ANTHROPIC_API_KEY", "TRIANGULATE_ANTHROPIC_API_KEY")
    )
    github_token: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("GITHUB_TOKEN", "TRIANGULATE_GITHUB_TOKEN")
    )

    # Third-party etiquette: User-Agent is "triangulate/0.1" (+ " (contact)" if given).
    contact: str | None = None

    # Visual module / Claude API
    llm_budget_usd: float = 2.00  # hard per-job cap
    triage_model: str = "claude-haiku-5-5"
    clue_model: str = "claude-opus-5-5"

    # Web
    host: str = "127.0.0.1"
    port: int = 8000
    basemap_enabled: bool = True
    basemap_url: str = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    max_upload_bytes: int = 2_000_000_000
    retention_days: int = 7  # job workspaces/uploads older than this are purged; 0 disables (WP24)
    max_jobs_per_day: int = 20  # new jobs (upload or rerun) per rolling 24 h; 0 disables (WP24)

    # Physics / search defaults
    speed_of_sound: float = 343.0
    h_obs_m: float = 5.0
    exclude_helicopters: bool = False

    @property
    def llm_enabled(self) -> bool:
        return bool(self.anthropic_api_key and self.anthropic_api_key.get_secret_value())

    @property
    def user_agent(self) -> str:
        return f"{USER_AGENT_BASE} ({self.contact})" if self.contact else USER_AGENT_BASE

    # Data layout (plan section 4). Nothing is created at import time; call ensure_dirs().
    @property
    def cache_dir(self) -> Path:
        return self.data / "cache"

    @property
    def jobs_dir(self) -> Path:
        return self.data / "jobs"

    @property
    def fixtures_dir(self) -> Path:
        return self.data / "fixtures"

    @property
    def db_path(self) -> Path:
        return self.data / "jobs.db"

    def cache_subdir(self, name: str) -> Path:
        """e.g. adsb, dem, osm, geoid, types, stages."""
        return self.cache_dir / name

    def ensure_dirs(self) -> None:
        for p in (
            *(self.cache_subdir(n) for n in ("adsb", "dem", "osm", "geoid", "types", "stages")),
            self.jobs_dir,
            self.fixtures_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    ef = _env_file()
    return Settings(_env_file=ef if ef.exists() else None)  # type: ignore[call-arg]


def reload_settings() -> Settings:
    get_settings.cache_clear()
    return get_settings()
