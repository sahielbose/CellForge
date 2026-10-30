"""Utilities shared by code generation backends."""

import os
from pathlib import Path


def load_project_env(project_root: Path) -> None:
    try:
        from dotenv import load_dotenv

        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except Exception:
        pass


def truthy_env(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}
