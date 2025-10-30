"""Runtime dependency bootstrapper.

Prefer installing dependencies via requirements.txt on deploy. This helper is
an escape hatch to install missing packages at runtime when developing locally
or on platforms that allow it. Enable with env var ALLOW_RUNTIME_INSTALL=1.
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from typing import Dict, Optional


DEFAULT_PACKAGES: Dict[str, Optional[str]] = {
    # pip name: exact version or None for latest
    "streamlit": None,
    # Use headless OpenCV by default for cloud environments
    "opencv-python-headless": None,
    "numpy": None,
    "Pillow": None,
}


def _module_name(pip_name: str) -> str:
    # Map pip package names to importable module names
    if pip_name == "Pillow":
        return "PIL"
    return pip_name.replace("-", "_")


def _is_installed(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def _pip_install(spec: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", spec])


def ensure_packages(packages: Dict[str, Optional[str]] = None, allow_runtime: bool = False) -> bool:
    """Ensure required packages are installed.

    Returns True if any packages were installed. If `allow_runtime` is False,
    this function will only run when the environment variable
    `ALLOW_RUNTIME_INSTALL` is set to a truthy value.
    """
    env_ok = os.getenv("ALLOW_RUNTIME_INSTALL", "").lower() in {"1", "true", "yes"}
    if not allow_runtime and not env_ok:
        return False
    if allow_runtime and not env_ok:
        # still require the env var to be set, even if called with allow_runtime
        return False

    pkg_map = packages or DEFAULT_PACKAGES
    installed_any = False
    for pip_name, version in pkg_map.items():
        module = _module_name(pip_name)
        if _is_installed(module):
            continue
        spec = f"{pip_name}=={version}" if version else pip_name
        try:
            _pip_install(spec)
            installed_any = True
        except Exception:
            # Ignore failures to avoid breaking the app; user can install manually.
            pass
    return installed_any
