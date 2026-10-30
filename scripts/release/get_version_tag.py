#!/usr/bin/env python3
"""Resolve the fixed MACA package version without importing sglang.

The default output is ``0.5.22+<commit>-maca<MACA_AI_VERSION>``.
``--scm-describe`` provides setuptools-scm's git-describe protocol, while
``--tag-only`` returns the fixed base tag for release scripts.
"""

import runpy
import subprocess
import sys
from pathlib import Path


def get_git_short_commit() -> str:
    """Return the short HEAD hash, or unknown outside a Git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
        )
    except OSError:
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    helpers = runpy.run_path(str(root / "python/sglang/_build_version.py"))
    base = helpers["BASE_VERSION"]
    if "--tag-only" in sys.argv:
        print(f"v{base}")
        return

    commit = get_git_short_commit()
    if "--scm-describe" in sys.argv:
        # setuptools-scm must parse a describe string, not a local version
        # containing a hyphen. Ignore tags, distance, branch and dirty state.
        print(f"v{base}-0-g{commit}" if commit != "unknown" else f"v{base}")
    else:
        # --maca-version remains an alias for the default output.
        print(helpers["maca_version"](commit))


if __name__ == "__main__":
    main()
