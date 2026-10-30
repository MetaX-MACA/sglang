"""Shared version policy, loadable by setup.py without importing sglang."""

import os
import re

BASE_VERSION = "0.5.22"


def with_base_version(version):
    return re.sub(r"^\d+\.\d+\.\d+", BASE_VERSION, version, count=1)


def version_scheme(version):
    from packaging.version import Version

    # Keep the local suffix here: setuptools-scm 10 deduplicates dot-separated
    # local_scheme segments, corrupting repeated MACA version parts.
    node = version.node or "unknown"
    return str(Version(maca_version(node.removeprefix("g"))))


def maca_version(commit):
    # PEP 440 local segments allow only letters, digits, and dot separators.
    def normalize(value):
        return re.sub(r"[^a-z0-9]+", ".", value.lower()).strip(".") or "unknown"

    node = normalize(commit)
    maca = normalize(os.environ.get("MACA_AI_VERSION", "0.0.0.0"))
    return f"{BASE_VERSION}+{node}-maca{maca}"


def version_tuple(version):
    from packaging.version import Version

    parsed = Version(version)
    fields = parsed.release
    if parsed.pre is not None:
        fields += (f"{parsed.pre[0]}{parsed.pre[1]}",)
    if parsed.post is not None:
        fields += (f"post{parsed.post}",)
    if parsed.dev is not None:
        fields += (f"dev{parsed.dev}",)
    if parsed.local is not None:
        fields += (parsed.local,)
    return fields


SCM_VERSION = {
    "version_scheme": version_scheme,
    "local_scheme": "no-local-version",
    "fallback_version": BASE_VERSION,
}
