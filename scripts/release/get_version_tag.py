#!/usr/bin/env python3
"""Resolve the correct version tag for setuptools-scm.

Called by setuptools-scm via git_describe_command in pyproject.toml.
Outputs either a bare tag (e.g., "v0.5.10") for exact-match commits,
or a `git describe --long` string (e.g., "v0.5.10-2-gabcdef0") for
untagged commits. Both formats are accepted by setuptools-scm.

This two-step approach avoids a strverscmp bug where
`git tag --sort=-version:refname` sorts v0.5.10rc0 above v0.5.10,
which would cause CI to build the wrong version.

Strategy:
1. If the current commit has an exact version tag, use it directly.
   This handles CI release builds (both stable and rc).
2. Otherwise, find the highest version tag across all branches
   and describe relative to it. This handles local dev installs
   from main where release tags only exist on release branches.
"""

import subprocess
import sys
import os



def run_git(*args: str, allow_failure: bool = False) -> str:
    """Run a git command and return stripped stdout.

    Args:
        allow_failure: If True, return "" on non-zero exit (expected for
            commands like --exact-match that legitimately fail).
            If False, log stderr on failure before returning "".
    """
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        print(f"ERROR: Failed to run 'git {' '.join(args)}': {exc}", file=sys.stderr)
        sys.exit(1)

    if result.returncode != 0:
        if not allow_failure:
            stderr_msg = result.stderr.strip()
            print(
                f"WARNING: git {' '.join(args)} failed "
                f"(exit {result.returncode}): {stderr_msg}",
                file=sys.stderr,
            )
        return ""

    return result.stdout.strip()

def get_git_branch() -> str:
    """获取当前分支名，并将斜杠替换为点（例如 release/v0.5.17 -> release.v0.5.17）。"""
    branch = run_git("branch", "--show-current")
    if not branch:
        branch = run_git("rev-parse", "--abbrev-ref", "HEAD")
        if branch == "HEAD":
            branch = "detached"

    if not branch:
        branch = "unknown"

    return branch.replace('/', '.')

def get_git_short_commit() -> str:
    """获取短 commit hash。"""
    return run_git("rev-parse", "--short", "HEAD", allow_failure=True) or "unknown"

def get_maca_ai_version() -> str:
    """从 环境变量 提取 MACA 版本，如 "3.8.0.4" """
    return os.environ.get('MACA_AI_VERSION', '0.0.0.0')

def get_current_base_version() -> str:
    """获取当前提交的最近标签（不含距离），如 'v0.5.14' 或 '0.5.14'，失败返回 '0.0.0'。"""
    tag = run_git("describe", "--tags", "--abbrev=0", allow_failure=True)
    if not tag:
        return "0.0.0"
    return tag.lstrip('v')


def main() -> None:
    # 获取基础版本（当前分支最近标签）
    base = get_current_base_version()

    # 获取自定义信息
    branch = get_git_branch()
    commitid = get_git_short_commit()
    ai_maca = get_maca_ai_version()

    # 拼接自定义部分：分支名-commitid-maca版本号
    result = f"{base}+{branch}-{commitid}-maca{ai_maca}"
    # print("环境变量获取maca版本：",ai_result)
    print(result)


if __name__ == "__main__":
    main()
