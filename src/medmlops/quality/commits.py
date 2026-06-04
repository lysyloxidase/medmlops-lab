"""Conventional Commit message validation for the local commit-msg hook."""

from __future__ import annotations

import re
import sys
from pathlib import Path

CONVENTIONAL_COMMIT = re.compile(
    r"^(build|chore|ci|docs|feat|fix|perf|refactor|revert|style|test)"
    r"(\([A-Za-z0-9._/-]+\))?!?: .+"
)


def is_conventional_commit_message(message: str) -> bool:
    """Return whether the first line follows Conventional Commits."""

    first_line = message.splitlines()[0].strip() if message.splitlines() else ""
    return bool(
        first_line.startswith(("Merge ", "Revert "))
        or CONVENTIONAL_COMMIT.match(first_line)
    )


def main() -> int:
    """Validate the commit-message file supplied by pre-commit."""

    if len(sys.argv) != 2:
        print("usage: python -m medmlops.quality.commits <commit-message-file>")
        return 2
    path = Path(sys.argv[1])
    if is_conventional_commit_message(path.read_text(encoding="utf-8")):
        return 0
    print(
        "Commit message must follow Conventional Commits, for example: "
        "'feat: add release gate'."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
