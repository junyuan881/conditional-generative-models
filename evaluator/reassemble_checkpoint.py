"""Reassemble the evaluator checkpoint from GitHub-friendly parts."""

from __future__ import annotations

import hashlib
from pathlib import Path


EXPECTED_SHA256 = "38a27369dac0ad64302bec2ee8d98bd6f27de17f4bde60ebcd2e3ff7981e9d63"


def main() -> None:
    evaluator_dir = Path(__file__).resolve().parent
    parts = sorted(evaluator_dir.glob("checkpoint.pth.part*"))
    if not parts:
        raise FileNotFoundError("No checkpoint.pth.part* files were found")

    target = evaluator_dir / "checkpoint.pth"
    temporary = evaluator_dir / "checkpoint.pth.tmp"
    digest = hashlib.sha256()

    with temporary.open("wb") as destination:
        for part in parts:
            with part.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    destination.write(chunk)
                    digest.update(chunk)

    actual_sha256 = digest.hexdigest()
    if actual_sha256 != EXPECTED_SHA256:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"Checkpoint checksum mismatch: expected {EXPECTED_SHA256}, "
            f"got {actual_sha256}"
        )

    temporary.replace(target)
    print(f"Reassembled {target.name} ({target.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
