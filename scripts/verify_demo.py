#!/usr/bin/env python3
"""Fail closed when published demo media or captured source drifts."""
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "docs" / "demo"


def run(*args):
    return subprocess.run(args, cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.strip()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    manifest = json.loads((DEMO / "capture.json").read_text())
    source = manifest["recorded_source_sha"]
    run("git", "merge-base", "--is-ancestor", source, "HEAD")
    for path, expected in manifest["runtime_input_blobs"].items():
        recorded = run("git", "rev-parse", f"{source}:{path}")
        current = run("git", "rev-parse", f"HEAD:{path}")
        if recorded != expected or current != expected:
            raise SystemExit(f"runtime source drift: {path}")
    for name, expected in manifest["artifacts"].items():
        path = DEMO / name
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise SystemExit(f"artifact mismatch: {name}")
        run("ffprobe", "-v", "error", "-count_frames", "-show_entries", "stream=width,height,nb_read_frames", "-of", "json", str(path))
    if manifest["receipt"]["model_results"]["sample_rows"] != 2000:
        raise SystemExit("published receipt is not the documented 2,000-row smoke replay")
    print(f"verified source {source} and {len(manifest['artifacts'])} fully decoded artifacts")


if __name__ == "__main__":
    main()
