#!/usr/bin/env python3
"""Record the tested coursework workflow as a source-pinned GIF and MP4."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "demo"
RUNTIME_INPUTS = [
    "econometrics_III.ipynb",
    "numerical.py",
    "run_research.py",
    "paeco526_take_home_final.csv",
    "requirements.txt",
    "tests/test_numerics.py",
    "tests/test_workflow.py",
    "scripts/generate_demo.py",
    "scripts/verify_demo.py",
]


def command(*args, env=None, capture=False):
    return subprocess.run(
        args,
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def git(*args):
    return command("git", *args, capture=True).stdout.strip()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def card(draw, box, title, value, accent=(87, 220, 190)):
    draw.rounded_rectangle(box, 18, fill=(25, 32, 50), outline=(53, 68, 96), width=2)
    draw.text((box[0] + 22, box[1] + 18), title.upper(), font=font(15, True), fill=(142, 158, 184))
    draw.text((box[0] + 22, box[1] + 56), value, font=font(27, True), fill=accent)


def slide(title, kicker, body, cards=(), footer="Synthetic coursework workflow • no causal or policy claim"):
    image = Image.new("RGB", (960, 540), (9, 14, 25))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 960, 8), fill=(87, 220, 190))
    draw.text((54, 40), kicker.upper(), font=font(15, True), fill=(87, 220, 190))
    draw.text((54, 70), title, font=font(38, True), fill=(242, 246, 255))
    y = 132
    for line in body:
        draw.text((56, y), line, font=font(21), fill=(188, 199, 220))
        y += 34
    if cards:
        width = (852 - 18 * (len(cards) - 1)) // len(cards)
        for index, (label, value) in enumerate(cards):
            left = 54 + index * (width + 18)
            card(draw, (left, 342, left + width, 454), label, value)
    draw.text((54, 496), footer, font=font(14), fill=(112, 128, 154))
    draw.text((870, 494), "AE III", font=font(16, True), fill=(87, 220, 190))
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--max-rows", type=int, default=2000)
    args = parser.parse_args()
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("Refusing to record with dirty tracked source files")
    if git("rev-parse", args.source_sha) != args.source_sha:
        raise SystemExit("--source-sha must be a full commit SHA")
    for path in RUNTIME_INPUTS:
        git("cat-file", "-e", f"{args.source_sha}:{path}")

    environment = os.environ.copy()
    environment["RESEARCH_MAX_ROWS"] = str(args.max_rows)
    with tempfile.TemporaryDirectory() as directory:
        receipt_dir = Path(directory) / "receipt"
        workflow = command(
            sys.executable,
            "run_research.py",
            "econometrics_III.ipynb",
            "--output",
            str(receipt_dir),
            env=environment,
            capture=True,
        )
        tests = command(sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v", capture=True)
        receipt = json.loads((receipt_dir / "execution.json").read_text())
        result = receipt["model_results"]
        test_count = tests.stdout.count(" ... ok")

        slides = [
            slide(
                "Birthweight workflow, replayed",
                "Advanced Econometrics III",
                ["Committed data → probit → signed residual", "→ outcome regression → machine-readable receipt"],
                [("Rows", f"{result['sample_rows']:,}"), ("Seed", str(receipt["seed"])), ("Tests", f"{test_count} passed")],
            ),
            slide(
                "The numerical bug",
                "Correction",
                ["Before: Φ(index) was passed through Φ again.", "One positive Mills ratio was assigned to both groups.", "That is not the probit generalized residual."],
                [("Input", "latent index η"), ("Groups", "signed T=0 / T=1")],
            ),
            slide(
                "Stable two-group residual",
                "Implemented",
                ["T=1:  φ(η) / Φ(η)", "T=0: −φ(η) / Φ(−η)", "Log-CDF arithmetic remains finite in extreme tails."],
                [("η minimum", f"{result['selection_index_min']:.3f}"), ("η maximum", f"{result['selection_index_max']:.3f}")],
            ),
            slide(
                "Actual bounded execution",
                "Receipt",
                ["Every notebook code cell executed in order.", "Figures and JSON were emitted from committed inputs.", "The 2,000-row lane is explicitly smoke-only."],
                [("Tobacco share", f"{100*result['tobacco_share']:.2f}%"), ("Cells", str(len(receipt["executed_cells"])))],
            ),
            slide(
                "Regression coverage",
                "Verified",
                ["Tail finiteness and treatment-group signs", "Invalid-input rejection and matching parity", "Real notebook process with JSON receipt"],
                [("Unit + E2E", f"{test_count} passed"), ("Process", "exit 0")],
            ),
            slide(
                "What this result does—and does not—say",
                "Research boundary",
                ["A reproducible coursework control-function exercise.", "Strong normality, exclusion, and model assumptions remain.", "No validated causal, clinical, policy, or forecast claim."],
                [("Dataset", "committed"), ("Mode", "bounded smoke")],
            ),
        ]

        OUTPUT.mkdir(parents=True, exist_ok=True)
        frames = Path(directory) / "frames"
        frames.mkdir()
        fps = 10
        hold = 22
        fade = 6
        sequence = []
        for index, current in enumerate(slides):
            sequence.extend([current] * hold)
            if index + 1 < len(slides):
                nxt = slides[index + 1]
                sequence.extend(Image.blend(current, nxt, step / (fade + 1)) for step in range(1, fade + 1))
        for index, frame in enumerate(sequence):
            frame.save(frames / f"frame-{index:04d}.png")
        slides[0].save(OUTPUT / "poster.png")
        command(
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
            "-i", str(frames / "frame-%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(OUTPUT / "advanced-econometrics-iii-demo.mp4"),
        )
        command(
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
            "-i", str(frames / "frame-%04d.png"), "-vf",
            "fps=8,scale=800:-2:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=64[p];[s1][p]paletteuse=dither=bayer",
            str(OUTPUT / "advanced-econometrics-iii-demo.gif"),
        )

    artifacts = {}
    for name in ["advanced-econometrics-iii-demo.gif", "advanced-econometrics-iii-demo.mp4", "poster.png"]:
        path = OUTPUT / name
        artifacts[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {
        "recorded_source_sha": args.source_sha,
        "recorded_source_tree": git("rev-parse", f"{args.source_sha}^{{tree}}"),
        "runtime_input_blobs": {path: git("rev-parse", f"{args.source_sha}:{path}") for path in RUNTIME_INPUTS},
        "workflow": f"RESEARCH_MAX_ROWS={args.max_rows} python run_research.py econometrics_III.ipynb",
        "tests": "python -m unittest discover -s tests -v",
        "receipt": receipt,
        "artifacts": artifacts,
        "scope": "Committed birthweight data; deterministic bounded coursework replay. No causal, clinical, policy, forecast, or performance claim.",
    }
    (OUTPUT / "capture.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
