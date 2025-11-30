import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "project")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

BUILD_CMD = [
    sys.executable,
    os.path.join(SRC_DIR, "build_dataset.py"),
    "--geometry", "10000",
    "--charts", "10000",
    "--ocr", "10000",
    "--lines", "10000",
    "--text", "10000",
]

TRAIN_CMD = [
    sys.executable,
    os.path.join(SRC_DIR, "train.py"),
    "--manifest", os.path.join(PROJECT_ROOT, "dataset", "annotations", "dataset_manifest.jsonl"),
    "--text", os.path.join(PROJECT_ROOT, "dataset", "text", "math_text.jsonl"),
    "--epochs", "1000000000",
    "--batch_size", "16",
    "--model_file", os.path.join(BASE_DIR, "models", "active.mathai"),
    "--save_every", "10",
]

def run(cmd):
    print("=" * 80)
    print("Executing:", " ".join(cmd))
    print("Working directory:", BASE_DIR)
    print("=" * 80)
    subprocess.check_call(cmd, cwd=BASE_DIR)

if __name__ == "__main__":
    print("Starting easytrain pipeline")
    run(BUILD_CMD)
    print("Dataset generation completed.")
    print("Starting long-running training...")
    run(TRAIN_CMD)
