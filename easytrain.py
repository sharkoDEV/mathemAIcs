import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")

BUILD_CMD = [
    sys.executable,
    "build_dataset.py",
    "--geometry", "10000",
    "--charts", "10000",
    "--ocr", "10000",
    "--lines", "10000",
    "--text", "10000",
]

TRAIN_CMD = [
    sys.executable,
    "train.py",
    "--manifest", os.path.join("..", "dataset", "annotations", "dataset_manifest.jsonl"),
    "--text", os.path.join("..", "dataset", "text", "math_text.jsonl"),
    "--epochs", "1000000000",
    "--batch_size", "16",
    "--model_file", os.path.join("..", "models", "active.mathai"),
    "--save_every", "10",
]

def run(cmd):
    print("Executing:", " ".join(cmd))
    cwd = SRC_DIR if cmd[1] in {"build_dataset.py", "train.py"} else BASE_DIR
    subprocess.check_call(cmd, cwd=cwd)

if __name__ == "__main__":
    run(BUILD_CMD)
    run(TRAIN_CMD)
