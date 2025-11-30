import subprocess
import sys

def install(package, extra_args=None):
    args = [sys.executable, "-m", "pip", "install", package]
    if extra_args:
        args.extend(extra_args)
    subprocess.check_call(args)

BASE_PACKAGES = [
    "pip==24.0",
    "wheel",
    "setuptools",
    "numpy",
    "matplotlib",
    "pillow",
    "sympy",
    "fastapi",
    "uvicorn[standard]",
    "python-multipart",
    "torchvision",
]

TORCH_INSTALL = [
    "torch",
    "torchvision",
    "torchaudio",
]

if __name__ == "__main__":
    for pkg in BASE_PACKAGES:
        install(pkg)
    install(
        TORCH_INSTALL[0],
        ["--index-url", "https://download.pytorch.org/whl/cu121"],
    )
    install(
        TORCH_INSTALL[1],
        ["--index-url", "https://download.pytorch.org/whl/cu121"],
    )
    install(
        TORCH_INSTALL[2],
        ["--index-url", "https://download.pytorch.org/whl/cu121"],
    )
