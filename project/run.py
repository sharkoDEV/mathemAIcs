import os
import sys

import uvicorn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)


def str_to_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


def main():
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload_flag = str_to_bool(os.getenv("UVICORN_RELOAD", "false"))
    uvicorn.run("api:app", host=host, port=port, reload=reload_flag)


if __name__ == "__main__":
    main()
