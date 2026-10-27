import argparse

import uvicorn


def main() -> None:
    ap = argparse.ArgumentParser(description="Driver Fatigue Monitor API server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    uvicorn.run("fatigue_api.app:create_app", factory=True, host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":
    main()
