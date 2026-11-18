"""Start the Labels page: python -m pool_segmentation.app, then open http://127.0.0.1:8001 (docs/labeling.md)."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m pool_segmentation.app",
                                     description="Check and fix the pool labels of the frames picked for training.")
    parser.add_argument("--labels", default="data/labels/pool", help="Labeling dataset (default: data/labels/pool)")
    parser.add_argument("--port", type=int, default=8001, help="Port to listen on (default: 8001)")
    args = parser.parse_args()

    import uvicorn

    from pool_segmentation.app.server import create_app

    print(f"Labels page: http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    # Local only: listen on 127.0.0.1, not on the network
    uvicorn.run(create_app(args.labels), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
