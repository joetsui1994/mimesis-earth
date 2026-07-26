"""Command-line entry point: `mimesis-earth serve`."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="mimesis-earth")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="run the web app + API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.command == "serve":
        import uvicorn

        from mimesis_earth.server import app

        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
