import argparse

import uvicorn

from gateway.app import build_app
from gateway.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Local LLM forwarding gateway")
    parser.add_argument(
        "-c", "--config", default="config.json",
        help="path to JSON config file (default: config.json)",
    )
    parser.add_argument(
        "--debug-log", action="store_true",
        help="log the full request content (overrides server.debug_log)",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.debug_log:
        config.debug_log = True
    uvicorn.run(build_app(config, config_path=args.config), host=config.host, port=config.port)


if __name__ == "__main__":
    main()
