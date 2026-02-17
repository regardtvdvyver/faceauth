"""Entry point for `python -m faceauth` - runs the daemon."""

import argparse

from .daemon import run_daemon


def main():
    parser = argparse.ArgumentParser(prog="faceauth", description="faceauth daemon")
    parser.add_argument("--system", action="store_true", help="Run in system mode")
    args = parser.parse_args()
    run_daemon(system=args.system)


if __name__ == "__main__":
    main()
