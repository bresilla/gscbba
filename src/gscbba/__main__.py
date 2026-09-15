from .experiments import main as run_experiments


def main(argv: list[str] | None = None) -> int:
    run_experiments(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
