def main() -> None:
    import sys

    if len(sys.argv) == 1:
        command = "ingest"
        argv: list[str] = []
    elif sys.argv[1] in {"-h", "--help", "help"}:
        print("Usage: python -m Trader <command> [options]")
        print("")
        print("Commands:")
        print("  ingest     Fetch aggregate bars from Massive and store them in SQLite")
        print("  visualize  Render a self-contained HTML chart from the local SQLite data")
        print("  backtest   Run a no-lookahead single-position strategy on local SQLite data")
        return
    else:
        command = sys.argv[1]
        argv = sys.argv[2:]

    if command == "ingest":
        try:
            from Trader.ingest import parse_args, run_ingest
        except ModuleNotFoundError:
            from ingest import parse_args, run_ingest

        settings, request = parse_args(argv)
        run_ingest(settings, request)
        return

    if command == "visualize":
        try:
            from Trader.visualize import parse_args, run_visualize
        except ModuleNotFoundError:
            from visualize import parse_args, run_visualize

        request = parse_args(argv)
        run_visualize(request)
        return

    if command == "backtest":
        try:
            from Trader.backtest import parse_args, run_backtest
        except ModuleNotFoundError:
            from backtest import parse_args, run_backtest

        request = parse_args(argv)
        run_backtest(request)
        return

    raise SystemExit(f"Unknown command: {command}. Use 'ingest', 'visualize', or 'backtest'.")


if __name__ == "__main__":
    main()
