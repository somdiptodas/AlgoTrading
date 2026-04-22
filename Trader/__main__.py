from Trader.ingest import parse_args, run_ingest


def main() -> None:
    settings, request = parse_args()
    run_ingest(settings, request)


if __name__ == "__main__":
    main()
