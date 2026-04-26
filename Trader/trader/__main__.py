from __future__ import annotations

import sys

from trader.cli import eval_cmd, ingest_cmd, ledger_cmd, loop_cmd, trades_cmd, visualize_cmd
from trader.backtest import main as backtest_main


def main() -> None:
    if len(sys.argv) == 1:
        command = "ingest"
        argv: list[str] = []
    elif sys.argv[1] in {"-h", "--help", "help"}:
        print("Usage: python -m trader <command> [options]")
        print("")
        print("Commands:")
        print("  ingest     Fetch aggregate bars from Massive and store them in SQLite")
        print("  visualize  Render a self-contained HTML chart from the local SQLite data")
        print("  trades     Render a trade/equity HTML review for one experiment")
        print("  backtest   Run the compatibility EMA CLI through the frozen execution engine")
        print("  eval       Evaluate a StrategySpec through the fixed evaluator")
        print("  ledger     Inspect research ledger summaries")
        print("  loop       Run the deterministic autonomous research loop")
        return
    else:
        command = sys.argv[1]
        argv = sys.argv[2:]

    if command == "ingest":
        ingest_cmd.main(argv)
        return
    if command == "visualize":
        visualize_cmd.main(argv)
        return
    if command == "trades":
        trades_cmd.main(argv)
        return
    if command == "backtest":
        backtest_main(argv)
        return
    if command == "eval":
        eval_cmd.main(argv)
        return
    if command == "ledger":
        ledger_cmd.main(argv)
        return
    if command == "loop":
        loop_cmd.main(argv)
        return
    raise SystemExit(
        f"Unknown command: {command}. Use 'ingest', 'visualize', 'trades', 'backtest', 'eval', 'ledger', or 'loop'."
    )


if __name__ == "__main__":
    main()
