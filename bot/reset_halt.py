"""Clear the persistent halt flag: python -m bot.reset_halt

This is the only way to clear a halt. The dashboard cannot do it.
"""
from __future__ import annotations

import argparse

from bot import db
from bot.config import settings


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Clear the bot's halted flag.")
    parser.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    args = parser.parse_args(argv)

    if not settings.db_path.exists():
        print(f"No database at {settings.db_path}; nothing to clear.")
        return
    conn = db.connect(settings.db_path)
    db.init_db(conn)
    if not db.is_halted(conn):
        print("The bot is not halted.")
        return
    print(f"Halted since {db.get_state(conn, db.HALTED_AT, 'unknown')}: {db.halt_reason(conn)}")
    print(
        f"If equity is still below MIN_EQUITY_USD ({settings.min_equity_usd:g}), the bot will halt again on its next cycle."
    )
    if not args.yes and input("Clear the halt? [y/N] ").strip().lower() not in ("y", "yes"):
        print("Left halted.")
        return
    db.clear_halt(conn)
    print("Halt cleared. A running bot picks this up on its next cycle.")


if __name__ == "__main__":
    main()
