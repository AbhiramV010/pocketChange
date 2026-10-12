import os
import sys
from pathlib import Path

# Tests assume the default strategy and a SOL-only wallet, whatever .env says.
os.environ.update(
    STRATEGY="sma", TIMEFRAME="1h", SMA_FAST="20", SMA_SLOW="50", ENTRY="cross", EDGE_BPS="0", TREND_CONFIRM="false",
    STOP_ATR="0", HARD_STOP_PCT="0", REENTRY_CANDLES="0", REGIME_LEN="0", ASSETS="SOL", RISKY_ASSETS="", RISKY="false", LOOP_INTERVAL_SECONDS="60",
)

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot import db  # noqa: E402


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(tmp_path / "test.db")
    db.init_db(connection)
    yield connection
    connection.close()
