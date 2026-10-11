import os
import sys
from pathlib import Path

os.environ["TIMEFRAME"] = "1h"  # tests assume hourly candles, whatever .env says

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot import db  # noqa: E402


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(tmp_path / "test.db")
    db.init_db(connection)
    yield connection
    connection.close()
