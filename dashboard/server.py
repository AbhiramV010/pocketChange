"""pocketchange trading terminal: a small API server plus a static frontend over the bot's SQLite state.

It imports no trading code and writes only the three
validated flags (pause, timeframe, RISKY). Halts can be seen here but only `bot.reset_halt` clears them.

    python -m dashboard.server --port 8501     # then open http://localhost:8501
"""
from __future__ import annotations

import argparse
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from bot import db
from bot.config import Settings, settings
from dashboard import view

WEB_DIR = Path(__file__).resolve().parent / "web"


class PauseBody(BaseModel):
    paused: bool


class RiskyBody(BaseModel):
    risky: bool


class TimeframeBody(BaseModel):
    timeframe: str


def _no_database(cfg: Settings) -> dict:
    if cfg.dashboard_public:
        return {"ready": False, "public": True, "message": "The bot has not started yet."}
    return {
        "ready": False, "public": False,
        "message": f"No database yet at `{cfg.db_path}`. Start the bot with `python -m bot.run`.",
    }


def create_app(cfg: Settings) -> FastAPI:
    app = FastAPI(title="pocketchange", docs_url=None, redoc_url=None, openapi_url=None)

    def read_only():
        return closing(db.connect(cfg.db_path, readonly=True))

    def now() -> datetime:
        return datetime.now(timezone.utc)

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.get("/")
    def index():
        from fastapi.responses import FileResponse

        return FileResponse(WEB_DIR / "index.html")

    def _state_payload() -> dict:
        if not cfg.db_path.exists():
            return _no_database(cfg)
        try:
            with read_only() as conn:
                return view.state(conn, cfg, now())
        except sqlite3.OperationalError:
            return {
                "ready": False, "public": cfg.dashboard_public,
                "message": "Waiting for the bot to initialise its database.",
            }

    @app.get("/api/state")
    def api_state():
        return _state_payload()

    @app.get("/api/series")
    def api_series(range: str = Query(default=view.DEFAULT_RANGE, pattern="^(24h|7d|30d|All)$")):
        if not cfg.db_path.exists():
            return {"points": [], "floor": cfg.min_equity_usd, "sniper_active": False}
        with read_only() as conn:
            return view.series(conn, cfg, range, now())

    @app.get("/api/trades")
    def api_trades(limit: int = Query(default=100, ge=1, le=1000)):
        if not cfg.db_path.exists():
            return {"rows": [], "total": 0}
        with read_only() as conn:
            return view.trades(conn, cfg, limit)

    @app.get("/api/candles")
    def api_candles(limit: int = Query(default=400, ge=10, le=5000)):
        return view.candles(cfg, limit)

    # --- the three writes ------------------------------------------------------------------

    def _writable() -> None:
        if cfg.dashboard_public:
            raise HTTPException(status_code=403, detail="this is a read-only public view")

    @app.post("/api/pause")
    def api_pause(body: Annotated[PauseBody, Body()]):
        _writable()
        with closing(db.connect(cfg.db_path)) as conn:
            db.set_paused(conn, bool(body.paused))
        return {"ok": True}

    @app.post("/api/risky")
    def api_risky(body: Annotated[RiskyBody, Body()]):
        _writable()
        with closing(db.connect(cfg.db_path)) as conn:
            db.set_risky(conn, bool(body.risky))
        return {"ok": True}

    @app.post("/api/timeframe")
    def api_timeframe(body: Annotated[TimeframeBody, Body()]):
        _writable()
        if not cfg.db_path.exists():
            raise HTTPException(status_code=400, detail="no database yet — start the bot first")
        with closing(db.connect(cfg.db_path, readonly=True)) as conn:
            if body.timeframe not in view.timeframe_options(conn):
                raise HTTPException(status_code=400, detail="timeframe must be one of the dashboard's choices")
        with closing(db.connect(cfg.db_path)) as conn:
            try:
                db.set_timeframe(conn, body.timeframe)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from None
        return {"ok": True}

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app


app = create_app(settings)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="pocketchange trading terminal")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
