# pocketchange

Python trading bot for a self-custody **Solana** wallet (~$20 USDC). Pipeline: **backtest, paper trade, live**. Trades via Jupiter, keeps state in SQLite, and has a Streamlit dashboard with a pause switch (separate process).

## Run (Windows, Python 3.11+)

```powershell
python launch.py
```

Sets up `.venv` and `.env` (from `.env.example`), checks the price feed, starts bot + dashboard (`http://localhost:8501`) and restarts crashes. Ctrl+C finishes the current cycle first. Flags: `--no-browser`, `--port N`, `--check`, `--yes`. Dry run (default) needs no key or funds.

By hand: `python -m bot.data` (candles), `bot.backtest`, `bot.run [--once]`, `streamlit run dashboard/app.py`, `python -m pytest`. Also `bot.reset_halt` (only way to clear a halt), `bot.chain new-wallet|check|swap buy 1.0`, `bot.ml train|eval`.

## Config (`.env`)

`DRY_RUN` (default `true`), `PRIVATE_KEY` (live only), `MAX_TRADE_USDC` (5), `MAX_SLIPPAGE_BPS` (100), `MIN_EQUITY_USD` (15; bot halts below it), `GAS_RESERVE_SOL` (0.01), `STRATEGY` (`sma`/`ml`/`dca`), `ASSETS`, `RISKY_ASSETS`, `MAX_POSITIONS`, `TIMEFRAME`, `SMA_FAST`/`SMA_SLOW`, `REGIME_LEN`. All options are in `.env.example` and `bot/config.py`.

## Safety

- The key is read only in `bot/chain.py` and never logged.
- Jupiter transactions are untrusted: the bot must be sole signer, and each is simulated first (input, slippage floor and fee checked) or nothing is sent.
- Trade size, slippage and gas reserve are hard limits; the pause flag is checked before every swap.
- Equity below `MIN_EQUITY_USD`, or a sent swap with unknown outcome, sets a persistent halt. Only `bot.reset_halt` clears it.
- The dashboard imports no trading code and writes only validated pause, timeframe and RISKY flags.

## Results (no proven edge)

1-year SOL backtest ($20, 0.4% cost/trade): buy & hold -40%, SMA 20/50 -28%. Multi-coin and ML (`STRATEGY=ml`) variants also lost after fees; a BTC trend filter (`REGIME_LEN=400`) helped most, mostly by trading less. Treat everything as an experiment; do not fund a wallet expecting profit.

## Before going live

1. Run dry run for a few days.
2. `python -m bot.chain new-wallet`; **back up `.env`**. Never reuse your main wallet.
3. Fund ~$17-18 USDC and ~0.03 SOL; set `MIN_EQUITY_USD` to suit.
4. `python -m bot.chain check`, then `DRY_RUN=false` and one swap under $2 (`swap buy 1.0`, check on Solscan, sell back).
5. Start `python -m bot.run`.

## Layout

`bot/` (config, db, assets, data, strategy, ml, backtest, sniper, chain, run), `dashboard/app.py`, `tests/`, `launch.py`, `data/` (gitignored).
