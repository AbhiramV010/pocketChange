# pocketchange

Python trading bot for a self-custody **Solana** wallet (~$20 USDC). Pipeline: **backtest, paper trade, live**. Trades via Jupiter, keeps state in SQLite, and has a Streamlit dashboard with a pause switch (separate process).

## Run (Windows, Python 3.11+)

```powershell
python launch.py
```

Sets up `.venv` and `.env` (from `.env.example`), checks the price feed, starts bot + dashboard (`http://localhost:8501`) and restarts crashes. Ctrl+C finishes the current cycle first. Flags: `--no-browser`, `--port N`, `--check`, `--yes`. Dry run (default) needs no key or funds.

By hand: `python -m bot.data` (candles), `bot.backtest`, `bot.portfolio_backtest [--ml] [--grid]`, `bot.run [--once]`, `streamlit run dashboard/app.py`, `python -m pytest`. Also `bot.reset_halt` (only way to clear a halt), `bot.chain new-wallet|check|swap buy 1.0`, `bot.ml train|eval`.

## Config (`.env`)

`DRY_RUN` (default `true`), `PRIVATE_KEY` (live only), `MAX_TRADE_USDC` (5), `MAX_SLIPPAGE_BPS` (100), `MIN_EQUITY_USD` (15; bot halts below it), `GAS_RESERVE_SOL` (0.01), `STRATEGY` (`sma`/`ml`/`dca`), `ASSETS`, `RISKY_ASSETS`, `MAX_POSITIONS`, `TIMEFRAME` (4h). Strategy defaults are the best backtested settings: `SMA_FAST`/`SMA_SLOW` 20/50, `ENTRY=cross`, `EDGE_BPS=10`, `TREND_CONFIRM=true`, `STOP_ATR=3`, `HARD_STOP_PCT=10`, `REGIME_LEN=400` (BTC filter), `RISK_PCT=1` (size each buy so a stop-out costs ~1% of equity). All options are in `.env.example` and `bot/config.py`.

## Backtests

`bot.portfolio_backtest` replays the live loop: the same decision and sizing functions, in the same order (exits, then entries best-ranked first), filled at the next candle's open with 0.3% fee + 0.1% slippage per side. Tests run the real live loop candle by candle and require identical fills (SMA and AI). `--ml` scores the AI only on candles its model never saw (walk-forward, retrained every 30 days, labels embargoed). `--grid` runs a grid of settings on each half of the period: an edge that only shows in one half is curve fitting.

The AI's labels are net of the 0.8% round trip: a "win" is a trade that keeps `ML_TARGET_PCT` after costs. `ML_ADAPT` moves the buy threshold only when the bot's own graded calls prove it (independent windows, mean two standard errors above zero).

## Safety

- The key is read only in `bot/chain.py` and never logged.
- Jupiter transactions are untrusted: the bot must be sole signer, and each is simulated first (input, slippage floor and fee checked) or nothing is sent.
- Trade size, slippage and gas reserve are hard limits; the pause flag is checked before every swap.
- Equity below `MIN_EQUITY_USD`, or a sent swap with unknown outcome, sets a persistent halt. Only `bot.reset_halt` clears it.
- The dashboard imports no trading code and writes only validated pause, timeframe and RISKY flags.

## Results (Oct 2026; past results, not a promise)

SOL, BTC, ETH, BNB, XRP, DOGE from Binance; $20 start, $5 cap per trade, up to 6 positions, 0.4% cost per side.

| Test | Strategy | Return | Max DD | Sharpe | Trades |
|---|---|---|---|---|---|
| 4h, Dec 2023 – Oct 2026 | Buy & hold (equal weight) | +13% | 64% | 0.37 | 6 |
| | Default rules | **+56%** | **16%** | **1.26** | 112 |
| | Default rules, costs doubled | +44% | 20% | 1.05 | 112 |
| 4h, first half / second half | Buy & hold | +55% / −24% | 49% / 65% | | |
| | Default rules (best first-half Sharpe of the grid) | +63% / −3% | 11% / 7% | 2.04 / −0.29 | 100 |
| 4h AI, out of sample, Apr 2024 – Oct 2026 | Buy & hold | +15% | 64% | 0.40 | 6 |
| | AI, `ML_ENTER` 0.55 / 0.6, default stops and filters | +11.5% / +8.6% | 8% / 6% | 0.67 / 0.59 | 338 / 174 |
| 1h, Oct 2024 – Oct 2026 | Default rules | −3% | 17% | −0.15 | 132 |
| 15m, Mar – Oct 2026 | Every setting tried | −0.2% to −96% | | | |

What this does and does not show:
- On 4h the rules beat buy & hold with a quarter of its drawdown, and nearby settings (10/30, 2–4 ATR stops, `REGIME_LEN=200`, 3 positions, +AVAX, +memecoins) were positive too. Only 30/80 lost (−2%).
- The edge is trend following: it made its money in the 2024–25 rally and mostly sat out the 2025–26 decline (−3% while the coins fell 24%). It is long-only, so a long bear market means flat at best.
- Below 4h, every setting lost after costs; trend mode did worse than cross mode almost everywhere. The dashboard still offers short timeframes, at your own risk.
- One 2.8-year history, $5 positions, simulated fills. Run the dry run first and fund only what you can lose.

## Before going live

1. Run dry run for a few days.
2. `python -m bot.chain new-wallet`; **back up `.env`**. Never reuse your main wallet.
3. Fund ~$17-18 USDC and ~0.03 SOL; set `MIN_EQUITY_USD` to suit.
4. `python -m bot.chain check`, then `DRY_RUN=false` and one swap under $2 (`swap buy 1.0`, check on Solscan, sell back).
5. Start `python -m bot.run`.

## Layout

`bot/` (config, db, assets, data, strategy, ml, backtest, sniper, chain, run), `dashboard/app.py`, `tests/`, `launch.py`, `data/` (gitignored).
