# pocketchange

Python trading bot for a self-custody **Solana** wallet (~$20 USDC). Pipeline: **backtest, paper trade, live**. Trades via Jupiter, keeps state in SQLite, and has a Streamlit dashboard with a pause switch (separate process).

## Run (Windows, Python 3.11+)

```powershell
python launch.py
```

Sets up `.venv` and `.env` (from `.env.example`), checks the price feed, starts bot + dashboard (`http://localhost:8501`) and restarts crashes. Ctrl+C finishes the current cycle first. Flags: `--no-browser`, `--port N`, `--check`, `--yes`. Dry run (default) needs no key or funds.

By hand: `python -m bot.data` (candles), `bot.backtest`, `bot.portfolio_backtest [--ml] [--grid]`, `bot.run [--once]`, `streamlit run dashboard/app.py`, `python -m pytest`. Also `bot.reset_halt` (only way to clear a halt), `bot.chain new-wallet|check|swap buy 1.0`, `bot.ml train|eval`.

## Config (`.env`)

`DRY_RUN` (default `true`), `PRIVATE_KEY` (live only), `MAX_TRADE_USDC` (5), `MAX_SLIPPAGE_BPS` (100), `MIN_EQUITY_USD` (15; bot halts below it), `GAS_RESERVE_SOL` (0.01), `STRATEGY` (`sma`/`ml`/`dca`), `ASSETS`, `RISKY_ASSETS`, `MAX_POSITIONS`, `TIMEFRAME` (4h). Strategy defaults are the best backtested settings: `SMA_FAST`/`SMA_SLOW` 20/50, `ENTRY=cross`, `EDGE_BPS=0`, `TREND_CONFIRM=true`, `STOP_ATR=3`, `HARD_STOP_PCT=10`, `REGIME_LEN=400` (BTC filter), `RISK_PCT=1` (size each buy so a stop-out costs ~1% of equity). All options are in `.env.example` and `bot/config.py`. Values in your `.env` override these defaults, so an `.env` copied from an older `.env.example` keeps the old settings until you edit it.

While BTC is below its `REGIME_LEN`-candle average the bot opens no new positions (exits and stops still run). In a falling market that can mean days or weeks without a trade; the dashboard shows the reason while it lasts.

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

SOL, BTC, ETH, BNB, XRP, DOGE from OKX USDT pairs (a close proxy for the USDC prices the bot trades); $20 start, $5 cap per trade, up to 6 positions, 0.4% cost per side.

| Test | Strategy | Return | Max DD | Sharpe | Trades |
|---|---|---|---|---|---|
| 4h, Dec 2023 – Oct 2026 | Buy & hold (equal weight) | +81% | 64% | 0.64 | 6 |
| | Default rules | **+124%** | **21%** | **1.55** | 242 |
| | Default rules, costs doubled | +99% | 23% | 1.33 | 240 |
| 4h, first half / second half (split May 2025) | Buy & hold | +166% / −29% | 50% / 64% | | |
| | Default rules | +136% / +6% | 21% / 10% | 2.44 / 0.38 | 208 |
| 4h AI, out of sample, Jun 2024 – Oct 2026 | Buy & hold | +9% | 64% | 0.38 | 6 |
| | Default rules, same period | +94% | 18% | 1.64 | 192 |
| | AI, `ML_ENTER` 0.55 / 0.6, default stops and filters | −1% / +8% | 11% / 5% | −0.04 / 0.58 | 450 / 222 |
| 1h, Oct 2024 – Oct 2026 | Buy & hold | +17% | 65% | 0.45 | 6 |
| | Default rules | −42% | 50% | −1.33 | 635 |
| | 9/21 trend, the old `.env.example` | −80% | 83% | −2.48 | 1861 |
| 15m, Mar – Oct 2026 | Buy & hold | +8% | 31% | 0.51 | 6 |
| | Default rules / 9/21 trend | −61% / −95% | 62% / 95% | | 784 / 1390 |

What this does and does not show:
- On 4h the rules beat buy & hold with a third of its drawdown, and every nearby setting tried was positive: 10/30, 30/80, 2 and 4 ATR stops, `REGIME_LEN=200`, no BTC filter, 3 positions, `EDGE_BPS=10`, no trend confirmation (+48% to +143%).
- The edge is trend following: it made its money in the 2024–25 rally and stayed slightly positive through the 2025–26 decline (+6% while the coins fell 29%). It is long-only, so a long bear market means flat at best.
- The AI separates winners from losers only a little (out-of-sample AUC 0.55) and did far worse than the plain rules over the same period.
- Below 4h every setting lost after costs, and trend mode lost most. The dashboard still offers short timeframes, at your own risk; the bot logs a warning when it starts on one.
- One 2.9-year history, $5 positions, simulated fills. Run the dry run first and fund only what you can lose.

## Before going live

1. Run dry run for a few days.
2. `python -m bot.chain new-wallet`; **back up `.env`**. Never reuse your main wallet.
3. Fund ~$17-18 USDC and ~0.03 SOL; set `MIN_EQUITY_USD` to suit.
4. `python -m bot.chain check`, then `DRY_RUN=false` and one swap under $2 (`swap buy 1.0`, check on Solscan, sell back).
5. Start `python -m bot.run`.

## Layout

`bot/` (config, db, assets, data, strategy, ml, backtest, sniper, chain, run), `dashboard/app.py`, `tests/`, `launch.py`, `data/` (gitignored).
