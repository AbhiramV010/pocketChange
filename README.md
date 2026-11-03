# pocketchange

Python trading bot for a self-custody **Solana** wallet (~$20 USDC). Pipeline: **backtest, paper trade, live**. Trades via Jupiter, keeps state in SQLite, and has a Streamlit dashboard with a pause switch (separate process).

## Run (Windows, Python 3.11+)

```powershell
python launch.py
```

Sets up `.venv` and `.env` (from `.env.example`), checks the price feed, starts bot + dashboard (`http://localhost:8501`) and restarts crashes. Ctrl+C finishes the current cycle first. Flags: `--no-browser`, `--no-dashboard`, `--port N`, `--check`, `--yes`. Dry run (default) needs no key or funds.

## Always on (Raspberry Pi or Linux server)

On Raspberry Pi OS 64-bit (Bookworm), Ubuntu 24.04 or Debian 12:

```bash
curl -fsSLO https://raw.githubusercontent.com/AbhiramV010/pocketChange/main/deploy/setup.sh
sudo bash setup.sh --tailscale        # or --no-dashboard, --tunnel, --public; --branch NAME for another branch
```

Installs into `/opt/pocketchange` and runs `launch.py` as the `pocketchange` systemd service (restarts on crash and reboot). Adds swap under 2 GB of RAM. On a first install it sets `LOOP_INTERVAL_SECONDS=300` and `SNIPER_INTERVAL_SECONDS=60`: the strategy trades 4h candles, so a cycle a minute only spends CPU and network. `--tailscale` serves the dashboard to your own devices only (disable key expiry for the machine in the Tailscale console). `--public` makes a dev-tunnel link anyone can open and sets `DASHBOARD_PUBLIC=true` (read-only). Re-run to update. Logs: `journalctl -u pocketchange -f`. After editing `/opt/pocketchange/.env`: `sudo systemctl restart pocketchange`.

By hand: `python -m bot.data` (candles), `bot.backtest`, `bot.portfolio_backtest [--ml] [--grid]`, `bot.run [--once]`, `streamlit run dashboard/app.py`, `python -m pytest`. Also `bot.reset_halt` (only way to clear a halt), `bot.chain new-wallet|check|swap buy 1.0`, `bot.ml train|eval`.

## Config (`.env`)

`DRY_RUN` (default `true`), `PRIVATE_KEY` (live only), `MAX_TRADE_USDC` (5), `MAX_SLIPPAGE_BPS` (100), `MIN_EQUITY_USD` (15; bot halts below it), `GAS_RESERVE_SOL` (0.01), `STRATEGY` (`sma`/`ml`/`dca`), `ASSETS`, `RISKY_ASSETS`, `MAX_POSITIONS` (3), `TIMEFRAME` (4h). Strategy defaults are the best backtested settings: `SMA_FAST`/`SMA_SLOW` 20/50, `ENTRY=cross`, `EDGE_BPS=0`, `TREND_CONFIRM=true`, `STOP_ATR=3`, `HARD_STOP_PCT=10`, `REGIME_LEN=400` (BTC filter), `RISK_PCT=1` (size each buy so a stop-out costs ~1% of equity). All options are in `.env.example` and `bot/config.py`. Values in your `.env` override these defaults, so an `.env` copied from an older `.env.example` keeps the old settings until you edit it.

`ENTRY=cross` buys when the fast average crosses above the slow one. A crossover the bot could not take at the time (BTC filter off, every slot full, no cash) is taken on a later candle while the trend holds, once per cross and within `SMA_SLOW` candles.

While BTC is below its `REGIME_LEN`-candle average the bot opens no new positions (exits and stops still run). In a falling market that can mean days or weeks without a trade; the dashboard shows the reason while it lasts.

## Sniper (simulated)

With RISKY on in a dry run, a sniper buys brand-new Solana tokens with its own bankroll (`SNIPER_BANKROLL_USDC`) and never touches the wallet. Prices are real Jupiter quotes; each leg also pays a latency haircut (`SNIPER_LATENCY_SLIPPAGE`, 2%) and a Solana network fee (`SNIPER_TX_FEE_SOL`, 0.0001 SOL) at the bot's SOL price. It exits at +50%, -20%, after 30 minutes, or at zero when a token can no longer be sold.

It learns from its own closed snipes (`python -m bot.sniper_learn` prints what it knows). Six entry filters (liquidity, holders, top-holder share, dev mints, token age, price impact) each have a grid of stricter settings; one is adopted only when 30+ closed snipes pass it and their mean return, minus a Bonferroni-corrected margin for the number of settings tried, beats what the sniper already earns. Learning only tightens filters, never loosens them below `.env`. Exits and the fill model are not learned, and quotes are not fills: treat its results as optimistic.

## Backtests

`bot.portfolio_backtest` replays the live loop: the same decision and sizing functions, in the same order (exits, then entries best-ranked first), filled at the next candle's open with 0.3% fee + 0.1% slippage per side. Tests run the real live loop candle by candle and require identical fills (SMA and AI). `--ml` scores the AI only on candles its model never saw (walk-forward, retrained every 30 days, labels embargoed). `--grid` runs a grid of settings on each half of the period: an edge that only shows in one half is curve fitting.

The AI's labels are net of the 0.8% round trip: a "win" is a trade that keeps `ML_TARGET_PCT` after costs. `ML_ADAPT` moves the buy threshold only when the bot's own graded calls prove it (independent windows, mean two standard errors above zero).

## Safety

- The key is read only in `bot/chain.py` and never logged.
- Jupiter transactions are untrusted: the bot must be sole signer, and each is simulated first (input, slippage floor and fee checked) or nothing is sent.
- Trade size, slippage and gas reserve are hard limits; the pause flag is checked before every swap.
- Equity below `MIN_EQUITY_USD`, or a sent swap with unknown outcome, sets a persistent halt. Only `bot.reset_halt` clears it.
- The dashboard imports no trading code and writes only validated pause, timeframe and RISKY flags.

## Results (Nov 2026; past results, not a promise)

OKX USDT pairs (a close proxy for the USDC prices the bot trades); $20 start, $5 cap per trade, up to 3 positions (the default), 0.4% cost per side. The settings were chosen on 2023–26 (SOL, BTC, ETH, BNB, XRP, DOGE). 2021–23 is a holdout that played no part in choosing them (OKX has no BNB history that far back, so it has the other five coins).

| Test | Strategy | Return | Max DD | Sharpe | Trades |
|---|---|---|---|---|---|
| 4h, Dec 2023 – Oct 2026 | Buy & hold (equal weight) | +81% | 64% | 0.64 | 6 |
| | Default rules | **+128%** | **21%** | **1.68** | 265 |
| | Default rules, costs doubled | +98% | 23% | 1.39 | 266 |
| 4h, first half / second half (split May 2025) | Buy & hold | +166% / −29% | 50% / 64% | | |
| | Default rules | +141% / +6% | 21% / 12% | 2.64 / 0.37 | 237 |
| 4h holdout, Jan 2021 – Dec 2023 | Buy & hold (equal weight) | +1141% | 90% | 1.30 | 5 |
| | Default rules | **+54%** | **16%** | **1.02** | 206 |
| | Default rules, costs doubled | +37% | 18% | 0.76 | 206 |
| 4h holdout, first half / second half (split Jun 2022) | Buy & hold | +582% / +84% | 86% / 42% | | |
| | Default rules | +21% / +29% | 12% / 13% | 0.99 / 1.12 | 195 |
| 4h AI, out of sample, May 2024 – Oct 2026 | Buy & hold | +9% | 64% | 0.38 | 6 |
| | Default rules, same period | +81% | 15% | 1.59 | 214 |
| | AI, `ML_ENTER` 0.55 / 0.6, default stops and filters | −6% / +4% | 9% / 4% | −0.41 / 0.39 | 352 / 182 |
| 1h, Oct 2024 – Oct 2026 | Buy & hold | +17% | 65% | 0.45 | 6 |
| | Default rules | −35% | 47% | −0.96 | 753 |
| | 9/21 trend, the old `.env.example` | −77% | 80% | −2.36 | 1725 |
| 15m, Mar – Oct 2026 | Buy & hold | +8% | 31% | 0.51 | 6 |
| | Default rules / 9/21 trend | −83% / −95% | 84% / 95% | | 954 / 1338 |

What this does and does not show:
- The rules make money with small drawdowns in both periods, including the one never used to pick them. On 2023–26 they also beat buy & hold. On 2021–23 they did not: buy & hold made +1141% through a 90% drawdown, and a bot that risks about 1% per trade with $5 positions cannot ride a mania like that. What carries over is the risk control.
- Every nearby setting tried was positive in both periods: 10/30, 30/80, 2 and 4 ATR stops, `REGIME_LEN=200`, no BTC filter, 6 positions, `EDGE_BPS=10`, no trend confirmation, `RISK_PCT=2` (+28% to +140% on 2023–26, +18% to +121% on 2021–23). Some did better than the defaults on 2021–23 (`REGIME_LEN=200`: Sharpe 1.42); switching to them now would be tuning on the holdout.
- Taking a missed crossover later was decided before testing: it raised the 2023–26 Sharpe from 1.58 to 1.68 and kept 2021–23 at 1.02 with more return (+54% against +41%).
- The edge is trend following and long-only: a long bear market means flat at best.
- The AI separates winners from losers only a little (out-of-sample AUC 0.55) and did far worse than the plain rules over the same period.
- Below 4h every setting lost after costs, the shorter the worse, and trend mode lost most. The dashboard warns while a shorter timeframe is selected.
- Two histories, $5 positions, simulated fills. Run the dry run first and fund only what you can lose.

## Before going live

1. Run dry run for a few days.
2. `python -m bot.chain new-wallet`; **back up `.env`**. Never reuse your main wallet.
3. Fund ~$17-18 USDC and ~0.03 SOL; set `MIN_EQUITY_USD` to suit.
4. `python -m bot.chain check`, then `DRY_RUN=false` and one swap under $2 (`swap buy 1.0`, check on Solscan, sell back).
5. Start `python -m bot.run`.

## Layout

`bot/` (config, db, assets, data, strategy, ml, backtest, portfolio_backtest, sniper, sniper_learn, chain, run), `dashboard/app.py`, `deploy/setup.sh`, `tests/`, `launch.py`, `data/` (gitignored).
