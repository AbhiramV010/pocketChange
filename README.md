# pocketchange

A small Python trading bot for a self-custody **Solana** wallet holding about $20 of USDC. The goal is a working, safe pipeline: **backtest, then paper trade, then live**. Profit is a bonus.

It trades USDC and SOL through the Jupiter aggregator, keeps all state in SQLite, and has a Streamlit dashboard with a pause switch. The bot and the dashboard are separate processes.

## Setup (Windows)

Needs Python 3.11+.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # cmd.exe: .venv\Scripts\activate.bat
pip install -r requirements.txt
Copy-Item .env.example .env         # cmd.exe: copy .env.example .env
```

`.env` is gitignored. In dry run (the default) you need no key and no funded wallet.

## Quick start

```powershell
python launch.py
```

That's it, with any Python 3.11+. `launch.py`:

1. Creates `.venv` and installs the requirements, re-installing only when `requirements.txt` changes.
2. Creates `.env` from `.env.example` and validates it.
3. In live mode, asks you to type `live` and checks the wallet.
4. Checks the price feed and initialises the database.
5. Starts the bot and the dashboard, and opens `http://localhost:8501`.

While running, it restarts either process if it crashes, and gives up after 5 crashes in 10 minutes. Ctrl+C stops everything, and the bot always finishes its current cycle first, so a swap is never cut off. Only one bot can run per database: a second copy refuses to start. Options: `--no-browser`, `--port N`, `--check` (pre-flight only), `--yes` (skip the live prompt).

## Run the pieces by hand

```powershell
python -m bot.data                  # download 1h SOL/USDC candles to data/candles.csv (incremental)
python -m bot.backtest              # compare buy & hold, DCA and SMA crossover
```

Then run two separate processes, each in its own terminal with the venv active:

```powershell
python -m bot.run                   # terminal 1: the trading loop
streamlit run dashboard/app.py      # terminal 2: the dashboard (http://localhost:8501)
```

Other commands:

| Command | What it does |
| --- | --- |
| `python -m bot.run --once` | run one cycle and exit |
| `python -m bot.reset_halt` | clear the halted flag (the only way to clear it) |
| `python -m bot.chain new-wallet` | create a dedicated bot wallet; the key goes into `.env` and only the address is printed |
| `python -m bot.chain check` | read-only: verify cluster and mints, show balances and quotes (`--address <pubkey>` needs no key) |
| `python -m bot.chain swap buy 1.0` | **real** swap (USDC in) for the first live test; needs `DRY_RUN=false` and asks to confirm |
| `python -m pytest` | run the test suite |

Bot logs go to the console and `data/bot.log`. State lives in `data/pocketchange.db`.

## Environment variables

Set these in `.env` (see `.env.example`). Real environment variables override `.env`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `PRIVATE_KEY` | none | Wallet secret key, as a base58 string or a JSON byte array. Only needed when `DRY_RUN=false`, and only read by `bot/chain.py`. `new-wallet` fills it in. |
| `SOLANA_RPC_URL` | `https://api.mainnet-beta.solana.com` | Solana mainnet RPC. The genesis hash is checked. |
| `DRY_RUN` | `true` | `true`: signals and simulated trades only, nothing on-chain. Only `false`/`0`/`no`/`off` turns it off; anything unparseable is an error. |
| `MAX_TRADE_USDC` | `5` | Hard cap on the USD size of every trade, buy or sell. |
| `MAX_SLIPPAGE_BPS` | `100` | Slippage passed to Jupiter and checked against the simulation (100 = 1%). |
| `MIN_EQUITY_USD` | `15` | Equity floor; below it the bot halts. Set it from your real starting balance (see below). |
| `GAS_RESERVE_SOL` | `0.01` | Native SOL the bot never trades. It also refuses to trade if the wallet has less. |
| `STRATEGY` | `sma` | `sma` (the moving-average signal below), `ml` (the AI) or `dca`. |
| `ML_HORIZON` / `ML_TARGET_PCT` | `6` / `0.4` | What the AI predicts: will the coin gain more than this percent over the next N candles? |
| `ML_ENTER` / `ML_EXIT` | `0.6` / `0.45` | Buy when the AI's probability is at least `ML_ENTER`; sell when it falls to `ML_EXIT` or below. |
| `ML_RETRAIN_HOURS` / `ML_TRAIN_DAYS` | `24` / `400` | The AI retrains itself on fresh candles this often, using this much history. |
| `ASSETS` | `SOL` | Coins traded all the time, each with its own signal and position: `SOL`, `BTC`, `ETH`, `BNB`, `XRP`, `DOGE`, `AVAX`. SOL is always included. `.env.example` turns on `SOL,BTC,ETH,BNB,XRP,DOGE`. |
| `RISKY_ASSETS` | none | Memecoins traded only while the yellow RISKY switch is on: `BONK`, `WIF`, `TRUMP`, `PENGU`. |
| `RISKY` | `false` | Starting state of the RISKY switch. The dashboard switch overrides it once clicked. |
| `MAX_POSITIONS` | `3` | Most coins held at once. When several want in, the strongest momentum goes first. |
| `TIMEFRAME` | `1h` | Candle size: `1m`, `5m`, `15m`, `30m`, `1h`, `4h`. The dashboard can switch it live. |
| `SMA_FAST` / `SMA_SLOW` / `ENTRY` | `20` / `50` / `cross` | Moving-average periods, and `trend` (long whenever fast > slow) or `cross` (act only on the crossover). |
| `EDGE_BPS` / `TREND_CONFIRM` | `0` / `false` | Entry filters: fast must beat slow by this many bps; and the slow average must be rising with price above it. |
| `STOP_ATR` / `HARD_STOP_PCT` | `0` / `0` | Exits: trailing stop N ATRs under the highest close since entry; and a stop this % under the entry price. `0` is off. |
| `REENTRY_CANDLES` | `0` | Candles to wait after a sell before buying that coin again. |
| `REGIME_LEN` | `0` | Open new positions only while BTC is above its moving average of this many candles. Exits never wait. `0` is off. |
| `SNIPER` and `SNIPER_*` | on | Settings for the simulated new-token sniper (see below). Dry run only. |
| `DCA_AMOUNT_USDC` / `DCA_INTERVAL_HOURS` | `1` / `24` | DCA buy size (still capped by `MAX_TRADE_USDC`) and interval. |
| `LOOP_INTERVAL_SECONDS` | `60` | Seconds between cycles. Pause and halt take effect within one cycle. |
| `SIM_START_USDC` / `SIM_START_SOL` | `20` / `0.05` | Starting balances for the dry-run simulator. |
| `MAX_QUOTE_DEVIATION_BPS` | `300` | Refuse a swap if the Jupiter quote is this far from the exchange price. |
| `MAX_FEE_USD` | `0.5` | Refuse a swap whose simulated SOL cost (fees plus any new account rent) exceeds this. |
| `JUPITER_API_URL` | `https://lite-api.jup.ag/swap/v1` | Jupiter swap API (keyless tier). |
| `CANDLE_EXCHANGE` | `binance` | ccxt exchange id for candles (public endpoints). `bybit`, `okx` and `kraken` are tried as fallbacks. |

## Design

### Swap venue: Jupiter

On Solana, SOL/USDC liquidity is split across many AMMs and order books. Jupiter routes across them and has a keyless API, so it gets better prices than any single pool. The cost is that Jupiter builds the transaction, so the bot treats it as untrusted:

- The bot's wallet must be the fee payer and the only signer.
- Before sending, the transaction is **simulated on current chain state**. The simulated balance changes must show:
  - no more input spent than requested,
  - output at or above the slippage floor,
  - SOL cost at or below `MAX_FEE_USD`.

  If any check fails, nothing is sent.
- Priority fees are capped at 0.0002 SOL.
- **Confirmation.** A Solana transaction can only land before its blockhash expires (~60–90 s). The bot watches until the transaction lands, fails, or expires. "Expired" is a definite "never executed", so the swap can be retried safely.

The full swap path (quote, build, parse, simulate and check) was run against real mainnet state for both a buy and a sell, with signing and sending stubbed out, so no funds were spent. A real-money test is still yours to run; see the checklist below.

### Position: wrapped SOL

The bot holds its position as **wSOL**, an SPL token, rather than native SOL (`wrapAndUnwrapSol=false`). Native SOL only pays fees and rent and is never traded, so the gas reserve can't be swapped away. Equity counts USDC and wSOL only.

### Tokens

The bot checks the cluster's genesis hash and both mints (owner program and decimals) before first use.

| Token | Mint | Decimals |
| --- | --- | --- |
| USDC (Circle) | `EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v` | 6 |
| wSOL (native mint) | `So11111111111111111111111111111111111111112` | 9 |
| wBTC, wETH, wBNB, wXRP (Wormhole) | see `bot/assets.py` | 8, 8, 8, 6 |
| DOGE, AVAX | see `bot/assets.py` | 8, 9 |
| BONK, WIF, TRUMP, PENGU (risky) | see `bot/assets.py` | 5, 6, 6, 6 |

Every mint in `bot/assets.py` was checked on-chain and quoted through Jupiter both ways for $5 (the round trip came back within 0.4%). Two caveats. **wXRP** carries a freeze authority (the bridge could freeze balances), and its Solana price sat about 6% above the exchange price when checked, so the quote-deviation guard (`MAX_QUOTE_DEVIATION_BPS`) will refuse live wXRP buys until the gap closes; in a dry run it fills at the exchange price. **Dry-run fills** use the exchange price plus a fixed 0.3% fee and 0.1% slippage, which is pessimistic for the big coins (their measured round trips cost 0.01% to 0.35%).
### Strategies (`bot/strategy.py`, pure functions)

- **Moving-average trend** (`STRATEGY=sma`): per coin, on closed candles only. Long while the fast average is above the slow one; flat when it falls below. Filters make it pickier: `EDGE_BPS` (fast must beat slow by a margin), `TREND_CONFIRM` (slow average rising, price above it), `STOP_ATR` and `HARD_STOP_PCT` (cut a position that turns against you), `REENTRY_CANDLES` (no instant re-buy after a sell) and `REGIME_LEN` (open positions only while BTC is above its long average). With every filter off it is the plain 20/50 crossover. A sell larger than `MAX_TRADE_USDC` is worked off one capped chunk per cycle.
- **DCA**: a fixed buy every `DCA_INTERVAL_HOURS`. It never sells.

The signal rules are array functions shared by the live loop and the backtest, so they cannot drift apart. Each cycle the bot plans every coin first, runs the sells (to free cash), then the buys in order of momentum, until `MAX_POSITIONS` or the cash runs out. Each coin keeps its own "already acted on this candle" state.

### The AI (`STRATEGY=ml`, `bot/ml.py`)

A gradient-boosted classifier (scikit-learn) trained on about a year of hourly candles for every coin at once. For each coin and closed candle it computes 29 features (returns over 1 to 48 candles, volatility, ATR, distance from moving averages, RSI, volume, position in the 24-candle range, time of day and week, and BTC's own trend as market context) and outputs the probability that the coin gains more than `ML_TARGET_PCT` over the next `ML_HORIZON` candles. The bot buys the highest probabilities first, once they reach `ML_ENTER`, and sells when the probability falls to `ML_EXIT` or after the horizon passes without the AI still liking it. Everything else is unchanged: the $5 cap, `MAX_POSITIONS`, ATR and hard stops, the equity floor, STOP and RISKY.

- `python -m bot.ml train` fits the model on all history and saves `data/models/ml_1h.joblib`. If a model is missing or older than `ML_RETRAIN_HOURS`, the running bot retrains itself in a background thread and swaps the new model in; a failed retrain keeps the old one. There is one model per timeframe, so switching the dashboard to a timeframe with no model means no trades until one is trained.
- `python -m bot.ml eval` runs the **walk-forward test**: for each 30-day block after the first 180 days, it trains only on data from before the block (with a gap equal to the horizon, so no label peeks into the block) and scores the block. A test checks that predictions for a block do not change if later data is removed.
- The dashboard shows each coin's current probability, and when the model was trained and how it scored out of sample.

**What the AI achieved out of sample.** Eight monthly blocks (Feb to Oct 2026), 7 coins traded, hourly candles, 0.1% cost per side, buying at 0.6+ and selling at 0.45 or below. "Win" means the coin gains more than the target over the next N candles.

| What the AI predicts | Out-of-sample AUC | Pure AI | With BTC filter + stops |
| --- | --- | --- | --- |
| +0.4% in 6 candles | 0.533 | -13.5% (138 trades) | -2.6% (38) |
| +0.8% in 12 | 0.517 | -16.8% (132) | -2.9% (42) |
| +1.2% in 24 | 0.496 | -9.4% (76) | -3.0% (34) |
| +2.0% in 24 | 0.531 | -5.8% (40) | -0.8% (4) |
| +2.0% in 48 | 0.494 | -14.4% (98) | -3.6% (36) |
| +3.0% in 48 | 0.523 | -3.2% (48) | -0.8% (12) |

Equal-weight buy and hold made **+15.3%** over the same window. An AUC of 0.5 is a coin flip, so 0.53 means the model sees *something*, but not enough to pay for fees: **every variant lost money after costs**. The BTC filter (`REGIME_LEN=400`) and the stops cut the losses a lot, mostly by trading much less. The defaults (6-candle horizon, `ML_ENTER=0.6`) are the setting with the best AUC that still trades often enough to watch, **not a profitable one**. Treat `STRATEGY=ml` as an experiment to observe in a dry run; do not fund a live wallet on it. Add `REGIME_LEN=400` to `.env` if you want it to behave more cautiously.

### Coins and the RISKY switch

The yellow **RISKY** switch beside STOP on the dashboard turns on memecoin trading. Off (the default), only `ASSETS` trade. On:

- the coins in `RISKY_ASSETS` (BONK, WIF, TRUMP, PENGU) join the same strategy, with the same caps and checks;
- in a dry run, a **simulated sniper** also scans Jupiter's list of brand-new tokens every 30 seconds. It keeps its own $10 bankroll in a separate table, and never touches the wallet or the equity floor. Entries need: mint and freeze authority revoked, liquidity of at least $3k, at least 5 holders, age under 15 minutes, no serial-launcher deployer, no concentrated top holders, a buy price impact under 3%, and a sell quote that returns at least 85% of the cost. Exits at +50%, -20%, 30 minutes, or when the sell route disappears (a rug, written off at zero). Both legs take a 2% haircut for being late;
- turning RISKY off stops new memecoin entries, but open positions still exit on their signals, so nothing is stranded.

The sniper is **simulation only** by design: `bot.run` never starts it unless `DRY_RUN=true`. A real new-token sniper needs per-token swap simulation for honeypots and transfer fees, and a poll every 30 seconds is always late against bots that act within a slot.

### Safety rails

| Rule | Where |
| --- | --- |
| `PRIVATE_KEY` is read only in `bot/chain.py`, from the environment, and is never logged. Errors are scrubbed of the key and the RPC URL (which often carries an API key). | `chain.py`; a test fails if any other module mentions it |
| `DRY_RUN` defaults to true. Dry run uses a simulated wallet and sends nothing. `ChainClient.swap` also refuses to run while `DRY_RUN` is true. | `config.py`, `run.py`, `chain.py` |
| `MAX_TRADE_USDC`, `MAX_SLIPPAGE_BPS` and `GAS_RESERVE_SOL` are hard limits. They are enforced when sizing and again in `chain.py`. | `config.py`, `strategy.py`, `chain.py` |
| The dashboard never touches the key and imports no trading code. It writes only three state keys, all validated: the pause flag (STOP/RESUME), the candle timeframe (1m to 1h switch) and the RISKY switch; the bot picks them up next cycle. The timeframe and RISKY switches are deliberate exceptions to the original "pause flag only" rule. Everything else is read over a read-only connection. | `dashboard/app.py`, tested |
| The pause flag is checked at the top of every cycle and again right before a swap. A paused bot keeps logging balances. | `run.py` |
| RPC and API errors are logged, then the loop sleeps and continues. A failed swap is retried at most once per cycle, and only if it definitely did not execute. | `run.py`, `chain.py` |
| Equity floor: equity = USDC + every position at its latest price (a held coin with no price this cycle uses its last known mark), excluding native SOL. It is checked at the start of every cycle and right after every trade, in dry run too. Below `MIN_EQUITY_USD` the bot sets a persistent `halted` flag and stops trading. It does not sell the open position. | `run.py` |

A halt survives restarts. Only `python -m bot.reset_halt` clears it; the dashboard cannot. If equity is still below the floor, the bot halts again on the next cycle.

The bot also sets the same halt if a transaction was sent but its outcome is still unknown after 2 minutes (for example, the RPC is down). Retrying then could double-spend, so it waits for you: check the wallet on Solscan, then run `bot.reset_halt`.

### Database (`bot/db.py`)

- `trades(timestamp, side, amount_in, amount_out, price, tx_hash, simulated, asset)` plus an `id`. A buy spends USDC for the coin; a sell spends the coin for USDC. `price` is USDC per unit of the coin, and `tx_hash` is the Solana signature.
- `balances(timestamp, usdc, sol, wsol)` plus `id`, `price` (the SOL/USD mark), `holdings` (JSON of every other coin's quantity and price, used for the equity curve) and `simulated`.
- `snipes(...)`: the simulated sniper's positions, with its own bankroll.
- `state(key, value)`: `paused`, `halted`, `halt_reason`, `dry_run`, `heartbeat`, `risky`, plus per-coin bookkeeping.

Older databases are migrated in place when the bot starts.

The dry-run wallet is `SIM_START_*` replayed through the simulated trades, using the same 0.3% fee and 0.1% slippage as the backtest.

## Backtest

`python -m bot.backtest` on one year of 1h SOL/USDC candles (2025-10-11 to 2026-10-11), starting with $20, charging 0.3% fee plus 0.1% slippage per trade, and filling at the next candle's open:

```
Strategy             Return    Max DD  Trades   Exposure   Final $
------------------------------------------------------------------
Buy & hold           -40.3%    -70.9%       1       100%     11.94
DCA $1/24h           -43.1%    -70.1%      20        97%     11.38
SMA 20/50            -27.8%    -34.1%     201        16%     14.44
SMA 20/50 all-in     -71.0%    -75.0%     203        48%      5.80
```

How to read it:

- SOL fell about 40% over this window, and **neither strategy beat buy & hold at the same exposure**.
- The SMA strategy loses to costs: about 200 trades at roughly 0.4% each. "SMA 20/50" only looks better than buy & hold because the $5 cap keeps it mostly in cash.
- DCA at $1/day spends the $20 in 20 days, so it behaves like a delayed buy & hold.
- This is one year of one asset, not a forecast. Use `--days N` to look at other windows.

### Multi-coin backtest (`python -m bot.portfolio_backtest`)

45 days of Binance 5m candles for SOL, BTC, ETH, BNB, XRP, DOGE and AVAX, starting with $20, $5 per position, up to 3 positions, shared cash. The first 30 days were for choosing settings and the last 15 for checking them. This is a short window, so read it as a warning about fees rather than proof of an edge. Equal-weight buy and hold made +12% then -5%.

| Timeframe | Cost per side | Best SMA trend result (train / test) |
| --- | --- | --- |
| 5m | 0.3% fee + 0.1% slippage | -58% / -28% |
| 5m | 0.1% | -32% / -15% |
| 15m | 0.1% | -10% / -10% (-6% / -2% with the BTC filter) |
| 1h | 0.1% | -0.3% / -10% (+3.9% / flat with the BTC filter) |

- **Fast trading loses to fees.** The faster the moving averages and the shorter the candles, the worse it did, in every variant. The 3/8 setting lost 95% on 5m.
- **What helped most** was refusing to open positions while BTC was below its long average (`REGIME_LEN`): at 1h it turned -10% into "stayed flat" on the weak last 15 days. Trailing stops, hard stops and re-entry waits changed little.
- The settings in `.env.example` (1h, 9/21 trend, 10 bps edge, confirmation, 3 ATR stop, BTC filter over 400 candles) are those results. They are the least-bad found, **not a proven edge**, and in a sideways or falling market they simply sit out. Choosing a faster timeframe on the dashboard means accepting the losses above.

## Before going live

Neither strategy has a proven edge (see the backtests above). The checklist below makes live trading *safe* to try with money you can lose; it does not make it profitable.

Dry run is implemented and tested, but it has not yet run for several days, and no real transaction has been sent. To finish:

1. Let `python -m bot.run` run in dry run for a few days and watch the dashboard.
2. Run `python -m bot.chain new-wallet` to create a wallet used only by the bot. **Back up `.env`**: losing it means losing what's in that wallet. Never use your main wallet's recovery phrase.
3. Fund the new address with about $17–18 of **USDC on Solana** and about 0.02–0.03 SOL for fees.
4. Set `MIN_EQUITY_USD` to suit your start. With about $17.5 of USDC, the default of 15 halts after a ~14% drop; 13 allows about 25%.
5. Run `python -m bot.chain check`. It should show your balances, with quotes within a few bps of the exchange price.
6. Set `DRY_RUN=false`, then do one swap under $2 by hand with `python -m bot.chain swap buy 1.0`. Check it on Solscan, then sell back with `python -m bot.chain swap sell <wsol_amount>`. The first buy creates the wSOL account, which costs about 0.002 SOL in refundable rent.
7. Only then start `python -m bot.run` with `DRY_RUN=false`.

## Layout

```
bot/
  config.py      settings and hard limits (.env)
  db.py          SQLite schema and helpers
  assets.py      the coins: mints, decimals, exchange markets, core vs risky
  data.py        candle download and CSV cache
  strategy.py    moving-average signal, stops, BTC filter, trade sizing (pure functions)
  ml.py          the AI: features, training, walk-forward test, live model management
  backtest.py    single-coin comparison table
  portfolio_backtest.py  multi-coin backtest of the live strategy
  sniper.py      simulated new-token sniper (dry run only)
  chain.py       balances, Jupiter quotes and swaps (the only place the key is read)
  run.py         the trading loop (with a dry-run simulator)
  reset_halt.py  python -m bot.reset_halt
dashboard/app.py Streamlit dashboard
tests/           pytest suite
launch.py        one-command start: venv, checks, bot + dashboard, supervision
data/            gitignored: SQLite file, candle cache, logs
```
