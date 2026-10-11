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
| `STRATEGY` | `sma` | `sma` (20/50 crossover on 1h closes) or `dca`. |
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

### Strategies (`bot/strategy.py`, pure functions)

- **SMA crossover**: BUY on the candle where the 20-period SMA crosses above the 50-period, SELL on the cross below. It uses closed candles only and does not stack positions. A sell larger than `MAX_TRADE_USDC` is worked off one capped chunk per cycle.
- **DCA**: a fixed buy every `DCA_INTERVAL_HOURS`. It never sells.

The backtest and the bot call the same strategy and sizing functions.

### Safety rails

| Rule | Where |
| --- | --- |
| `PRIVATE_KEY` is read only in `bot/chain.py`, from the environment, and is never logged. Errors are scrubbed of the key and the RPC URL (which often carries an API key). | `chain.py`; a test fails if any other module mentions it |
| `DRY_RUN` defaults to true. Dry run uses a simulated wallet and sends nothing. `ChainClient.swap` also refuses to run while `DRY_RUN` is true. | `config.py`, `run.py`, `chain.py` |
| `MAX_TRADE_USDC`, `MAX_SLIPPAGE_BPS` and `GAS_RESERVE_SOL` are hard limits. They are enforced when sizing and again in `chain.py`. | `config.py`, `strategy.py`, `chain.py` |
| The dashboard never touches the key and imports no trading code. It writes only the pause flag and reads everything else over a read-only connection. | `dashboard/app.py`, tested |
| The pause flag is checked at the top of every cycle and again right before a swap. A paused bot keeps logging balances. | `run.py` |
| RPC and API errors are logged, then the loop sleeps and continues. A failed swap is retried at most once per cycle, and only if it definitely did not execute. | `run.py`, `chain.py` |
| Equity floor: equity = USDC + wSOL at the latest price, excluding native SOL. It is checked at the start of every cycle and right after every trade, in dry run too. Below `MIN_EQUITY_USD` the bot sets a persistent `halted` flag and stops trading. It does not sell the open position. | `run.py` |

A halt survives restarts. Only `python -m bot.reset_halt` clears it; the dashboard cannot. If equity is still below the floor, the bot halts again on the next cycle.

The bot also sets the same halt if a transaction was sent but its outcome is still unknown after 2 minutes (for example, the RPC is down). Retrying then could double-spend, so it waits for you: check the wallet on Solscan, then run `bot.reset_halt`.

### Database (`bot/db.py`)

- `trades(timestamp, side, amount_in, amount_out, price, tx_hash, simulated)` plus an `id`. A buy spends USDC for wSOL; a sell spends wSOL for USDC. `price` is USDC per SOL, and `tx_hash` is the Solana signature.
- `balances(timestamp, usdc, sol, wsol)` plus `id`, `price` (the SOL/USD mark, used for the equity curve) and `simulated`.
- `state(key, value)`: `paused`, `halted`, `halt_reason`, `dry_run`, `heartbeat`, plus bookkeeping.

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

## Before going live

Dry run is implemented and tested, but it has not yet run for several days, and no real transaction has been sent. To finish:

1. Let `python -m bot.run` run in dry run for a few days and watch the dashboard.
2. Run `python -m bot.chain new-wallet` to create a wallet used only by the bot. **Back up `.env`**: losing it means losing what's in that wallet. Never use your main wallet's recovery phrase.
3. Fund the new address with about $17–18 of **USDC on Solana** and about 0.02–0.03 SOL for fees.
4. Set `MIN_EQUITY_USD` to suit your start. With about $17.5 of USDC, the default of 15 halts after a ~14% drop; 13 allows about 25%.
5. Run `python -m bot.chain check`. It should show your balances, with quotes within a few bps of the exchange price.
6. Set `DRY_RUN=false`, then do one swap under $2 by hand with `python -m bot.chain swap buy 1.0`. Check it on Solscan, then sell back with `python -m bot.chain swap sell <wsol_amount>`. The first buy creates the wSOL account, which costs about 0.002 SOL in refundable rent.
7. Only then start `python -m bot.run` with `DRY_RUN=false`.

## Running on Google Cloud

Not yet run against a real VM (Google Cloud requires a billing account). `deploy/deploy.py` puts everything on one e2-micro VM (Debian 12, in the free tier in `us-central1`). Run it from Windows after installing the Google Cloud CLI and running `gcloud init` with billing enabled on the project:

```powershell
python deploy/deploy.py up        # create the VM, upload code + .env, install, start services, print the public URL
python deploy/deploy.py push      # after changing code or .env: re-upload and restart
python deploy/deploy.py url       # current public URL
python deploy/deploy.py status    # are the services running?
python deploy/deploy.py logs      # last bot log lines
python deploy/deploy.py private   # full dashboard (with pause) at http://localhost:8501 over SSH
```

The VM runs four systemd services, each restarting on failure and at boot:

- **The bot.**
- **A private dashboard** on `127.0.0.1:8501`, reachable only through `deploy.py private`.
- **A public read-only dashboard** with `DASHBOARD_PUBLIC=true`: no pause control, never writes, and no transaction links, since those would reveal the wallet.
- **A Cloudflare quick tunnel** that gives the public dashboard an HTTPS `*.trycloudflare.com` URL with no open ports. That URL changes when the tunnel restarts; a permanent one needs a domain on Cloudflare.

Notes:

- The VM sets `CANDLE_EXCHANGE=kraken`, because Binance blocks US IP addresses.
- `.env` is uploaded with permissions `600`.
- **Cost:** the e2-micro and its 30 GB standard disk are in the free tier. The VM's external IPv4 address and any network traffic beyond the free allowance may be billed (a few dollars a month at most), so check the billing page after the first week.

## Layout

```
bot/
  config.py      settings and hard limits (.env)
  db.py          SQLite schema and helpers
  data.py        candle download and CSV cache
  strategy.py    DCA and SMA crossover, trade sizing (pure functions)
  backtest.py    comparison table
  chain.py       balances, Jupiter quotes and swaps (the only place the key is read)
  run.py         the trading loop (with a dry-run simulator)
  reset_halt.py  python -m bot.reset_halt
dashboard/app.py Streamlit dashboard
tests/           pytest suite
launch.py        one-command start: venv, checks, bot + dashboard, supervision
deploy/          Google Cloud VM deployment (untested on a real VM)
data/            gitignored: SQLite file, candle cache, logs
```
