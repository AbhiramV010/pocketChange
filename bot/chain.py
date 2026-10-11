from __future__ import annotations

import argparse
import base64
import functools
import json
import logging
import os
import time
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from pathlib import Path
from urllib.parse import urlparse

import requests
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.transaction import VersionedTransaction

from bot.config import ROOT, Settings, settings
from bot.db import Balances, Fill

log = logging.getLogger(__name__)

# Mints were checked on-chain (owner program, decimals); the bot re-checks the cluster and
# both mints' decimals before first use.
MAINNET_GENESIS = "5eykt4UsFv8P8NJdTREpY1vzqKqZKvdpKuc147dw2N9d"
USDC_MINT = Pubkey.from_string("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v")  # Circle USDC
WSOL_MINT = Pubkey.from_string("So11111111111111111111111111111111111111112")  # wrapped SOL (native mint)
TOKEN_PROGRAM = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
ATA_PROGRAM = Pubkey.from_string("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL")
USDC_DECIMALS, SOL_DECIMALS = 6, 9

CONFIRM_TIMEOUT_S = 120  # longer than a blockhash lives (~60-90s), so we normally see landed or expired
MAX_PRIORITY_LAMPORTS = 200_000  # priority fee ceiling handed to Jupiter (0.0002 SOL)
HTTP_TIMEOUT_S = 20

BUY, SELL = "buy", "sell"


class ChainError(Exception):
    pass


class SwapFailed(ChainError):
    pass


class SwapUncertain(ChainError):
    pass


class RpcError(ChainError):
    pass


@dataclass(frozen=True)
class Quote:
    side: str
    amount_in_raw: int
    amount_out_raw: int
    min_out_raw: int  # Jupiter's otherAmountThreshold: output after max slippage
    route: str
    response: dict  # passed back to Jupiter unchanged to build the transaction

    @property
    def price(self) -> float:
        usdc_raw, sol_raw = (
            (self.amount_in_raw, self.amount_out_raw) if self.side == BUY else (self.amount_out_raw, self.amount_in_raw)
        )
        return (usdc_raw / 10**USDC_DECIMALS) / (sol_raw / 10**SOL_DECIMALS)


@dataclass(frozen=True)
class Snapshot:

    lamports: int
    usdc: int
    wsol: int


def to_raw(amount: float | Decimal | str, decimals: int) -> int:
    return int((Decimal(str(amount)) * 10**decimals).to_integral_value(rounding=ROUND_DOWN))


def from_raw(raw: int, decimals: int) -> float:
    return float(Decimal(raw) / 10**decimals)


def min_amount_out(amount_out_raw: int, slippage_bps: int) -> int:
    return amount_out_raw * (10_000 - slippage_bps) // 10_000


def ata(owner: Pubkey, mint: Pubkey) -> Pubkey:
    return Pubkey.find_program_address([bytes(owner), bytes(TOKEN_PROGRAM), bytes(mint)], ATA_PROGRAM)[0]


def token_amount(account: dict | None) -> int:
    if not account:
        return 0
    data = base64.b64decode(account["data"][0])
    return int.from_bytes(data[64:72], "little")  # layout: mint(32) owner(32) amount(u64)


def _mints(side: str) -> tuple[Pubkey, Pubkey, int, int]:
    if side == BUY:
        return USDC_MINT, WSOL_MINT, USDC_DECIMALS, SOL_DECIMALS
    if side == SELL:
        return WSOL_MINT, USDC_MINT, SOL_DECIMALS, USDC_DECIMALS
    raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")


def _load_keypair() -> Keypair:
    raw = os.environ.get("PRIVATE_KEY", "").strip()
    if not raw:
        raise ChainError("PRIVATE_KEY is not set (required when DRY_RUN=false)")
    try:
        if raw.startswith("["):  # solana-keygen JSON file contents
            return Keypair.from_bytes(bytes(json.loads(raw)))
        return Keypair.from_base58_string(raw)  # Phantom / Solflare export
    except Exception:
        raise ChainError("PRIVATE_KEY is not a valid Solana secret key") from None


def _guard(fn):

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except ChainError:
            raise
        except Exception as exc:
            raise ChainError(f"{fn.__name__} failed: {self._scrub(f'{type(exc).__name__}: {exc}')}") from None

    return wrapper


class ChainClient:

    def __init__(self, cfg: Settings = settings, *, address: str | None = None, http: requests.Session | None = None):
        self._cfg = cfg
        self._http = http or requests.Session()
        keypair = None if address else _load_keypair()
        self._keypair = keypair
        self.owner = Pubkey.from_string(address) if address else keypair.pubkey()
        self.address = str(self.owner)
        self.usdc_account = ata(self.owner, USDC_MINT)
        self.wsol_account = ata(self.owner, WSOL_MINT)
        self._verified = False  # cluster/mint check runs on first use, so a flaky RPC at startup can be retried

    def __repr__(self) -> str:
        return f"ChainClient(address={self.address})"

    def _scrub(self, text: str) -> str:
        secrets = []
        parsed = urlparse(self._cfg.solana_rpc_url)
        secrets += [self._cfg.solana_rpc_url, parsed.path.strip("/"), parsed.query]
        if self._keypair is not None:
            secrets += [str(self._keypair), bytes(self._keypair).hex()]
        for secret in sorted((s for s in secrets if len(s) > 3), key=len, reverse=True):
            text = text.replace(secret, "<redacted>")
        return text

    def _rpc(self, method: str, params: list | None = None):
        resp = self._http.post(
            self._cfg.solana_rpc_url,
            json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params or []},
            timeout=HTTP_TIMEOUT_S,
        )
        resp.raise_for_status()
        body = resp.json()
        if "error" in body:
            raise RpcError(f"{method}: {self._scrub(json.dumps(body['error']))[:300]}")
        return body["result"]

    def _jupiter(self, path: str, params: dict | None = None, body: dict | None = None) -> dict:
        url = f"{self._cfg.jupiter_api_url}/{path}"
        if body is None:
            resp = self._http.get(url, params=params, timeout=HTTP_TIMEOUT_S)
        else:
            resp = self._http.post(url, json=body, timeout=HTTP_TIMEOUT_S)
        if resp.status_code != 200:
            raise ChainError(f"Jupiter {path} returned HTTP {resp.status_code}: {resp.text[:200]}")
        return resp.json()

    def _ensure_verified(self) -> None:
        if self._verified:
            return
        if self._rpc("getGenesisHash") != MAINNET_GENESIS:
            raise ChainError("SOLANA_RPC_URL is not Solana mainnet-beta")
        for label, mint, decimals in (("USDC", USDC_MINT, USDC_DECIMALS), ("wSOL", WSOL_MINT, SOL_DECIMALS)):
            info = self._rpc("getAccountInfo", [str(mint), {"encoding": "jsonParsed"}])["value"]
            parsed = (info or {}).get("data", {}).get("parsed", {})
            if info is None or info["owner"] != str(TOKEN_PROGRAM) or parsed.get("info", {}).get("decimals") != decimals:
                raise ChainError(f"{label} mint does not look right on this cluster")
        self._verified = True

    def _snapshot(self) -> Snapshot:
        accounts = self._rpc(
            "getMultipleAccounts",
            [
                [self.address, str(self.usdc_account), str(self.wsol_account)],
                {"encoding": "base64", "commitment": "confirmed"},
            ],
        )["value"]
        owner, usdc, wsol = accounts
        return Snapshot((owner or {}).get("lamports", 0), token_amount(usdc), token_amount(wsol))

    @_guard
    def balances(self) -> Balances:
        self._ensure_verified()
        snap = self._snapshot()
        return Balances(
            from_raw(snap.usdc, USDC_DECIMALS), from_raw(snap.lamports, SOL_DECIMALS), from_raw(snap.wsol, SOL_DECIMALS)
        )

    @_guard
    def quote(self, side: str, amount_in_raw: int) -> Quote:
        mint_in, mint_out, _, _ = _mints(side)
        q = self._jupiter(
            "quote",
            params={
                "inputMint": str(mint_in),
                "outputMint": str(mint_out),
                "amount": str(amount_in_raw),
                "slippageBps": str(self._cfg.max_slippage_bps),
                "swapMode": "ExactIn",
                "restrictIntermediateTokens": "true",
            },
        )
        if (q.get("inputMint"), q.get("outputMint"), int(q.get("inAmount", -1))) != (
            str(mint_in), str(mint_out), amount_in_raw,
        ):
            raise ChainError("Jupiter quote does not match the request")
        out, min_out = int(q["outAmount"]), int(q["otherAmountThreshold"])
        if out <= 0 or min_out < min_amount_out(out, self._cfg.max_slippage_bps):
            raise ChainError("Jupiter quote has an unexpected minimum output")
        route = " > ".join(step.get("swapInfo", {}).get("label", "?") for step in q.get("routePlan", []))
        return Quote(side, amount_in_raw, out, min_out, route, q)

    @_guard
    def swap(self, side: str, amount_in: float, ref_price: float) -> Fill:
        cfg = self._cfg
        if self._keypair is None:
            raise ChainError("read-only client cannot swap")
        if cfg.dry_run:
            raise ChainError("refusing to send a transaction while DRY_RUN is true")
        self._ensure_verified()
        _, _, dec_in, dec_out = _mints(side)
        cap_raw = to_raw(cfg.max_trade_usdc, USDC_DECIMALS)
        amount_in_raw = to_raw(amount_in, dec_in)
        if amount_in_raw <= 0:
            raise SwapFailed("amount is zero")
        if side == BUY and amount_in_raw > cap_raw:
            raise SwapFailed(f"buy of {amount_in} USDC exceeds MAX_TRADE_USDC={cfg.max_trade_usdc}")

        before = self._snapshot()
        if before.lamports < to_raw(cfg.gas_reserve_sol, SOL_DECIMALS):
            raise SwapFailed(
                f"native SOL {from_raw(before.lamports, SOL_DECIMALS):.6f} is below GAS_RESERVE_SOL={cfg.gas_reserve_sol}"
            )
        if amount_in_raw > (before.usdc if side == BUY else before.wsol):
            raise SwapFailed("insufficient balance for this swap")

        quote = self.quote(side, amount_in_raw)
        if side == SELL and quote.amount_out_raw > cap_raw:
            # Keep the USD value of a sell under the hard cap: shrink and re-quote.
            amount_in_raw = amount_in_raw * cap_raw * 999 // (quote.amount_out_raw * 1000)
            quote = self.quote(side, amount_in_raw)
            if amount_in_raw <= 0 or quote.amount_out_raw > cap_raw:
                raise SwapFailed("could not size sell under MAX_TRADE_USDC")
        deviation_bps = abs(quote.price / ref_price - 1) * 10_000
        if deviation_bps > cfg.max_quote_deviation_bps:
            raise SwapFailed(
                f"quote {quote.price:.2f} is {deviation_bps:.0f} bps from reference {ref_price:.2f}; refusing"
            )
        log.info(
            "%s: in=%s quote_out=%s min_out=%s price=%.2f via %s",
            side, from_raw(amount_in_raw, dec_in), from_raw(quote.amount_out_raw, dec_out),
            from_raw(quote.min_out_raw, dec_out), quote.price, quote.route,
        )

        built = self._jupiter(
            "swap",
            body={
                "quoteResponse": quote.response,
                "userPublicKey": self.address,
                "wrapAndUnwrapSol": False,  # trade wSOL; native SOL stays untouched for fees
                "dynamicComputeUnitLimit": True,
                "prioritizationFeeLamports": {
                    "priorityLevelWithMaxLamports": {"maxLamports": MAX_PRIORITY_LAMPORTS, "priorityLevel": "medium"}
                },
            },
        )
        tx = VersionedTransaction.from_bytes(base64.b64decode(built["swapTransaction"]))
        last_valid = int(built["lastValidBlockHeight"])
        self._check_message(tx)
        signed = self._sign(tx)
        self._check_simulation(signed, side, quote, before, ref_price)
        signature = str(signed.signatures[0])
        self._send_and_confirm(signed, signature, last_valid)

        after = self._settled_snapshot(before)
        spent = (before.usdc - after.usdc) if side == BUY else (before.wsol - after.wsol)
        got = (after.wsol - before.wsol) if side == BUY else (after.usdc - before.usdc)
        if spent <= 0 or got <= 0:
            log.warning("could not read the swap result from balances; recording the quoted amounts")
            spent, got = amount_in_raw, quote.amount_out_raw
        amount_in_f, amount_out_f = from_raw(spent, dec_in), from_raw(got, dec_out)
        price = amount_in_f / amount_out_f if side == BUY else amount_out_f / amount_in_f
        return Fill(amount_in_f, amount_out_f, price, signature)

    def _check_message(self, tx: VersionedTransaction) -> None:
        msg = tx.message
        if msg.account_keys[0] != self.owner:
            raise SwapFailed("Jupiter transaction has an unexpected fee payer")
        if msg.header.num_required_signatures != 1:
            raise SwapFailed("Jupiter transaction requires unexpected extra signers")

    def _sign(self, tx: VersionedTransaction) -> VersionedTransaction:
        return VersionedTransaction(tx.message, [self._keypair])

    def _check_simulation(
        self, tx: VersionedTransaction, side: str, quote: Quote, before: Snapshot, ref_price: float
    ) -> None:
        result = self._rpc(
            "simulateTransaction",
            [
                base64.b64encode(bytes(tx)).decode(),
                {
                    "encoding": "base64",
                    "commitment": "confirmed",
                    "sigVerify": False,
                    "accounts": {
                        "encoding": "base64",
                        "addresses": [self.address, str(self.usdc_account), str(self.wsol_account)],
                    },
                },
            ],
        )["value"]
        if result.get("err") is not None:
            logs = " | ".join((result.get("logs") or [])[-3:])
            raise SwapFailed(f"simulation failed: {json.dumps(result['err'])[:120]} {logs[:200]}")
        owner, usdc, wsol = result["accounts"]
        after = Snapshot((owner or {}).get("lamports", 0), token_amount(usdc), token_amount(wsol))
        spent = (before.usdc - after.usdc) if side == BUY else (before.wsol - after.wsol)
        got = (after.wsol - before.wsol) if side == BUY else (after.usdc - before.usdc)
        sol_cost = before.lamports - after.lamports
        if spent > quote.amount_in_raw:
            raise SwapFailed(f"simulation spends {spent}, more than the requested {quote.amount_in_raw}")
        if got < min(quote.min_out_raw, min_amount_out(quote.amount_out_raw, self._cfg.max_slippage_bps)):
            raise SwapFailed(f"simulation returns {got}, below the slippage floor")
        cost_usd = max(sol_cost, 0) / 10**SOL_DECIMALS * ref_price
        if cost_usd > self._cfg.max_fee_usd:
            raise SwapFailed(f"swap would cost ${cost_usd:.3f} in SOL fees/rent, above MAX_FEE_USD={self._cfg.max_fee_usd}")

    def _send_and_confirm(self, tx: VersionedTransaction, signature: str, last_valid_height: int) -> None:
        try:
            self._rpc(
                "sendTransaction",
                [
                    base64.b64encode(bytes(tx)).decode(),
                    {"encoding": "base64", "skipPreflight": False, "preflightCommitment": "confirmed", "maxRetries": 5},
                ],
            )
        except RpcError as exc:  # the node rejected it in preflight: it was not broadcast
            raise SwapFailed(f"rejected before broadcast: {exc}") from None
        except Exception as exc:  # transport trouble: it may or may not have gone out, so keep watching
            log.warning("sendTransaction errored (%s); watching %s until it lands or expires", type(exc).__name__, signature)

        deadline = time.monotonic() + CONFIRM_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                status = self._rpc("getSignatureStatuses", [[signature], {"searchTransactionHistory": True}])["value"][0]
                if status and status.get("confirmationStatus") in ("confirmed", "finalized"):
                    if status.get("err") is not None:
                        raise SwapFailed(f"transaction {signature} failed on-chain: {json.dumps(status['err'])[:120]}")
                    return
                if status is None and self._rpc("getBlockHeight", [{"commitment": "confirmed"}]) > last_valid_height:
                    # Blockhash expired: the network can no longer include it. Re-check once to be sure.
                    status = self._rpc("getSignatureStatuses", [[signature], {"searchTransactionHistory": True}])["value"][0]
                    if status is None:
                        raise SwapFailed(f"transaction {signature} expired without landing")
            except SwapFailed:
                raise
            except Exception as exc:
                log.warning("status check failed, retrying: %s", self._scrub(str(exc))[:150])
            time.sleep(2)
        raise SwapUncertain(f"transaction {signature} outcome unknown after {CONFIRM_TIMEOUT_S}s")

    def _settled_snapshot(self, before: Snapshot) -> Snapshot:
        for _ in range(5):
            try:
                snap = self._snapshot()
            except Exception:
                snap = before
            if snap != before:
                return snap
            time.sleep(1)
        return before


# Command line: a read-only check and a manual swap for the first live test.

def _reference_price() -> float:
    from bot import data

    return float(data.fetch_recent(limit=3)["close"].iloc[-1])


def _cli_check(client: ChainClient) -> None:
    ref = _reference_price()
    b = client.balances()
    print(f"Solana mainnet OK | wallet {client.address}")
    print(f"USDC {b.usdc:.6f} | wSOL {b.wsol:.9f} | SOL (fees) {b.sol:.6f} | reference SOL price {ref:.2f}")
    for side, amount in ((BUY, 1.0), (SELL, 1.0 / ref)):
        _, _, dec_in, _ = _mints(side)
        q = client.quote(side, to_raw(amount, dec_in))
        print(f"quote {side:<4} ~$1: price {q.price:.2f} ({(q.price / ref - 1) * 1e4:+.0f} bps vs reference) via {q.route}")


def new_wallet(env_path: Path, force: bool = False) -> str:
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    current = next((ln.split("=", 1)[1].strip() for ln in lines if ln.startswith("PRIVATE_KEY=")), "")
    if current and not force:
        raise ChainError(f"{env_path.name} already has a PRIVATE_KEY; refusing to overwrite it (use --force)")
    keypair = Keypair()
    entry = f"PRIVATE_KEY={keypair}"
    if any(ln.startswith("PRIVATE_KEY=") for ln in lines):
        lines = [entry if ln.startswith("PRIVATE_KEY=") else ln for ln in lines]
    else:
        lines.insert(0, entry)
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(keypair.pubkey())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Manual on-chain tools. 'check' is read-only; 'swap' sends a real transaction."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="verify cluster/mints, show balances and quotes (no key needed with --address)")
    check.add_argument("--address", help="read-only: inspect this wallet instead of the PRIVATE_KEY wallet")
    swap = sub.add_parser("swap", help="REAL swap; needs DRY_RUN=false. buy: amount in USDC, sell: amount in wSOL")
    swap.add_argument("side", choices=[BUY, SELL])
    swap.add_argument("amount", type=float)
    swap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    wallet = sub.add_parser("new-wallet", help="create a dedicated bot wallet and save its key to .env")
    wallet.add_argument("--force", action="store_true", help="replace an existing PRIVATE_KEY")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.command == "check":
        _cli_check(ChainClient(address=args.address))
        return
    if args.command == "new-wallet":
        env_path = ROOT / ".env"
        if not env_path.exists() and (ROOT / ".env.example").exists():
            env_path.write_text((ROOT / ".env.example").read_text(encoding="utf-8"), encoding="utf-8")
        try:
            address = new_wallet(env_path, force=args.force)
        except ChainError as exc:
            raise SystemExit(str(exc))
        print(f"New bot wallet: {address}")
        print(f"Its secret key was written to {env_path} (not shown). Back that file up somewhere safe:")
        print("losing it means losing whatever is in the wallet. Fund it with ~$20 USDC and ~0.03 SOL.")
        return
    if settings.dry_run:
        raise SystemExit("DRY_RUN is true. Set DRY_RUN=false in .env to send a real transaction.")
    client = ChainClient()
    ref = _reference_price()
    unit = "USDC" if args.side == BUY else "wSOL"
    print(f"About to {args.side} {args.amount} {unit} from {client.address} (limit MAX_TRADE_USDC={settings.max_trade_usdc}).")
    if not args.yes and input("Type 'yes' to send: ").strip().lower() != "yes":
        raise SystemExit("Aborted.")
    fill = client.swap(args.side, args.amount, ref)
    print(f"Done: in {fill.amount_in:.9f}, out {fill.amount_out:.9f}, price {fill.price:.2f}, tx {fill.tx_hash}")
    print(f"https://solscan.io/tx/{fill.tx_hash}")


if __name__ == "__main__":
    main()
