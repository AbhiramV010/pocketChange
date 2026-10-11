import base64
import json
from dataclasses import replace

import pytest
from solders.hash import Hash
from solders.keypair import Keypair
from solders.message import MessageV0
from solders.signature import Signature
from solders.system_program import TransferParams, transfer
from solders.transaction import VersionedTransaction

from bot import chain
from bot.chain import BUY, SELL, ChainClient, ChainError, Quote, RpcError, SwapFailed, SwapUncertain
from bot.config import load_settings

KEY = Keypair()
CFG = replace(
    load_settings({}),
    dry_run=False,
    solana_rpc_url="https://rpc.example/?api-key=SECRETTOKEN123",
    gas_reserve_sol=0.01,
)
PRICE = 100.0  # USDC per SOL in these tests


def token_account(amount: int | None):
    if amount is None:
        return None
    data = bytes(64) + amount.to_bytes(8, "little") + bytes(93)
    return {"data": [base64.b64encode(data).decode(), "base64"], "lamports": 2_039_280}


def unsigned_tx(payer) -> str:
    """A real, unsigned v0 transaction with `payer` as fee payer, like Jupiter returns."""
    ix = transfer(TransferParams(from_pubkey=payer, to_pubkey=Keypair().pubkey(), lamports=1))
    msg = MessageV0.try_compile(payer, [ix], [], Hash.default())
    return base64.b64encode(bytes(VersionedTransaction.populate(msg, [Signature.default()]))).decode()


class FakeNet:
    """A scripted Solana RPC node and Jupiter API."""

    def __init__(self, client, lamports=10**8, usdc=20_000_000, wsol=0):
        self.client = client
        self.before = (lamports, usdc, wsol)
        self.after = None  # balances once the swap has landed
        self.sim = None  # simulated post-balances; defaults to `after`
        self.sim_err = None
        self.send_error = None
        self.lands_despite_error = False
        self.statuses = [{"confirmationStatus": "confirmed", "err": None}]
        self.block_height = 100
        self.calls = []
        self.swap_body = None
        self.landed = False

    @staticmethod
    def _accounts(lamports, usdc, wsol):
        return [{"lamports": lamports, "data": ["", "base64"]}, token_account(usdc), token_account(wsol)]

    @staticmethod
    def quote_out(side, amount_in):
        if side == BUY:  # 1 USDC (1e6) -> 1e9 / PRICE lamports
            return amount_in * 10**3 // int(PRICE)
        return amount_in * int(PRICE) // 10**3

    def rpc(self, method, params=None):
        self.calls.append(method)
        if method == "getGenesisHash":
            return chain.MAINNET_GENESIS
        if method == "getAccountInfo":
            decimals = 6 if params[0] == str(chain.USDC_MINT) else 9
            return {"value": {"owner": str(chain.TOKEN_PROGRAM), "data": {"parsed": {"info": {"decimals": decimals}}}}}
        if method == "getMultipleAccounts":
            state = self.after if (self.landed and self.after) else self.before
            return {"value": self._accounts(*state)}
        if method == "simulateTransaction":
            if self.sim_err:
                return {"value": {"err": self.sim_err, "logs": ["Program log: boom"], "accounts": None}}
            return {"value": {"err": None, "accounts": self._accounts(*(self.sim or self.after))}}
        if method == "sendTransaction":
            if self.send_error:
                self.landed = self.lands_despite_error
                raise self.send_error
            self.landed = True
            return "sig"
        if method == "getSignatureStatuses":
            status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
            if isinstance(status, Exception):
                raise status
            return {"value": [status]}
        if method == "getBlockHeight":
            return self.block_height
        raise AssertionError(method)

    def jupiter(self, path, params=None, body=None):
        self.calls.append(f"jup:{path}")
        if path == "quote":
            side = BUY if params["inputMint"] == str(chain.USDC_MINT) else SELL
            amount = int(params["amount"])
            out = self.quote_out(side, amount)
            return {
                "inputMint": params["inputMint"], "outputMint": params["outputMint"], "inAmount": str(amount),
                "outAmount": str(out), "otherAmountThreshold": str(out * (10_000 - int(params["slippageBps"])) // 10_000),
                "routePlan": [{"swapInfo": {"label": "TestAMM"}}],
            }
        self.swap_body = body
        return {"swapTransaction": unsigned_tx(self.client.owner), "lastValidBlockHeight": 150}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("PRIVATE_KEY", str(KEY))
    monkeypatch.setattr(chain.time, "sleep", lambda s: None)
    return ChainClient(CFG)


def wire(client, **kw):
    net = FakeNet(client, **kw)
    client._rpc = net.rpc
    client._jupiter = net.jupiter
    return net


def fair_buy(net, usdc_in=1_000_000, fee=10_000):
    lamports, usdc, wsol = net.before
    out = net.quote_out(BUY, usdc_in)
    net.after = (lamports - fee, usdc - usdc_in, wsol + out)
    return out


# --- pure helpers ---------------------------------------------------------

def test_to_raw_rounds_down_and_handles_floats():
    assert chain.to_raw(1.0, 6) == 1_000_000
    assert chain.to_raw(0.1 + 0.2, 6) == 300_000
    assert chain.to_raw(1e-05, 9) == 10_000
    assert chain.from_raw(1_500_000, 6) == 1.5


def test_min_amount_out():
    assert chain.min_amount_out(10_000, 100) == 9_900


def test_quote_price_both_directions():
    assert Quote(BUY, 100_000_000, 10**9, 0, "", {}).price == pytest.approx(100)
    assert Quote(SELL, 10**9, 100_000_000, 0, "", {}).price == pytest.approx(100)


def test_ata_is_deterministic_and_per_mint():
    owner = KEY.pubkey()
    assert chain.ata(owner, chain.USDC_MINT) == chain.ata(owner, chain.USDC_MINT)
    assert chain.ata(owner, chain.USDC_MINT) != chain.ata(owner, chain.WSOL_MINT)


def test_token_amount_parses_layout():
    assert chain.token_amount(token_account(123_456)) == 123_456
    assert chain.token_amount(None) == 0


def test_mints_rejects_unknown_side():
    with pytest.raises(ValueError):
        chain._mints("hodl")


# --- private key handling --------------------------------------------------

def test_missing_or_bad_key_never_leaks(monkeypatch):
    monkeypatch.delenv("PRIVATE_KEY", raising=False)
    with pytest.raises(ChainError, match="not set"):
        chain._load_keypair()
    monkeypatch.setenv("PRIVATE_KEY", "not-a-key-SUPERSECRET")
    with pytest.raises(ChainError) as err:
        chain._load_keypair()
    assert "SUPERSECRET" not in str(err.value)
    assert err.value.__cause__ is None and err.value.__suppress_context__


def test_key_accepted_as_base58_or_json_bytes(monkeypatch):
    monkeypatch.setenv("PRIVATE_KEY", str(KEY))
    assert chain._load_keypair().pubkey() == KEY.pubkey()
    monkeypatch.setenv("PRIVATE_KEY", json.dumps(list(bytes(KEY))))
    assert chain._load_keypair().pubkey() == KEY.pubkey()


def test_scrub_removes_rpc_secret_and_key(client):
    out = client._scrub(f"GET https://rpc.example/?api-key=SECRETTOKEN123 failed; key {KEY}")
    assert "SECRETTOKEN123" not in out and str(KEY) not in out


def test_repr_has_no_key(client):
    assert str(KEY) not in repr(client) and client.address in repr(client)


def test_guard_wraps_unexpected_errors_with_scrubbed_message(client):
    def boom(*a, **k):
        raise RuntimeError("down at https://rpc.example/?api-key=SECRETTOKEN123")

    client._rpc = boom
    with pytest.raises(ChainError) as err:
        client.balances()
    assert "SECRETTOKEN123" not in str(err.value) and "down" in str(err.value)


def test_new_wallet_writes_key_once_and_returns_only_address(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("PRIVATE_KEY=\nDRY_RUN=true\n")
    address = chain.new_wallet(env)
    text = env.read_text()
    secret = text.splitlines()[0].split("=", 1)[1]
    assert secret and secret != address and "DRY_RUN=true" in text
    monkeypatch.setenv("PRIVATE_KEY", secret)
    assert str(chain._load_keypair().pubkey()) == address
    with pytest.raises(ChainError, match="refusing"):
        chain.new_wallet(env)


# --- network checks and reads ---------------------------------------------------

def test_wrong_cluster_is_rejected(client):
    net = wire(client)
    client._rpc = lambda m, p=None: "devnet-genesis" if m == "getGenesisHash" else net.rpc(m, p)
    with pytest.raises(ChainError, match="mainnet"):
        client.balances()


def test_balances(client):
    wire(client, lamports=5 * 10**7, usdc=12_340_000, wsol=2 * 10**8)
    b = client.balances()
    assert (b.usdc, b.sol, b.wsol) == (12.34, 0.05, 0.2)


def test_quote_validates_response(client):
    net = wire(client)
    q = client.quote(BUY, 1_000_000)
    assert q.price == pytest.approx(PRICE) and q.route == "TestAMM"
    client._jupiter = lambda path, params=None, body=None: {**net.jupiter(path, params, body), "inAmount": "1"}
    with pytest.raises(ChainError, match="does not match"):
        client.quote(BUY, 1_000_000)


# --- swap safety checks ----------------------------------------------------

def test_swap_refused_in_dry_run(monkeypatch):
    monkeypatch.setenv("PRIVATE_KEY", str(KEY))
    with pytest.raises(ChainError, match="DRY_RUN"):
        ChainClient(replace(CFG, dry_run=True)).swap(BUY, 1.0, PRICE)


def test_read_only_client_cannot_swap():
    with pytest.raises(ChainError, match="read-only"):
        ChainClient(CFG, address=str(KEY.pubkey())).swap(BUY, 1.0, PRICE)


def test_buy_over_cap_refused(client):
    net = wire(client)
    with pytest.raises(SwapFailed, match="MAX_TRADE_USDC"):
        client.swap(BUY, CFG.max_trade_usdc + 0.01, PRICE)
    assert "jup:quote" not in net.calls


def test_refused_below_gas_reserve(client):
    wire(client, lamports=chain.to_raw(CFG.gas_reserve_sol, 9) - 1)
    with pytest.raises(SwapFailed, match="GAS_RESERVE_SOL"):
        client.swap(BUY, 1.0, PRICE)


def test_refused_on_insufficient_balance(client):
    wire(client, usdc=500_000)
    with pytest.raises(SwapFailed, match="insufficient"):
        client.swap(BUY, 1.0, PRICE)


def test_refused_when_quote_far_from_reference(client):
    wire(client)
    with pytest.raises(SwapFailed, match="refusing"):
        client.swap(BUY, 1.0, PRICE * 1.25)


def test_happy_path_buy(client):
    net = wire(client)
    out = fair_buy(net)
    fill = client.swap(BUY, 1.0, PRICE)
    assert fill.amount_in == 1.0 and fill.amount_out == pytest.approx(out / 1e9)
    assert fill.price == pytest.approx(PRICE, rel=1e-3)
    assert fill.tx_hash
    body = net.swap_body
    assert body["wrapAndUnwrapSol"] is False and body["userPublicKey"] == client.address
    assert body["prioritizationFeeLamports"]["priorityLevelWithMaxLamports"]["maxLamports"] == chain.MAX_PRIORITY_LAMPORTS
    assert net.calls.index("simulateTransaction") < net.calls.index("sendTransaction")


def test_sell_is_clamped_under_cap(client):
    net = wire(client, wsol=10**9)  # 1 SOL = $100, far above the $5 cap
    net.after = (net.before[0] - 5000, net.before[1] + 4_990_000, net.before[2] - 49_900_000)
    client.swap(SELL, 1.0, PRICE)
    assert net.calls.count("jup:quote") == 2  # re-quoted at a smaller size
    assert int(net.swap_body["quoteResponse"]["outAmount"]) <= chain.to_raw(CFG.max_trade_usdc, 6)


# --- simulation checks: nothing is sent when the simulated effect is wrong ----------------

@pytest.mark.parametrize(
    "tamper, message",
    [
        (lambda n, a: (a[0], a[1] - 1_000_000, a[2]), "more than the requested"),  # drains extra USDC
        (lambda n, a: (a[0], a[1], n.before[2] + 1), "slippage floor"),  # almost no SOL out
        (lambda n, a: (a[0] - 10**8, a[1], a[2]), "MAX_FEE_USD"),  # burns 0.1 SOL ($10)
    ],
)
def test_bad_simulated_effects_are_refused(client, tamper, message):
    net = wire(client, lamports=10**9)
    fair_buy(net)
    net.sim = tamper(net, net.after)
    with pytest.raises(SwapFailed, match=message):
        client.swap(BUY, 1.0, PRICE)
    assert "sendTransaction" not in net.calls


def test_simulation_error_is_refused(client):
    net = wire(client)
    fair_buy(net)
    net.sim_err = {"InstructionError": [2, "Custom"]}
    with pytest.raises(SwapFailed, match="simulation failed"):
        client.swap(BUY, 1.0, PRICE)
    assert "sendTransaction" not in net.calls


def test_unexpected_fee_payer_is_refused(client):
    net = wire(client)
    fair_buy(net)

    def other_payer(path, params=None, body=None):
        res = net.jupiter(path, params, body)
        if path == "swap":
            res["swapTransaction"] = unsigned_tx(Keypair().pubkey())
        return res

    client._jupiter = other_payer
    with pytest.raises(SwapFailed, match="fee payer"):
        client.swap(BUY, 1.0, PRICE)
    assert "simulateTransaction" not in net.calls


# --- broadcast and confirmation -----------------------------------------------------

def test_preflight_rejection_is_retry_safe(client):
    net = wire(client)
    fair_buy(net)
    net.send_error = RpcError("sendTransaction: blockhash not found")
    with pytest.raises(SwapFailed, match="before broadcast"):
        client.swap(BUY, 1.0, PRICE)


def test_transport_error_then_landed_is_success(client):
    net = wire(client)
    fair_buy(net)
    net.send_error = ConnectionError("reset")
    net.lands_despite_error = True  # the node got it even though the call errored
    assert client.swap(BUY, 1.0, PRICE).amount_in == 1.0


def test_expired_blockhash_is_definitely_failed(client):
    net = wire(client)
    fair_buy(net)
    net.send_error = ConnectionError("reset")
    net.statuses = [None]
    net.block_height = 151  # past lastValidBlockHeight=150
    with pytest.raises(SwapFailed, match="expired"):
        client.swap(BUY, 1.0, PRICE)


def test_failed_on_chain_is_failed(client):
    net = wire(client)
    fair_buy(net)
    net.statuses = [{"confirmationStatus": "confirmed", "err": {"InstructionError": [3, {"Custom": 6001}]}}]
    with pytest.raises(SwapFailed, match="failed on-chain"):
        client.swap(BUY, 1.0, PRICE)


def test_rpc_down_while_confirming_is_uncertain(client, monkeypatch):
    net = wire(client)
    fair_buy(net)
    net.statuses = [ConnectionError("down")]
    clock = iter(range(0, 10_000, 30))
    monkeypatch.setattr(chain.time, "monotonic", lambda: next(clock))
    with pytest.raises(SwapUncertain):
        client.swap(BUY, 1.0, PRICE)
