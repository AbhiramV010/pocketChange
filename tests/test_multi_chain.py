from dataclasses import replace

import pytest
from solders.keypair import Keypair

from bot import chain
from bot.assets import ASSETS
from bot.chain import BUY, SELL, ChainClient, SwapFailed
from bot.config import load_settings
from tests.test_chain import token_account, unsigned_tx

KEY = Keypair()
BTC = ASSETS["BTC"]
BTC_PRICE = 50_000.0
CFG = replace(
    load_settings({}), dry_run=False, solana_rpc_url="https://rpc.example/", assets=("SOL", "BTC"), gas_reserve_sol=0.01
)


class Net:
    """A fake chain with SOL, USDC and wBTC accounts, and a Jupiter that quotes BTC at $50,000."""

    def __init__(self, client, lamports=10**8, usdc=20_000_000, wsol=0, btc=0):
        self.client = client
        self.before = (lamports, usdc, wsol, btc)
        self.after = None
        self.landed = False
        self.sim = None
        self.quote_params = []
        self.calls = []

    @staticmethod
    def accounts(lamports, usdc, wsol, btc):
        return [{"lamports": lamports, "data": ["", "base64"]}, token_account(usdc), token_account(wsol), token_account(btc)]

    def rpc(self, method, params=None):
        self.calls.append(method)
        if method == "getGenesisHash":
            return chain.MAINNET_GENESIS
        if method == "getAccountInfo":
            decimals = {str(chain.USDC_MINT): 6, BTC.mint: 8}.get(params[0], 9)
            return {"value": {"owner": str(chain.TOKEN_PROGRAM), "data": {"parsed": {"info": {"decimals": decimals}}}}}
        if method == "getMultipleAccounts":
            assert len(params[0]) == 4  # owner, USDC, wSOL, wBTC
            return {"value": self.accounts(*(self.after if self.landed and self.after else self.before))}
        if method == "simulateTransaction":
            return {"value": {"err": None, "accounts": self.accounts(*(self.sim or self.after))}}
        if method == "sendTransaction":
            self.landed = True
            return "sig"
        if method == "getSignatureStatuses":
            return {"value": [{"confirmationStatus": "confirmed", "err": None}]}
        if method == "getBlockHeight":
            return 100
        raise AssertionError(method)

    def jupiter(self, path, params=None, body=None):
        if path == "quote":
            self.quote_params.append(params)
            amount = int(params["amount"])
            if params["inputMint"] == str(chain.USDC_MINT):  # buy: USDC (6 dp) -> BTC (8 dp)
                out = int(amount / 1e6 / BTC_PRICE * 1e8)
            else:  # sell: BTC -> USDC
                out = int(amount / 1e8 * BTC_PRICE * 1e6)
            return {
                "inputMint": params["inputMint"], "outputMint": params["outputMint"], "inAmount": str(amount),
                "outAmount": str(out), "otherAmountThreshold": str(out * 99 // 100), "routePlan": [],
            }
        return {"swapTransaction": unsigned_tx(self.client.owner), "lastValidBlockHeight": 150}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("PRIVATE_KEY", str(KEY))
    monkeypatch.setattr(chain.time, "sleep", lambda s: None)
    return ChainClient(CFG)


def wire(client, **kw):
    net = Net(client, **kw)
    client._rpc, client._jupiter = net.rpc, net.jupiter
    return net


def test_balances_include_the_other_assets(client):
    wire(client, wsol=2 * 10**8, btc=150_000)
    b = client.balances()
    assert b.wsol == 0.2 and b.extra == {"BTC": 0.0015}
    assert client.asset_accounts["BTC"] == chain.ata(client.owner, chain.Pubkey.from_string(BTC.mint))


def test_buy_btc_quotes_the_right_mints_and_prices_in_dollars(client):
    net = wire(client)
    out = int(1.0 / BTC_PRICE * 1e8)
    net.after = (net.before[0] - 10_000, net.before[1] - 1_000_000, 0, out)
    fill = client.swap(BUY, 1.0, BTC_PRICE, "BTC")
    assert net.quote_params[0]["inputMint"] == str(chain.USDC_MINT) and net.quote_params[0]["outputMint"] == BTC.mint
    assert fill.amount_in == 1.0 and fill.amount_out == pytest.approx(out / 1e8)
    assert fill.price == pytest.approx(BTC_PRICE, rel=1e-3)


def test_sell_btc_is_clamped_under_the_cap(client):
    net = wire(client, btc=10**8)  # 1 BTC = $50,000, far above the $5 cap
    net.after = (net.before[0] - 5000, net.before[1] + 4_990_000, 0, net.before[3] - 9_990)
    client.swap(SELL, 1.0, BTC_PRICE, "BTC")
    sells = [q for q in net.quote_params if q["inputMint"] == BTC.mint]
    assert len(sells) == 2 and int(sells[1]["amount"]) < int(sells[0]["amount"])  # re-quoted smaller


def test_an_asset_outside_the_config_is_refused(client):
    net = wire(client)
    with pytest.raises(SwapFailed, match="not one of the configured"):
        client.swap(BUY, 1.0, 4000.0, "ETH")
    assert not net.quote_params


def test_a_quote_far_from_that_assets_reference_price_is_refused(client):
    wire(client)
    with pytest.raises(SwapFailed, match="refusing"):
        client.swap(BUY, 1.0, BTC_PRICE * 1.2, "BTC")  # the exchange says 60,000, Jupiter 50,000


def test_a_bad_simulated_btc_effect_is_refused_before_sending(client):
    net = wire(client)
    out = int(1.0 / BTC_PRICE * 1e8)
    net.after = (net.before[0] - 10_000, net.before[1] - 1_000_000, 0, out)
    net.sim = (net.before[0] - 10_000, net.before[1] - 1_000_000, 0, out // 2)  # half the BTC we were quoted
    with pytest.raises(SwapFailed, match="slippage floor"):
        client.swap(BUY, 1.0, BTC_PRICE, "BTC")
    assert "sendTransaction" not in net.calls

