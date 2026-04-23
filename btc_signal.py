"""
Multi-asset crypto signal agent — BTC, ETH
7-agent independent council: each uses a genuinely different data source.
5/7 votes required to BUY, 4/7 to SELL.

Agents:
  1. KronosAgent          — ML forecast: conviction + R/R + tail risk
  2. PositioningAgent     — Futures L/S ratio + funding rate (crowding)
  3. MarketStructureAgent — OI change vs price divergence
  4. MacroRegimeAgent     — CoinGecko total market cap + BTC dominance
  5. OrderFlowAgent       — Real-time order book bid/ask imbalance
  6. SentimentAgent       — Fear & Greed index (contrarian)
  7. MultiTimeframeAgent  — 1h + 4h price structure confluence

Run:
    source ~/Kronos/.venv/bin/activate
    cd ~/Kronos
    python ~/Documents/Hpearce/btc_signal.py
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add Kronos repo to path so 'model' module is importable from anywhere
sys.path.insert(0, str(Path.home() / "Kronos"))

import argparse
import os
import smtplib
from datetime import timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import ccxt
import numpy as np
import pandas as pd
import requests

# ── Config ─────────────────────────────────────────────────────────────────────

ASSETS         = ["BTC/USD", "ETH/USD"]
PORTFOLIO_FILE = Path(__file__).parent / "portfolio.json"

LOOKBACK_1H  = 400
LOOKBACK_4H  = 200
PRED_LEN     = 12
SAMPLE_COUNT = 10    # Kronos paths per asset

BUY_VOTES_NEEDED  = 5   # out of 7 (~71%)
SELL_VOTES_NEEDED = 4   # out of 7 (~57%)

MIN_MOVE     = 0.004
SIZE_LOW     = 0.15
SIZE_MED     = 0.22
SIZE_HIGH    = 0.30
MAX_DEPLOYED = 0.60
STOP_BUFFER  = 1.005

# ── Portfolio ──────────────────────────────────────────────────────────────────

def load_portfolio():
    if PORTFOLIO_FILE.exists():
        return json.loads(PORTFOLIO_FILE.read_text())
    return {"usd": 1000.0, "positions": {}, "trades": [], "start_value": 1000.0}

def save_portfolio(p):
    PORTFOLIO_FILE.write_text(json.dumps(p, indent=2, default=str))

def total_value(portfolio, prices):
    v = portfolio["usd"]
    for asset, pos in portfolio["positions"].items():
        v += pos["qty"] * prices.get(asset, pos["entry"])
    return v

def deployed_pct(portfolio, prices):
    tv = total_value(portfolio, prices)
    if tv == 0:
        return 0
    crypto = sum(pos["qty"] * prices.get(a, pos["entry"])
                 for a, pos in portfolio["positions"].items())
    return crypto / tv

# ── Data ───────────────────────────────────────────────────────────────────────

def fetch_candles(symbol, timeframe="1h", limit=450):
    exchange = ccxt.kraken({"enableRateLimit": True})
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df  = pd.DataFrame(raw, columns=["timestamps","open","high","low","close","volume"])
    df["timestamps"] = pd.to_datetime(df["timestamps"], unit="ms")
    df["amount"]     = df["volume"] * df["close"]
    return df

# ── Indicators ─────────────────────────────────────────────────────────────────

def rsi(closes, period=14):
    c = np.array(closes, dtype=float)
    d = np.diff(c)
    g = np.where(d > 0, d, 0.0)
    l = np.where(d < 0, -d, 0.0)
    ag, al = g[:period].mean(), l[:period].mean()
    for i in range(period, len(g)):
        ag = (ag * (period - 1) + g[i]) / period
        al = (al * (period - 1) + l[i]) / period
    return 100.0 if al == 0 else 100.0 - (100.0 / (1 + ag / al))

def ema(closes, period):
    c = np.array(closes, dtype=float)
    k = 2 / (period + 1)
    e = c[0]
    for x in c[1:]:
        e = x * k + e * (1 - k)
    return e

def volume_ratio(volumes, period=20):
    v = np.array(volumes, dtype=float)
    avg = v[-period-1:-1].mean() if len(v) > period else v.mean()
    return float(v[-1] / avg) if avg > 0 else 1.0

def hourly_vol(closes, period=20):
    c = np.array(closes[-period-1:], dtype=float)
    r = np.diff(np.log(c))
    return float(r.std() * np.sqrt(8760))

# ── Kronos forecast ────────────────────────────────────────────────────────────

def kronos_paths(df_1h, predictor):
    x_df = df_1h.iloc[:LOOKBACK_1H][["open","high","low","close","volume","amount"]]
    x_ts = df_1h.iloc[:LOOKBACK_1H]["timestamps"]
    # Must be pd.Series not DatetimeIndex — Kronos calls .dt on it
    y_ts = pd.Series(pd.date_range(x_ts.iloc[-1], periods=PRED_LEN+1, freq="1h")[1:])
    paths = []
    for _ in range(SAMPLE_COUNT):
        pred = predictor.predict(
            df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
            pred_len=PRED_LEN, T=1.0, top_p=0.9, sample_count=1,
        )
        paths.append(pred["close"].values)
    return np.array(paths)

# ── Market context (fetched once per cycle, shared across all assets) ──────────

def fetch_fear_greed():
    """
    Alternative.me Fear & Greed Index (0=extreme fear, 100=extreme greed).
    Returns (value: int, label: str)
    """
    try:
        r    = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        data = r.json()["data"][0]
        return int(data["value"]), data["value_classification"]
    except Exception:
        return 50, "Unknown"


def fetch_funding_rates():
    """
    BTC/ETH perpetual funding rate from Bybit.
    Returns dict: asset → (rate: float)
    """
    perp_map = {
        "BTC/USD": "BTC/USDT:USDT",
        "ETH/USD": "ETH/USDT:USDT",
    }
    results = {}
    try:
        bybit = ccxt.bybit({"enableRateLimit": True})
        for spot, perp in perp_map.items():
            info = bybit.fetch_funding_rate(perp)
            results[spot] = float(info["fundingRate"])
    except Exception:
        results = {a: 0.0 for a in ASSETS}
    return results


def fetch_onchain(asset):
    """
    Open interest change (24h) and long/short ratio from Bybit public API.
    Returns dict: oi_change_24h, long_ratio, oi_usd
    """
    perp_map = {"BTC/USD": ("BTC/USDT:USDT", "BTCUSDT"),
                "ETH/USD": ("ETH/USDT:USDT", "ETHUSDT")}
    perp, sym = perp_map.get(asset, ("BTC/USDT:USDT", "BTCUSDT"))
    out = {"oi_change_24h": 0.0, "long_ratio": 0.5, "oi_usd": 0.0}

    try:
        bybit   = ccxt.bybit({"enableRateLimit": True})
        oi_hist = bybit.fetch_open_interest_history(perp, "1h", limit=25)
        if len(oi_hist) >= 2:
            oi_now = float(oi_hist[-1].get("openInterestAmount", 0))
            oi_24h = float(oi_hist[0].get("openInterestAmount", oi_now))
            out["oi_usd"]        = oi_now
            out["oi_change_24h"] = (oi_now / oi_24h - 1) if oi_24h > 0 else 0
    except Exception:
        pass

    try:
        r    = requests.get("https://api.bybit.com/v5/market/account-ratio",
                            params={"category": "linear", "symbol": sym,
                                    "period": "1h", "limit": 1}, timeout=10)
        data = r.json()
        if data.get("retCode") == 0:
            lst = data["result"]["list"]
            if lst:
                out["long_ratio"] = float(lst[0]["buyRatio"])
    except Exception:
        pass

    return out


def fetch_macro():
    """CoinGecko global market: total cap 24h change + BTC dominance."""
    try:
        r    = requests.get("https://api.coingecko.com/api/v3/global", timeout=10)
        data = r.json()["data"]
        return {
            "btc_dominance": data["market_cap_percentage"].get("btc", 50.0),
            "cap_chg_24h":   data["market_cap_change_percentage_24h_usd"],
        }
    except Exception:
        return {"btc_dominance": 50.0, "cap_chg_24h": 0.0}


def fetch_orderbook(asset):
    """Bybit perp order book — top-25 bid/ask volume imbalance."""
    perp_map = {"BTC/USD": "BTC/USDT:USDT", "ETH/USD": "ETH/USDT:USDT"}
    perp = perp_map.get(asset, "BTC/USDT:USDT")
    try:
        bybit = ccxt.bybit({"enableRateLimit": True})
        ob    = bybit.fetch_order_book(perp, limit=25)
        bid   = sum(b[1] for b in ob["bids"])
        ask   = sum(a[1] for a in ob["asks"])
        imb   = bid / (bid + ask) if (bid + ask) > 0 else 0.5
        return {"imbalance": imb, "bid_vol": bid, "ask_vol": ask}
    except Exception:
        return {"imbalance": 0.5, "bid_vol": 0, "ask_vol": 0}


# ── 7-Agent independent council ────────────────────────────────────────────────
#
# Each agent receives a unified context dict (ctx) and returns:
#   {"vote": "BUY"|"SELL"|"HOLD", "reason": str}
#
# Data sources are non-overlapping:
#   KronosAgent          → ML model paths
#   PositioningAgent     → L/S ratio + funding rate
#   MarketStructureAgent → OI change vs price direction
#   MacroRegimeAgent     → CoinGecko global market cap
#   OrderFlowAgent       → Order book bid/ask ratio
#   SentimentAgent       → Fear & Greed index
#   MultiTimeframeAgent  → 1h + 4h price structure

class Agent:
    def __init__(self, name):
        self.name = name

    def analyze(self, ctx):
        raise NotImplementedError


class KronosAgent(Agent):
    """
    ML forecast combines conviction (% profitable paths), reward/risk ratio,
    and tail risk (worst-20% scenario) into a single decisive vote.
    Replaces three correlated agents that each cherry-picked one Kronos metric.
    """
    def __init__(self): super().__init__("Kronos (ML Forecast)")

    def analyze(self, ctx):
        paths = ctx["paths"]
        price = ctx["price"]
        fee   = 0.001

        peaks      = paths.max(axis=1)
        dips       = paths.min(axis=1)
        conviction = ((peaks / price - 1 - 2 * fee) > 0).sum() / len(peaks)
        med_peak   = float(np.median(peaks) / price - 1)
        med_dip    = float(np.median(dips)  / price - 1)
        rr         = (med_peak / abs(med_dip)) if med_dip < 0 else 99.0

        n_bad     = max(1, len(paths) // 5)
        worst_dip = float(paths[np.argsort(paths[:, -1])[:n_bad]].min(axis=1).mean() / price - 1)

        if conviction >= 0.65 and rr >= 2.0 and worst_dip > -0.035:
            return {"vote": "BUY",
                    "reason": f"conviction {conviction:.0%}, R/R {rr:.1f}x, tail {worst_dip:.2%}"}
        if conviction < 0.35 or worst_dip < -0.06:
            return {"vote": "SELL",
                    "reason": f"conviction {conviction:.0%}, tail risk {worst_dip:.2%} — model bearish"}
        return {"vote": "HOLD",
                "reason": f"conviction {conviction:.0%}, R/R {rr:.1f}x — below threshold"}


class PositioningAgent(Agent):
    """
    Futures market crowding: L/S ratio shows which side retail is on,
    funding rate shows which side is paying a premium to hold.
    Shorts crowded + negative funding = squeeze setup (bullish).
    Longs crowded + high funding = flush risk (bearish).
    """
    def __init__(self): super().__init__("Positioning (L/S + Funding)")

    def analyze(self, ctx):
        long_ratio   = ctx.get("long_ratio", 0.5)
        funding_rate = ctx.get("funding_rate", 0.0)

        squeeze = long_ratio < 0.43 and funding_rate < 0.0
        crowded = long_ratio > 0.63 and funding_rate > 0.0004

        if squeeze:
            return {"vote": "BUY",
                    "reason": f"shorts crowded ({long_ratio:.0%} long), funding {funding_rate*100:+.4f}% — squeeze setup"}
        if long_ratio < 0.47:
            return {"vote": "BUY",
                    "reason": f"shorts dominant ({long_ratio:.0%} long) — positioning favours upside"}
        if crowded:
            return {"vote": "SELL",
                    "reason": f"longs crowded ({long_ratio:.0%}), funding {funding_rate*100:+.4f}% — flush risk"}
        if long_ratio > 0.67:
            return {"vote": "SELL",
                    "reason": f"longs overcrowded ({long_ratio:.0%}) — retail euphoria, fade signal"}
        return {"vote": "HOLD",
                "reason": f"positioning neutral ({long_ratio:.0%} long, funding {funding_rate*100:+.4f}%)"}


class MarketStructureAgent(Agent):
    """
    OI change vs price direction reveals who's in control.
    OI up + price up = healthy long accumulation (bullish).
    OI up + price down = shorts piling in aggressively (bearish).
    OI flushed = forced deleveraging over, safer to enter (bullish).
    """
    def __init__(self): super().__init__("Market Structure (OI)")

    def analyze(self, ctx):
        oi_chg = ctx.get("oi_change_24h", 0.0)
        closes = ctx["df_1h"]["close"].values
        px_chg = float(closes[-1] / closes[-25] - 1) if len(closes) > 25 else 0

        if oi_chg > 0.04 and px_chg > 0.01:
            return {"vote": "BUY",
                    "reason": f"OI +{oi_chg:.1%} with price +{px_chg:.1%} — healthy long accumulation"}
        if oi_chg < -0.05:
            return {"vote": "BUY",
                    "reason": f"OI flushed {oi_chg:.1%} — forced deleveraging done, safer entry"}
        if oi_chg > 0.06 and px_chg < -0.01:
            return {"vote": "SELL",
                    "reason": f"OI +{oi_chg:.1%} while price {px_chg:.1%} — shorts piling in"}
        if oi_chg > 0.03 and px_chg < -0.02:
            return {"vote": "SELL",
                    "reason": f"OI rising into price weakness — bearish divergence"}
        return {"vote": "HOLD",
                "reason": f"OI {oi_chg:+.1%}, price {px_chg:+.1%} — no clear structure signal"}


class MacroRegimeAgent(Agent):
    """
    CoinGecko global: total crypto market cap change signals risk appetite.
    Rising market cap = risk-on (buy). Falling = risk-off (sell).
    For alts: rising BTC dominance means capital rotating to BTC specifically,
    which can be a headwind for ETH.
    """
    def __init__(self): super().__init__("Macro Regime (CoinGecko)")

    def analyze(self, ctx):
        macro   = ctx.get("macro", {})
        cap_chg = macro.get("cap_chg_24h", 0.0)
        btc_dom = macro.get("btc_dominance", 50.0)
        asset   = ctx.get("asset", "BTC/USD")

        risk_on  = cap_chg > 1.5
        risk_off = cap_chg < -2.0

        if risk_on:
            if asset != "BTC/USD" and btc_dom > 58:
                return {"vote": "HOLD",
                        "reason": f"market +{cap_chg:.1f}% but BTC dom {btc_dom:.1f}% — alt headwind"}
            return {"vote": "BUY",
                    "reason": f"total market cap +{cap_chg:.1f}% — risk-on regime"}
        if risk_off:
            return {"vote": "SELL",
                    "reason": f"total market cap {cap_chg:.1f}% — risk-off, reduce exposure"}
        if cap_chg < -0.5:
            return {"vote": "HOLD",
                    "reason": f"market cap {cap_chg:.1f}% — mild headwind, cautious"}
        return {"vote": "BUY",
                "reason": f"macro neutral-positive (cap {cap_chg:+.1f}%, BTC dom {btc_dom:.1f}%)"}


class OrderFlowAgent(Agent):
    """
    Real-time order book: bid vs ask volume in the top 25 price levels.
    A dominant bid wall means active buy pressure at current price.
    A dominant ask wall means active sell pressure (supply overhead).
    This is the only agent using live microstructure data.
    """
    def __init__(self): super().__init__("Order Flow (Book)")

    def analyze(self, ctx):
        ob  = ctx.get("orderbook", {})
        imb = ob.get("imbalance", 0.5)

        if imb >= 0.62:
            return {"vote": "BUY",
                    "reason": f"bid wall dominant ({imb:.0%} bids) — active buy pressure"}
        if imb >= 0.55:
            return {"vote": "BUY",
                    "reason": f"bids favoured ({imb:.0%}) — mild buy pressure"}
        if imb <= 0.38:
            return {"vote": "SELL",
                    "reason": f"ask wall dominant ({imb:.0%} bids) — active sell pressure"}
        if imb <= 0.45:
            return {"vote": "SELL",
                    "reason": f"asks favoured ({imb:.0%} bids) — mild sell pressure"}
        return {"vote": "HOLD",
                "reason": f"order book balanced ({imb:.0%} bids)"}


class SentimentAgent(Agent):
    """
    Fear & Greed index as a standalone contrarian vote.
    Extreme fear = historically excellent buy zones (capitulation).
    Extreme greed = historically poor entries (euphoria peaks).
    Previously a weak modifier; now a full independent vote.
    """
    def __init__(self): super().__init__("Sentiment (Fear & Greed)")

    def analyze(self, ctx):
        fg_value = ctx.get("fg_value", 50)
        fg_label = ctx.get("fg_label", "Unknown")

        if fg_value <= 20:
            return {"vote": "BUY",
                    "reason": f"F&G {fg_value}/100 ({fg_label}) — extreme fear = historically great entry"}
        if fg_value <= 40:
            return {"vote": "BUY",
                    "reason": f"F&G {fg_value}/100 ({fg_label}) — fear zone, good contrarian entry"}
        if fg_value <= 60:
            return {"vote": "HOLD",
                    "reason": f"F&G {fg_value}/100 ({fg_label}) — sentiment neutral"}
        if fg_value <= 80:
            return {"vote": "SELL",
                    "reason": f"F&G {fg_value}/100 ({fg_label}) — greed elevated, caution"}
        return {"vote": "SELL",
                "reason": f"F&G {fg_value}/100 ({fg_label}) — extreme greed = historically poor entry"}


class MultiTimeframeAgent(Agent):
    """
    1h and 4h price structure must both agree before a vote fires.
    Replaces MomentumBull + TrendBull which were 80% correlated (both
    just checked if EMAs were stacked). Now requires full confluence:
    1h above EMA20, 4h EMA20 above EMA50, 4h RSI not overbought.
    """
    def __init__(self): super().__init__("Multi-TF (1h + 4h)")

    def analyze(self, ctx):
        df_1h   = ctx["df_1h"]
        df_4h   = ctx["df_4h"]
        price   = ctx["price"]
        holding = ctx.get("holding", False)

        c1 = df_1h["close"].values
        c4 = df_4h["close"].values

        e20_1h = ema(c1, 20)
        e20_4h = ema(c4, 20)
        e50_4h = ema(c4, 50)
        rsi_4h = rsi(c4[-50:])
        chg_24h = float(c1[-1] / c1[-25] - 1) if len(c1) > 25 else 0

        bull_1h        = price > e20_1h
        bull_4h        = e20_4h > e50_4h
        not_overbought = rsi_4h < 72
        rising_24h     = chg_24h > 0

        score = sum([bull_1h, bull_4h, not_overbought, rising_24h])

        if score >= 4:
            return {"vote": "BUY",
                    "reason": f"all TF aligned: 1h>{e20_1h:,.0f}, EMA20>{e50_4h:,.0f}(4h), RSI {rsi_4h:.0f}"}
        if score >= 3 and bull_4h:
            return {"vote": "BUY",
                    "reason": f"TF mostly aligned (4h trend up, RSI {rsi_4h:.0f}, 24h {chg_24h:+.2%})"}
        if not bull_4h and not bull_1h:
            return {"vote": "SELL",
                    "reason": f"both TF bearish: EMA20(4h) {e20_4h:,.0f}<EMA50 {e50_4h:,.0f}, price<1h EMA"}
        if not bull_4h and holding:
            return {"vote": "SELL",
                    "reason": f"4h trend broken (EMA20 {e20_4h:,.0f} < EMA50 {e50_4h:,.0f}) — exit signal"}
        return {"vote": "HOLD",
                "reason": f"mixed TF signals ({score}/4 bullish, RSI {rsi_4h:.0f})"}


# ── Council (7 agents) ─────────────────────────────────────────────────────────

COUNCIL = [
    KronosAgent(),
    PositioningAgent(),
    MarketStructureAgent(),
    MacroRegimeAgent(),
    OrderFlowAgent(),
    SentimentAgent(),
    MultiTimeframeAgent(),
]

def run_council(ctx):
    results = []
    for agent in COUNCIL:
        try:
            r = agent.analyze(ctx)
        except Exception as e:
            r = {"vote": "HOLD", "reason": f"Error: {e}"}
        results.append({"name": agent.name, **r})

    buy_votes  = sum(1 for r in results if r["vote"] == "BUY")
    sell_votes = sum(1 for r in results if r["vote"] == "SELL")
    holding    = ctx.get("holding", False)

    if buy_votes >= BUY_VOTES_NEEDED and not holding:
        verdict = "BUY"
    elif sell_votes >= SELL_VOTES_NEEDED and holding:
        verdict = "SELL"
    elif holding:
        verdict = "HOLD"
    else:
        verdict = "WAIT"

    return {
        "verdict":    verdict,
        "buy_votes":  buy_votes,
        "sell_votes": sell_votes,
        "agents":     results,
    }

# ── Position sizing ────────────────────────────────────────────────────────────

def calc_position_usd(buy_votes, portfolio, prices):
    tv = total_value(portfolio, prices)
    if   buy_votes >= 7: alloc = SIZE_HIGH   # near-unanimous
    elif buy_votes >= 6: alloc = SIZE_MED
    else:                alloc = SIZE_LOW    # bare majority (5/7)
    headroom = max(0, MAX_DEPLOYED - deployed_pct(portfolio, prices))
    alloc    = min(alloc, headroom)
    return min(tv * alloc, portfolio["usd"])

# ── Paper trades ───────────────────────────────────────────────────────────────

def paper_buy(portfolio, asset, usd, price, stop_price):
    qty = (usd / price) * 0.999
    portfolio["usd"] -= usd
    portfolio["positions"][asset] = {"qty": qty, "entry": price, "stop": stop_price}
    portfolio["trades"].append({"type":"BUY","asset":asset,"price":price,"qty":qty,"usd":usd,"time":str(datetime.now())})
    return qty

def paper_sell(portfolio, asset, price, reason="signal"):
    pos      = portfolio["positions"].pop(asset)
    proceeds = pos["qty"] * price * 0.999
    portfolio["usd"] += proceeds
    pnl_pct  = (price / pos["entry"] - 1) * 100
    portfolio["trades"].append({"type":"SELL","asset":asset,"price":price,"qty":pos["qty"],"entry":pos["entry"],"pnl_pct":pnl_pct,"reason":reason,"time":str(datetime.now())})
    return pnl_pct

# ── Dashboard ──────────────────────────────────────────────────────────────────

VOTE_ICONS = {"BUY": "▲", "SELL": "▼", "HOLD": "·"}
VERDICT_LABELS = {
    "BUY":  "✅ BUY CONFIRMED",
    "SELL": "🔴 SELL CONFIRMED",
    "HOLD": "◆ HOLD",
    "WAIT": "· WAITING",
}

def print_dashboard(results, portfolio, prices, next_min, context=None):
    tv  = total_value(portfolio, prices)
    pnl = tv - portfolio["start_value"]
    dep = deployed_pct(portfolio, prices) * 100
    w   = 66

    def row(text=""):
        pad = w - 2 - len(text)
        print(f"║ {text}{' ' * max(0,pad)} ║")

    print("\n╔" + "═" * w + "╗")
    hdr = f"  SIGNAL AGENT  │  7-Agent Council  │  {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    print(f"║{hdr}{' ' * max(0, w - len(hdr))}║")
    print("╠" + "═" * w + "╣")
    row()
    row(f"  Portfolio: ${tv:,.2f}   P&L: {pnl:+.2f} ({pnl/portfolio['start_value']*100:+.1f}%)")
    row(f"  Cash: ${portfolio['usd']:,.2f}   Deployed: {dep:.0f}%   Target: $5,000")
    filled = int(40 * tv / 5000)
    bar    = "█" * filled + "░" * (40 - filled)
    row(f"  [{bar}] {tv/5000*100:.1f}% to goal")
    if context:
        fg_v = context.get("fg_value", 50)
        fg_l = context.get("fg_label", "?")
        macro = context.get("macro", {})
        cap_chg = macro.get("cap_chg_24h", 0.0)
        row(f"  F&G: {fg_v}/100 ({fg_l})  │  Market cap 24h: {cap_chg:+.1f}%")
    row()

    for asset, res in results.items():
        price   = prices[asset]
        council = res["council"]
        verdict = council["verdict"]
        bv      = council["buy_votes"]
        sv      = council["sell_votes"]
        ctx     = res.get("ctx", {})
        ticker  = asset.replace("/USD", "")

        print("╠" + "═" * w + "╣")
        row()
        label = VERDICT_LABELS[verdict]
        row(f"  {ticker}  ${price:,.4f}    {label}   ({bv} buy / {sv} sell)")
        row()

        row("  Agent votes:")
        for a in council["agents"]:
            icon   = VOTE_ICONS[a["vote"]]
            prefix = f"  {icon} {a['name']:<32}"
            reason = a["reason"]
            if len(prefix) + len(reason) > w - 2:
                reason = reason[:w - 2 - len(prefix) - 3] + "..."
            row(f"{prefix}{reason}")
        row()

        # Derivatives context
        fund_rate  = ctx.get("funding_rate", 0.0)
        oi_chg     = ctx.get("oi_change_24h", 0.0)
        long_ratio = ctx.get("long_ratio", 0.5)
        ob_imb     = ctx.get("orderbook", {}).get("imbalance", 0.5)
        row(f"  Funding: {fund_rate*100:+.4f}%  │  OI 24h: {oi_chg:+.1%}"
            f"  │  L/S: {long_ratio:.0%}  │  Book: {ob_imb:.0%} bid")

        # Open position
        if asset in portfolio["positions"]:
            pos    = portfolio["positions"][asset]
            unreal = (price / pos["entry"] - 1) * 100
            row()
            row(f"  ● {pos['qty']:.6f} {ticker} @ ${pos['entry']:,.4f}  ({unreal:+.1f}%)  Stop: ${pos['stop']:,.4f}")
        row()

    print("╠" + "═" * w + "╣")
    recent = portfolio["trades"][-4:]
    if recent:
        row("  Recent trades:")
        for t in recent:
            if t["type"] == "SELL":
                row(f"  {t['time'][:16]}  SELL {t['asset'].replace('/USD','')}  P&L {t['pnl_pct']:+.1f}%  [{t['reason']}]")
            else:
                row(f"  {t['time'][:16]}  BUY  {t['asset'].replace('/USD','')}  ${t['usd']:,.0f}")
        row()
    row(f"  Next check in {next_min} min  (need {BUY_VOTES_NEEDED}/7 to BUY, {SELL_VOTES_NEEDED}/7 to SELL)")
    print("╚" + "═" * w + "╝")
    sys.stdout.flush()

# ── Main loop ──────────────────────────────────────────────────────────────────

ALERT_EMAIL = "hunterpearce14@gmail.com"

def send_signal_email(subject, asset, action, price, council, portfolio, prices, extra=""):
    password = os.environ.get("EMAIL_PASSWORD", "")
    if not password:
        print(f"  [email] No EMAIL_PASSWORD set — skipping email for {subject}")
        return

    tv     = total_value(portfolio, prices)
    pnl    = tv - portfolio["start_value"]
    agents = council["agents"]
    bv     = council["buy_votes"]
    sv     = council["sell_votes"]
    icon   = "▲" if action == "BUY" else ("▼" if action == "SELL" else "⚠")
    ticker = asset.replace("/USD", "")

    agent_lines = "\n".join(
        f"  {VOTE_ICONS[a['vote']]}  {a['name']:<32} {a['reason']}"
        for a in agents
    )

    body = f"""\
{icon} SIGNAL AGENT — {action} CONFIRMED
{'━' * 52}

Asset:   {asset}
Action:  {action}
Price:   ${price:,.4f}
{extra}
Council: {bv}/7 buy votes  |  {sv}/7 sell votes

AGENTS:
{agent_lines}

{'━' * 52}
Portfolio: ${tv:,.2f}  |  P&L: {pnl:+.2f} ({pnl/portfolio['start_value']*100:+.1f}%)
Cash: ${portfolio['usd']:,.2f}  |  Progress to $5k: {tv/5000*100:.1f}%
{'━' * 52}
Signal Agent — {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
"""

    msg = MIMEMultipart()
    msg["Subject"] = f"{icon} {action}: {ticker} @ ${price:,.2f} — {bv}/7 agents"
    msg["From"]    = ALERT_EMAIL
    msg["To"]      = ALERT_EMAIL
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(ALERT_EMAIL, password)
            smtp.send_message(msg)
        print(f"  [email] Sent to {ALERT_EMAIL}")
    except Exception as e:
        print(f"  [email] Failed: {e}")


def load_kronos():
    print("Loading Kronos model (one-time)...")
    from model import Kronos, KronosTokenizer, KronosPredictor
    tok  = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    mdl  = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    pred = KronosPredictor(mdl, tok, max_context=512)
    print("Model ready.\n")
    return pred

def run_cycle(predictor, portfolio):
    prices, asset_data = {}, {}

    print(f"[{datetime.now().strftime('%H:%M')}] Fetching candles...", end=" ", flush=True)
    for asset in ASSETS:
        df_1h = fetch_candles(asset, "1h", limit=LOOKBACK_1H + 20)
        df_4h = fetch_candles(asset, "4h", limit=LOOKBACK_4H + 20)
        prices[asset]     = float(df_1h["close"].iloc[-1])
        asset_data[asset] = (df_1h, df_4h)
    print("done.")

    print(f"[{datetime.now().strftime('%H:%M')}] Fetching market context...", end=" ", flush=True)
    fg_value, fg_label = fetch_fear_greed()
    funding_rates      = fetch_funding_rates()
    onchain_data       = {asset: fetch_onchain(asset) for asset in ASSETS}
    macro              = fetch_macro()
    orderbook_data     = {asset: fetch_orderbook(asset) for asset in ASSETS}
    print(f"F&G={fg_value} ({fg_label}), macro cap {macro.get('cap_chg_24h',0):+.1f}%, OI+book fetched.")

    print(f"[{datetime.now().strftime('%H:%M')}] Running Kronos + council ({SAMPLE_COUNT} paths × {len(ASSETS)} assets)...", end=" ", flush=True)
    results = {}
    for asset, (df_1h, df_4h) in asset_data.items():
        price   = prices[asset]
        holding = asset in portfolio["positions"]
        paths   = kronos_paths(df_1h, predictor)
        onchain = onchain_data.get(asset, {})

        ctx = {
            "paths":         paths,
            "df_1h":         df_1h,
            "df_4h":         df_4h,
            "price":         price,
            "holding":       holding,
            "asset":         asset,
            "long_ratio":    onchain.get("long_ratio", 0.5),
            "oi_change_24h": onchain.get("oi_change_24h", 0.0),
            "funding_rate":  funding_rates.get(asset, 0.0),
            "macro":         macro,
            "orderbook":     orderbook_data.get(asset, {}),
            "fg_value":      fg_value,
            "fg_label":      fg_label,
        }
        council = run_council(ctx)
        results[asset] = {"council": council, "paths": paths, "ctx": ctx,
                          "df_1h": df_1h, "df_4h": df_4h}
    print("done.")

    # Store shared context for dashboard
    results["_context"] = {"fg_value": fg_value, "fg_label": fg_label, "macro": macro}

    # Stop-loss checks
    for asset, pos in list(portfolio["positions"].items()):
        if prices[asset] <= pos["stop"]:
            pnl = paper_sell(portfolio, asset, prices[asset], reason="stop-loss")
            print(f"  ⚠ STOP-LOSS: {asset} @ ${prices[asset]:,.4f}  ({pnl:+.1f}%)")
            send_signal_email(
                f"Stop-Loss: {asset}", asset, "STOP", prices[asset],
                results[asset]["council"], portfolio, prices,
                extra=f"P&L:    {pnl:+.1f}%",
            )
            # Re-run council with holding=False
            new_ctx = {**results[asset]["ctx"], "holding": False}
            results[asset]["council"] = run_council(new_ctx)
            results[asset]["ctx"]     = new_ctx

    sells = [(a, r) for a, r in results.items() if isinstance(r, dict) and r.get("council", {}).get("verdict") == "SELL"]
    buys  = [(a, r) for a, r in results.items() if isinstance(r, dict) and r.get("council", {}).get("verdict") == "BUY"]

    for asset, res in sells:
        pnl = paper_sell(portfolio, asset, prices[asset], reason="council")
        sv  = res["council"]["sell_votes"]
        print(f"  ▼ SELL {asset}: P&L {pnl:+.1f}% ({sv}/7 voted sell)")
        send_signal_email(
            f"SELL: {asset}", asset, "SELL", prices[asset],
            res["council"], portfolio, prices,
            extra=f"P&L:    {pnl:+.1f}%",
        )

    for asset, res in sorted(buys, key=lambda x: -x[1]["council"]["buy_votes"]):
        usd = calc_position_usd(res["council"]["buy_votes"], portfolio, prices)
        if usd < 10:
            continue
        paths = res["paths"]
        dip   = float(np.median(paths.min(axis=1)) / prices[asset] - 1)
        stop  = prices[asset] * (1 + dip * STOP_BUFFER)
        stop  = min(stop, prices[asset] * 0.950)
        qty   = paper_buy(portfolio, asset, usd, prices[asset], stop)
        bv    = res["council"]["buy_votes"]
        print(f"  ▲ BUY  {asset}: {qty:.6f} (${usd:.0f}) @ ${prices[asset]:,.4f}  [{bv}/7]  Stop: ${stop:,.4f}")
        send_signal_email(
            f"BUY: {asset}", asset, "BUY", prices[asset],
            res["council"], portfolio, prices,
            extra=f"Amount:  ${usd:.0f}  ({qty:.6f} {asset.replace('/USD','')})\nStop:    ${stop:,.4f}",
        )

    save_portfolio(portfolio)

    next_min = 60 - datetime.now().minute
    context  = results.pop("_context", {})
    display_results = {
        a: {"council": r["council"], "ctx": r.get("ctx", {})}
        for a, r in results.items() if isinstance(r, dict) and "council" in r
    }
    print_dashboard(display_results, portfolio, prices, next_min, context)
    return next_min

def send_weekly_summary(portfolio):
    password = os.environ.get("EMAIL_PASSWORD", "")
    if not password:
        print("[email] No EMAIL_PASSWORD — skipping weekly summary")
        return

    prices = {}
    try:
        exchange = ccxt.kraken({"enableRateLimit": True})
        for asset in ASSETS:
            ticker = exchange.fetch_ticker(asset)
            prices[asset] = float(ticker["last"])
    except Exception as e:
        print(f"[email] Price fetch failed: {e}")
        prices = {a: portfolio["positions"][a]["entry"]
                  for a in portfolio["positions"]}

    tv      = total_value(portfolio, prices)
    pnl     = tv - portfolio["start_value"]
    pnl_pct = pnl / portfolio["start_value"] * 100
    dep     = deployed_pct(portfolio, prices) * 100

    cutoff      = (datetime.now() - timedelta(days=7)).isoformat()
    week_trades = [t for t in portfolio["trades"] if t["time"] >= cutoff]
    closed      = [t for t in week_trades if t["type"] == "SELL"]
    wins        = [t for t in closed if t.get("pnl_pct", 0) > 0]
    losses      = [t for t in closed if t.get("pnl_pct", 0) <= 0]
    win_rate    = len(wins) / len(closed) * 100 if closed else 0
    best        = max(closed, key=lambda t: t.get("pnl_pct", 0), default=None)
    worst       = min(closed, key=lambda t: t.get("pnl_pct", 0), default=None)

    pos_lines = []
    for asset, pos in portfolio["positions"].items():
        price  = prices.get(asset, pos["entry"])
        unreal = (price / pos["entry"] - 1) * 100
        val    = pos["qty"] * price
        pos_lines.append(
            f"  {asset:<12}  {pos['qty']:.6f}  entry ${pos['entry']:,.4f}  "
            f"now ${price:,.4f}  ({unreal:+.1f}%)  value ${val:,.2f}"
        )

    trade_lines = []
    for t in week_trades[-20:]:
        if t["type"] == "SELL":
            trade_lines.append(
                f"  {t['time'][:16]}  SELL  {t['asset'].replace('/USD',''):<5}  "
                f"@ ${t['price']:,.4f}  P&L {t.get('pnl_pct', 0):+.1f}%  [{t.get('reason','')}]"
            )
        else:
            trade_lines.append(
                f"  {t['time'][:16]}  BUY   {t['asset'].replace('/USD',''):<5}  "
                f"@ ${t['price']:,.4f}  ${t.get('usd', 0):,.0f} deployed"
            )

    filled = int(30 * min(tv, 5000) / 5000)
    bar    = "█" * filled + "░" * (30 - filled)

    body = f"""\
📊 WEEKLY PORTFOLIO SUMMARY
{'━' * 52}
{datetime.now().strftime('%A, %d %B %Y')}

PORTFOLIO VALUE
  Total:     ${tv:,.2f}
  P&L:       {pnl:+.2f} ({pnl_pct:+.1f}%)
  Cash:      ${portfolio['usd']:,.2f}
  Deployed:  {dep:.0f}%

GOAL: $1,000 → $5,000
  [{bar}] {tv/5000*100:.1f}%
  ${tv:,.2f} of $5,000  (${5000-tv:,.2f} to go)

{'━' * 52}
OPEN POSITIONS ({len(portfolio['positions'])})
{'  None' if not pos_lines else chr(10).join(pos_lines)}

{'━' * 52}
THIS WEEK'S TRADES ({len(week_trades)})
  Closed:    {len(closed)}  |  Wins: {len(wins)}  |  Losses: {len(losses)}  |  Win rate: {win_rate:.0f}%
  Best:      {f"{best['asset'].replace('/USD','')} {best['pnl_pct']:+.1f}%" if best else 'N/A'}
  Worst:     {f"{worst['asset'].replace('/USD','')} {worst['pnl_pct']:+.1f}%" if worst else 'N/A'}

{chr(10).join(trade_lines) if trade_lines else '  No trades this week'}

{'━' * 52}
Checks run every 2 hours. Signals require {BUY_VOTES_NEEDED}/7 agents to BUY, {SELL_VOTES_NEEDED}/7 to SELL.
Signal Agent — hunterpearce14@gmail.com
"""

    msg = MIMEMultipart()
    msg["Subject"] = f"📊 Weekly Summary — ${tv:,.2f} ({pnl_pct:+.1f}%) — {datetime.now().strftime('%d %b')}"
    msg["From"]    = ALERT_EMAIL
    msg["To"]      = ALERT_EMAIL
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(ALERT_EMAIL, password)
            smtp.send_message(msg)
        print(f"[email] Weekly summary sent to {ALERT_EMAIL}")
    except Exception as e:
        print(f"[email] Failed: {e}")


def run_backtest():
    """
    Technical-only backtest over ~90 days of hourly data.
    Skips Kronos (too slow for 100+ windows) — tests RSI/EMA/volume edge only.
    """
    print("Running backtest (technical signals only, ~90 days)...")
    exchange = ccxt.kraken({"enableRateLimit": True})

    for asset in ASSETS:
        print(f"\n{'─'*50}\n{asset}")
        raw    = exchange.fetch_ohlcv(asset, "1h", limit=2000)
        df     = pd.DataFrame(raw, columns=["ts","open","high","low","close","volume"])
        closes = df["close"].values
        vols   = df["volume"].values

        trades, position = [], None

        for i in range(300, len(df) - 1):
            price    = closes[i]
            rsi_val  = rsi(closes[max(0,i-60):i+1])
            e20      = ema(closes[max(0,i-50):i+1], 20)
            e50      = ema(closes[max(0,i-100):i+1], 50)
            vr       = volume_ratio(vols[max(0,i-21):i+1])
            trend_up = e20 > e50

            buy_sig  = rsi_val < 45 and trend_up and vr > 1.0
            sell_sig = rsi_val > 65 or e20 < e50

            if position is None and buy_sig:
                position = {"entry": price, "idx": i}
            elif position is not None and sell_sig:
                pnl = (price / position["entry"] - 1) * 100
                trades.append(pnl)
                position = None

        if not trades:
            print("  No completed trades in window")
            continue

        wins       = [t for t in trades if t > 0]
        losses     = [t for t in trades if t <= 0]
        win_rate   = len(wins) / len(trades) * 100
        avg_win    = np.mean(wins)   if wins   else 0
        avg_loss   = np.mean(losses) if losses else 0
        total      = sum(trades)
        expectancy = (win_rate/100 * avg_win) + ((1-win_rate/100) * avg_loss)

        print(f"  Trades:      {len(trades)}  ({len(wins)}W / {len(losses)}L)")
        print(f"  Win rate:    {win_rate:.1f}%")
        print(f"  Avg win:     {avg_win:+.2f}%")
        print(f"  Avg loss:    {avg_loss:+.2f}%")
        print(f"  Expectancy:  {expectancy:+.2f}% per trade")
        print(f"  Total P&L:   {total:+.2f}%")
        verdict = "✅ Positive edge" if expectancy > 0 else "❌ No edge detected"
        print(f"  Verdict:     {verdict}")

    print("\nNote: this tests technical signals only. Kronos + full agent council may differ.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once",     action="store_true", help="Run one cycle then exit")
    parser.add_argument("--summary",  action="store_true", help="Send weekly summary email then exit")
    parser.add_argument("--backtest", action="store_true", help="Run technical backtest then exit")
    args = parser.parse_args()

    if args.summary:
        portfolio = load_portfolio()
        send_weekly_summary(portfolio)
        return

    if args.backtest:
        run_backtest()
        return

    predictor = load_kronos()
    portfolio = load_portfolio()
    print(f"Portfolio: ${portfolio['usd']:.2f} cash, {len(portfolio['positions'])} open positions\n")

    if args.once:
        run_cycle(predictor, portfolio)
        return

    while True:
        try:
            next_min = run_cycle(predictor, portfolio)
        except KeyboardInterrupt:
            print("\nStopped.")
            save_portfolio(portfolio)
            break
        except Exception as e:
            print(f"\n[ERROR] {e}")
            next_min = 2

        time.sleep(next_min * 60)

if __name__ == "__main__":
    main()
