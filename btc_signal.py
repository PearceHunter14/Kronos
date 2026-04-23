"""
Multi-asset crypto signal agent — BTC, ETH, XRP
10-agent debate council (5 bulls, 5 bears) must reach overwhelming majority
before any trade fires. Kronos forecasting + technical confluence.

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

ASSETS       = ["BTC/USD", "ETH/USD", "XRP/USD"]
PORTFOLIO_FILE = Path(__file__).parent / "portfolio.json"

LOOKBACK_1H  = 400
LOOKBACK_4H  = 200
PRED_LEN     = 12
SAMPLE_COUNT = 10    # Kronos paths per asset (raise to 20 for more accuracy)

BUY_VOTES_NEEDED  = 8   # out of 10 agents
SELL_VOTES_NEEDED = 7   # slightly easier to exit

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
    """Annualised volatility from recent 1h returns."""
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
    Contrarian signal: extreme fear → good time to buy, extreme greed → caution.
    Returns (value: int, label: str, pts: int)
    """
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        data  = r.json()["data"][0]
        value = int(data["value"])
        label = data["value_classification"]
        if value <= 25:
            pts = 12    # extreme fear = historically good entry
        elif value <= 45:
            pts = 6
        elif value <= 55:
            pts = 0
        elif value <= 75:
            pts = -6
        else:
            pts = -12   # extreme greed = historically bad entry
        return value, label, pts
    except Exception:
        return 50, "Unknown", 0


def fetch_funding_rates():
    """
    BTC perpetual funding rate from Bybit (no API key needed).
    Positive rate = longs pay shorts (crowded/overleveraged longs → bearish lean).
    Negative rate = shorts pay longs (crowded shorts → potential squeeze, bullish lean).
    Returns dict: asset → (rate: float, pts: int)
    """
    # Map our spot assets to Bybit perp symbols
    perp_map = {
        "BTC/USD": "BTC/USDT:USDT",
        "ETH/USD": "ETH/USDT:USDT",
        "XRP/USD": "XRP/USDT:USDT",
    }
    results = {}
    try:
        bybit = ccxt.bybit({"enableRateLimit": True})
        for spot, perp in perp_map.items():
            info = bybit.fetch_funding_rate(perp)
            rate = float(info["fundingRate"])
            if rate < -0.0005:
                pts = 10    # heavily shorted → squeeze potential
            elif rate < 0:
                pts = 5
            elif rate < 0.0003:
                pts = 0     # neutral
            elif rate < 0.001:
                pts = -5    # crowded longs
            else:
                pts = -10   # very crowded → danger
            results[spot] = (rate, pts)
    except Exception:
        results = {a: (0.0, 0) for a in ASSETS}
    return results

# ── 10-Agent debate council ────────────────────────────────────────────────────
#
# Each agent returns {"vote": "BUY"|"SELL"|"HOLD", "reason": str}
# 5 bulls have low bars to BUY, high bars to SELL.
# 5 bears have low bars to SELL, high bars to BUY.
# A trade fires only when enough agents across both camps agree.

class Agent:
    def __init__(self, name, bias):
        self.name = name
        self.bias = bias  # "bull" or "bear"

    def analyze(self, paths, df_1h, df_4h, price, holding):
        raise NotImplementedError


# ── Bull agents ────────────────────────────────────────────────────────────────

class MomentumBull(Agent):
    """Trend-following: wants price above EMAs and rising volume."""
    def __init__(self): super().__init__("Mo (Momentum)", "bull")

    def analyze(self, paths, df_1h, df_4h, price, holding):
        closes = df_1h["close"].values
        e10    = ema(closes, 10)
        e20    = ema(closes, 20)
        vr     = volume_ratio(df_1h["volume"].values)
        chg_6h = (closes[-1] / closes[-7] - 1) if len(closes) > 7 else 0

        if price > e10 and price > e20 and vr > 1.1 and chg_6h > 0:
            return {"vote": "BUY",  "reason": f"Price above EMA10/20, vol {vr:.1f}x, +{chg_6h:.2%} 6h"}
        if price < e20 and chg_6h < -0.02:
            return {"vote": "SELL", "reason": f"Price below EMA20, {chg_6h:.2%} 6h drop"}
        return {"vote": "HOLD", "reason": f"Waiting for momentum — vol {vr:.1f}x"}


class ValueBull(Agent):
    """Dip buyer: wants oversold RSI and Kronos showing recovery."""
    def __init__(self): super().__init__("Warren (Value)", "bull")

    def analyze(self, paths, df_1h, df_4h, price, holding):
        rsi_val  = rsi(df_4h["close"].values[-50:])
        med_end  = float(np.median(paths[:, -1]) / price - 1)
        low_48h  = df_1h["low"].values[-48:].min()
        near_low = price < low_48h * 1.03

        if rsi_val < 45 and med_end > 0.005 and near_low:
            return {"vote": "BUY",  "reason": f"RSI {rsi_val:.0f} oversold, near 48h low, Kronos +{med_end:.2%}"}
        if rsi_val > 72 and holding:
            return {"vote": "SELL", "reason": f"RSI {rsi_val:.0f} — taking profits at top"}
        return {"vote": "HOLD", "reason": f"RSI {rsi_val:.0f} — not cheap enough yet"}


class KronosBull(Agent):
    """Pure model follower: trusts Kronos conviction above all else."""
    def __init__(self): super().__init__("Oracle (Kronos)", "bull")

    def analyze(self, paths, df_1h, df_4h, price, holding):
        fee        = 0.001
        peaks      = paths.max(axis=1)
        net        = (peaks / price) - 1 - (2 * fee)
        conviction = (net > 0).sum() / len(net)
        med_peak   = float(np.median(peaks) / price - 1)

        if conviction >= 0.62:
            return {"vote": "BUY",  "reason": f"{conviction:.0%} paths profitable, median peak +{med_peak:.2%}"}
        if conviction < 0.38:
            return {"vote": "SELL", "reason": f"Only {conviction:.0%} paths profitable — model bearish"}
        return {"vote": "HOLD", "reason": f"Conviction {conviction:.0%} — below threshold"}


class BreakoutBull(Agent):
    """Breakout trader: wants new highs confirmed by volume."""
    def __init__(self): super().__init__("Rex (Breakout)", "bull")

    def analyze(self, paths, df_1h, df_4h, price, holding):
        high_48h   = df_1h["high"].values[-48:].max()
        near_break = price >= high_48h * 0.995
        vr         = volume_ratio(df_1h["volume"].values)
        chg_1h     = float(df_1h["close"].values[-1] / df_1h["close"].values[-2] - 1)

        if near_break and vr > 1.3 and chg_1h > 0:
            return {"vote": "BUY",  "reason": f"Breaking 48h high ${high_48h:,.2f}, vol {vr:.1f}x"}
        if not near_break and holding and price < high_48h * 0.97:
            return {"vote": "SELL", "reason": f"Broke back below resistance — exit"}
        return {"vote": "HOLD", "reason": f"{((high_48h-price)/high_48h)*100:.1f}% below 48h high"}


class TrendBull(Agent):
    """Multi-EMA trend alignment: bulls when all EMAs stack up."""
    def __init__(self): super().__init__("Atlas (Trend)", "bull")

    def analyze(self, paths, df_1h, df_4h, price, holding):
        c4     = df_4h["close"].values
        e20    = ema(c4, 20)
        e50    = ema(c4, 50)
        e100   = ema(c4, 100) if len(c4) >= 100 else ema(c4, len(c4)//2)
        stacked = e20 > e50 > e100 and price > e20

        if stacked:
            return {"vote": "BUY",  "reason": f"4h EMAs fully stacked: {e20:,.2f}>{e50:,.2f}>{e100:,.2f}"}
        if e20 < e50 and holding:
            return {"vote": "SELL", "reason": f"4h EMA20 {e20:,.2f} crossed below EMA50 {e50:,.2f}"}
        return {"vote": "HOLD", "reason": f"4h EMAs not fully aligned (EMA20 {'>' if e20>e50 else '<'} EMA50)"}


# ── Bear agents ────────────────────────────────────────────────────────────────

class RiskBear(Agent):
    """Risk manager: vetoes trades with poor reward-to-risk."""
    def __init__(self): super().__init__("Sigma (Risk)", "bear")

    def analyze(self, paths, df_1h, df_4h, price, holding):
        med_peak = float(np.median(paths.max(axis=1)) / price - 1)
        med_dip  = float(np.median(paths.min(axis=1)) / price - 1)
        rr       = (med_peak / abs(med_dip)) if med_dip < 0 else 99

        if rr >= 2.0 and med_peak > 0.005:
            return {"vote": "BUY",  "reason": f"R/R {rr:.1f}x — upside {med_peak:.2%} vs downside {med_dip:.2%}"}
        if holding and med_dip < -0.025:
            return {"vote": "SELL", "reason": f"Forecast dip {med_dip:.2%} too deep — protecting capital"}
        return {"vote": "HOLD", "reason": f"R/R {rr:.1f}x — insufficient for entry (need 2x+)"}


class OverboughtBear(Agent):
    """RSI guard: refuses to buy extended assets, quick to take profits."""
    def __init__(self): super().__init__("Cecil (Overbought)", "bear")

    def analyze(self, paths, df_1h, df_4h, price, holding):
        rsi_val = rsi(df_4h["close"].values[-50:])
        chg_24h = float(df_1h["close"].values[-1] / df_1h["close"].values[-25] - 1) if len(df_1h) > 25 else 0

        if rsi_val < 55 and chg_24h < 0.06:
            return {"vote": "BUY",  "reason": f"RSI {rsi_val:.0f} fine, 24h move {chg_24h:.2%} — not extended"}
        if rsi_val > 68 or chg_24h > 0.08:
            return {"vote": "SELL", "reason": f"RSI {rsi_val:.0f}, +{chg_24h:.2%} 24h — overbought, avoid"}
        return {"vote": "HOLD", "reason": f"RSI {rsi_val:.0f} — marginal, passing"}


class VolatilityBear(Agent):
    """Volatility cop: pulls the plug when conditions are chaotic."""
    def __init__(self): super().__init__("Storm (Volatility)", "bear")

    def analyze(self, paths, df_1h, df_4h, price, holding):
        ann_vol  = hourly_vol(df_1h["close"].values)
        path_std = float(paths[:, -1].std() / price)  # disagreement between paths

        if ann_vol < 0.90 and path_std < 0.025:
            return {"vote": "BUY",  "reason": f"Vol {ann_vol:.0%} manageable, path spread {path_std:.2%}"}
        if ann_vol > 1.50 or path_std > 0.05:
            return {"vote": "SELL", "reason": f"Vol {ann_vol:.0%} extreme or paths disagree {path_std:.2%}"}
        return {"vote": "HOLD", "reason": f"Vol {ann_vol:.0%} elevated — waiting for calm"}


class DoomBear(Agent):
    """Worst-case analyst: focuses on the bottom 20% of Kronos scenarios."""
    def __init__(self): super().__init__("Doom (Tail Risk)", "bear")

    def analyze(self, paths, df_1h, df_4h, price, holding):
        n_bad     = max(1, len(paths) // 5)
        worst_idx = np.argsort(paths[:, -1])[:n_bad]
        worst_end = float(paths[worst_idx, -1].mean() / price - 1)
        worst_dip = float(paths[worst_idx].min(axis=1).mean() / price - 1)

        if worst_end > -0.015 and worst_dip > -0.025:
            return {"vote": "BUY",  "reason": f"Worst 20%: end {worst_end:.2%}, dip {worst_dip:.2%} — acceptable"}
        if worst_dip < -0.04 and holding:
            return {"vote": "SELL", "reason": f"Worst 20% dip {worst_dip:.2%} — tail risk too high"}
        return {"vote": "HOLD", "reason": f"Worst 20% dip {worst_dip:.2%} — too risky to enter"}


class ContraryBear(Agent):
    """Contrarian: fades consensus when everyone screams BUY."""
    def __init__(self): super().__init__("Contra (Contrarian)", "bear")

    def analyze(self, paths, df_1h, df_4h, price, holding):
        fee        = 0.001
        peaks      = paths.max(axis=1)
        conviction = ((peaks / price) - 1 - (2 * fee) > 0).sum() / len(peaks)
        vr         = volume_ratio(df_1h["volume"].values)
        chg_4h     = float(df_1h["close"].values[-1] / df_1h["close"].values[-5] - 1) if len(df_1h) > 5 else 0

        # FOMO warning: high conviction + volume surge + already moved = dangerous
        fomo_score = (conviction > 0.80) + (vr > 2.0) + (chg_4h > 0.04)
        if fomo_score >= 2:
            return {"vote": "SELL", "reason": f"FOMO alert: conviction {conviction:.0%}, vol {vr:.1f}x, +{chg_4h:.2%} 4h"}
        if conviction < 0.75 or fomo_score == 0:
            return {"vote": "BUY",  "reason": f"No FOMO detected — conviction {conviction:.0%} reasonable"}
        return {"vote": "HOLD", "reason": f"Mildly cautious — watching for FOMO spike"}


# ── Council ────────────────────────────────────────────────────────────────────

COUNCIL = [
    MomentumBull(), ValueBull(), KronosBull(), BreakoutBull(), TrendBull(),
    RiskBear(), OverboughtBear(), VolatilityBear(), DoomBear(), ContraryBear(),
]

def run_council(paths, df_1h, df_4h, price, holding, fg_pts=0, funding_pts=0):
    results = []
    for agent in COUNCIL:
        try:
            r = agent.analyze(paths, df_1h, df_4h, price, holding)
        except Exception as e:
            r = {"vote": "HOLD", "reason": f"Error: {e}"}
        results.append({"name": agent.name, "bias": agent.bias, **r})

    buy_votes  = sum(1 for r in results if r["vote"] == "BUY")
    sell_votes = sum(1 for r in results if r["vote"] == "SELL")

    # Fear/Greed and funding rate act as vote modifiers (not full agents)
    # Combined bonus > +15 can tip a 7-vote situation to 8 (effectively)
    context_score = fg_pts + funding_pts
    effective_buy_votes  = buy_votes  + (1 if context_score >= 15 else 0)
    effective_sell_votes = sell_votes + (1 if context_score <= -15 else 0)

    if effective_buy_votes >= BUY_VOTES_NEEDED and not holding:
        verdict = "BUY"
    elif effective_sell_votes >= SELL_VOTES_NEEDED and holding:
        verdict = "SELL"
    elif holding:
        verdict = "HOLD"
    else:
        verdict = "WAIT"

    return {
        "verdict": verdict,
        "buy_votes": buy_votes,
        "sell_votes": sell_votes,
        "context_score": context_score,
        "agents": results,
    }

# ── Position sizing ────────────────────────────────────────────────────────────

def calc_position_usd(buy_votes, portfolio, prices):
    tv = total_value(portfolio, prices)
    if   buy_votes >= 9: alloc = SIZE_HIGH
    elif buy_votes >= 8: alloc = SIZE_MED
    else:                alloc = SIZE_LOW
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
    hdr = f"  SIGNAL AGENT  │  10-Agent Council  │  {datetime.now().strftime('%Y-%m-%d %H:%M')}"
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
        fg_p = context.get("fg_pts", 0)
        row(f"  Fear & Greed: {fg_v}/100 — {fg_l}  ({fg_p:+d} pts context bonus)")
    row()

    for asset, res in results.items():
        price   = prices[asset]
        council = res["council"]
        verdict = council["verdict"]
        bv      = council["buy_votes"]
        sv      = council["sell_votes"]
        ticker  = asset.replace("/USDT", "")

        print("╠" + "═" * w + "╣")
        row()
        label = VERDICT_LABELS[verdict]
        row(f"  {ticker}  ${price:,.4f}    {label}   ({bv} buy / {sv} sell)")
        row()

        # Agent vote grid
        row("  BULLS                              BEARS")
        bull_agents = [a for a in council["agents"] if a["bias"] == "bull"]
        bear_agents = [a for a in council["agents"] if a["bias"] == "bear"]
        for b, br in zip(bull_agents, bear_agents):
            bv_icon = VOTE_ICONS[b["vote"]]
            brv_icon = VOTE_ICONS[br["vote"]]
            b_col  = f"{bv_icon} {b['name']:<20}"
            br_col = f"{brv_icon} {br['name']}"
            row(f"  {b_col}  │  {br_col}")
        row()

        # Reasoning for each agent
        row("  Agent reasoning:")
        for a in council["agents"]:
            icon   = VOTE_ICONS[a["vote"]]
            prefix = f"  {icon} {a['name']:<22}"
            reason = a["reason"]
            # truncate long reasons
            if len(prefix) + len(reason) > w - 2:
                reason = reason[:w - 2 - len(prefix) - 3] + "..."
            row(f"{prefix}{reason}")

        # Funding rate
        fund_rate, fund_pts = res.get("funding", (0.0, 0))
        row(f"  Funding rate: {fund_rate*100:+.4f}%  ({fund_pts:+d} pts context bonus)")

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
                row(f"  {t['time'][:16]}  SELL {t['asset'].replace('/USDT','')}  P&L {t['pnl_pct']:+.1f}%  [{t['reason']}]")
            else:
                row(f"  {t['time'][:16]}  BUY  {t['asset'].replace('/USDT','')}  ${t['usd']:,.0f}")
        row()
    row(f"  Next check in {next_min} min  (need {BUY_VOTES_NEEDED}/10 to BUY, {SELL_VOTES_NEEDED}/10 to SELL)")
    print("╚" + "═" * w + "╝")
    sys.stdout.flush()

# ── Main loop ──────────────────────────────────────────────────────────────────

ALERT_EMAIL = "hunterpearce14@gmail.com"

def send_signal_email(subject, asset, action, price, council, portfolio, prices, extra=""):
    """Send email only when a real trade signal fires."""
    password = os.environ.get("EMAIL_PASSWORD", "")
    if not password:
        print(f"  [email] No EMAIL_PASSWORD set — skipping email for {subject}")
        return

    tv      = total_value(portfolio, prices)
    pnl     = tv - portfolio["start_value"]
    agents  = council["agents"]
    bulls   = [a for a in agents if a["bias"] == "bull"]
    bears   = [a for a in agents if a["bias"] == "bear"]
    bv, sv  = council["buy_votes"], council["sell_votes"]

    icon    = "▲" if action == "BUY" else ("▼" if action == "SELL" else "⚠")
    ticker  = asset.replace("/USDT", "")

    def agent_lines(group):
        lines = []
        for a in group:
            v = {"BUY": "▲", "SELL": "▼", "HOLD": "·"}[a["vote"]]
            lines.append(f"  {v}  {a['name']:<24} {a['reason']}")
        return "\n".join(lines)

    body = f"""\
{icon} SIGNAL AGENT — {action} CONFIRMED
{'━' * 52}

Asset:   {asset}
Action:  {action}
Price:   ${price:,.4f}
{extra}
Council: {bv}/10 buy votes  |  {sv}/10 sell votes

BULL AGENTS:
{agent_lines(bulls)}

BEAR AGENTS:
{agent_lines(bears)}

{'━' * 52}
Portfolio: ${tv:,.2f}  |  P&L: {pnl:+.2f} ({pnl/portfolio['start_value']*100:+.1f}%)
Cash: ${portfolio['usd']:,.2f}  |  Progress to $5k: {tv/5000*100:.1f}%
{'━' * 52}
Signal Agent — {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
"""

    msg = MIMEMultipart()
    msg["Subject"] = f"{icon} {action}: {ticker} @ ${price:,.2f} — {bv}/10 agents"
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
        df_1h = fetch_candles(asset, "1h",  limit=LOOKBACK_1H + 20)
        df_4h = fetch_candles(asset, "4h",  limit=LOOKBACK_4H + 20)
        prices[asset]     = float(df_1h["close"].iloc[-1])
        asset_data[asset] = (df_1h, df_4h)
    print("done.")

    print(f"[{datetime.now().strftime('%H:%M')}] Fetching market context...", end=" ", flush=True)
    fg_value, fg_label, fg_pts = fetch_fear_greed()
    funding_rates = fetch_funding_rates()
    print(f"F&G={fg_value} ({fg_label}), funding fetched.")

    print(f"[{datetime.now().strftime('%H:%M')}] Running Kronos + council ({SAMPLE_COUNT} paths × {len(ASSETS)} assets)...", end=" ", flush=True)
    results = {}
    for asset, (df_1h, df_4h) in asset_data.items():
        price        = prices[asset]
        holding      = asset in portfolio["positions"]
        paths        = kronos_paths(df_1h, predictor)
        fund_pts     = funding_rates.get(asset, (0.0, 0))[1]
        council      = run_council(paths, df_1h, df_4h, price, holding, fg_pts, fund_pts)
        results[asset] = {
            "council": council, "paths": paths,
            "df_1h": df_1h, "df_4h": df_4h,
            "funding": funding_rates.get(asset, (0.0, 0)),
        }
    print("done.")
    # Store context for dashboard
    results["_context"] = {"fg_value": fg_value, "fg_label": fg_label, "fg_pts": fg_pts}

    # Stop-loss checks
    for asset, pos in list(portfolio["positions"].items()):
        if prices[asset] <= pos["stop"]:
            pnl    = paper_sell(portfolio, asset, prices[asset], reason="stop-loss")
            print(f"  ⚠ STOP-LOSS: {asset} @ ${prices[asset]:,.4f}  ({pnl:+.1f}%)")
            send_signal_email(
                f"Stop-Loss: {asset}", asset, "STOP", prices[asset],
                results[asset]["council"], portfolio, prices,
                extra=f"P&L:    {pnl:+.1f}%",
            )
            df_1h, df_4h = asset_data[asset]
            fund_pts = results[asset]["funding"][1]
            results[asset]["council"] = run_council(
                results[asset]["paths"], df_1h, df_4h, prices[asset], False, fg_pts, fund_pts)

    sells = [(a, r) for a, r in results.items() if isinstance(r, dict) and r.get("council", {}).get("verdict") == "SELL"]
    buys  = [(a, r) for a, r in results.items() if isinstance(r, dict) and r.get("council", {}).get("verdict") == "BUY"]

    for asset, res in sells:
        pnl = paper_sell(portfolio, asset, prices[asset], reason="council")
        print(f"  ▼ SELL {asset}: P&L {pnl:+.1f}% ({res['council']['sell_votes']}/10 voted sell)")
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
        stop  = min(stop, prices[asset] * 0.97)
        qty   = paper_buy(portfolio, asset, usd, prices[asset], stop)
        bv    = res["council"]["buy_votes"]
        print(f"  ▲ BUY  {asset}: {qty:.6f} (${usd:.0f}) @ ${prices[asset]:,.4f}  [{bv}/10]  Stop: ${stop:,.4f}")
        send_signal_email(
            f"BUY: {asset}", asset, "BUY", prices[asset],
            res["council"], portfolio, prices,
            extra=f"Amount:  ${usd:.0f}  ({qty:.6f} {asset.replace('/USDT','')})\nStop:    ${stop:,.4f}",
        )

    save_portfolio(portfolio)

    next_min = 60 - datetime.now().minute
    context  = results.pop("_context", {})
    display_results = {
        a: {"council": r["council"], "funding": r.get("funding", (0, 0))}
        for a, r in results.items() if isinstance(r, dict) and "council" in r
    }
    print_dashboard(display_results, portfolio, prices, next_min, context)
    return next_min

def send_weekly_summary(portfolio):
    password = os.environ.get("EMAIL_PASSWORD", "")
    if not password:
        print("[email] No EMAIL_PASSWORD — skipping weekly summary")
        return

    # Fetch current prices for unrealised P&L
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

    # Weekly trades (last 7 days)
    cutoff = (datetime.now() - timedelta(days=7)).isoformat()
    week_trades = [t for t in portfolio["trades"] if t["time"] >= cutoff]
    closed = [t for t in week_trades if t["type"] == "SELL"]

    wins      = [t for t in closed if t.get("pnl_pct", 0) > 0]
    losses    = [t for t in closed if t.get("pnl_pct", 0) <= 0]
    win_rate  = len(wins) / len(closed) * 100 if closed else 0
    best      = max(closed, key=lambda t: t.get("pnl_pct", 0), default=None)
    worst     = min(closed, key=lambda t: t.get("pnl_pct", 0), default=None)

    # Open positions
    pos_lines = []
    for asset, pos in portfolio["positions"].items():
        price   = prices.get(asset, pos["entry"])
        unreal  = (price / pos["entry"] - 1) * 100
        val     = pos["qty"] * price
        pos_lines.append(
            f"  {asset:<12}  {pos['qty']:.6f}  entry ${pos['entry']:,.4f}  "
            f"now ${price:,.4f}  ({unreal:+.1f}%)  value ${val:,.2f}"
        )

    # Trade history this week
    trade_lines = []
    for t in week_trades[-20:]:
        if t["type"] == "SELL":
            trade_lines.append(
                f"  {t['time'][:16]}  SELL  {t['asset'].replace('/USDT',''):<5}  "
                f"@ ${t['price']:,.4f}  P&L {t.get('pnl_pct', 0):+.1f}%  [{t.get('reason','')}]"
            )
        else:
            trade_lines.append(
                f"  {t['time'][:16]}  BUY   {t['asset'].replace('/USDT',''):<5}  "
                f"@ ${t['price']:,.4f}  ${t.get('usd', 0):,.0f} deployed"
            )

    progress_bar_len = 30
    filled = int(progress_bar_len * min(tv, 5000) / 5000)
    bar    = "█" * filled + "░" * (progress_bar_len - filled)

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
  Best:      {f"{best['asset'].replace('/USDT','')} {best['pnl_pct']:+.1f}%" if best else 'N/A'}
  Worst:     {f"{worst['asset'].replace('/USDT','')} {worst['pnl_pct']:+.1f}%" if worst else 'N/A'}

{chr(10).join(trade_lines) if trade_lines else '  No trades this week'}

{'━' * 52}
Checks run every 2 hours. Signals require 8/10 agents to BUY, 7/10 to SELL.
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
    Technical-only backtest over last ~90 days of hourly data.
    Skips Kronos (too slow for 100+ windows) — tests RSI/EMA/volume edge only.
    """
    print("Running backtest (technical signals only, ~90 days)...")
    exchange = ccxt.kraken({"enableRateLimit": True})

    for asset in ASSETS:
        print(f"\n{'─'*50}\n{asset}")
        raw   = exchange.fetch_ohlcv(asset, "1h", limit=2000)
        df    = pd.DataFrame(raw, columns=["ts","open","high","low","close","volume"])
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

            # Entry: oversold + uptrend + volume
            buy_sig  = rsi_val < 45 and trend_up and vr > 1.0
            # Exit: overbought OR trend breaks
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

        wins     = [t for t in trades if t > 0]
        losses   = [t for t in trades if t <= 0]
        win_rate = len(wins) / len(trades) * 100
        avg_win  = np.mean(wins)  if wins   else 0
        avg_loss = np.mean(losses) if losses else 0
        total    = sum(trades)
        expectancy = (win_rate/100 * avg_win) + ((1-win_rate/100) * avg_loss)

        print(f"  Trades:      {len(trades)}  ({len(wins)}W / {len(losses)}L)")
        print(f"  Win rate:    {win_rate:.1f}%")
        print(f"  Avg win:     {avg_win:+.2f}%")
        print(f"  Avg loss:    {avg_loss:+.2f}%")
        print(f"  Expectancy:  {expectancy:+.2f}% per trade")
        print(f"  Total P&L:   {total:+.2f}%")
        verdict = "✅ Positive edge" if expectancy > 0 else "❌ No edge detected"
        print(f"  Verdict:     {verdict}")

    print("\nNote: this tests technical signals only. Kronos + agent council may differ.")


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
