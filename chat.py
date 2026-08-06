"""Chat brain for the cockpit. Uses a local Ollama model if one is running
(no paid API), otherwise falls back to a rule-based responder that answers from
the bot's live state + journal. Install Ollama + `ollama pull llama3.2` to get
full natural-language answers; until then the fallback still works.
"""
from __future__ import annotations

import json

import requests

import config
import journal

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"
STATE_PATH = config.ROOT / "paper_state.json"


def _state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:  # noqa: BLE001
            pass
    return {}


def _context_blob() -> str:
    st = _state()
    pos = st.get("open_positions", [])
    lines = [
        f"Bot running: {st.get('running')}",
        f"Equity: ${st.get('equity')} (start ${st.get('start_equity')}, PnL ${st.get('pnl')})",
        f"Daily drawdown: {st.get('daily_dd_pct')}%  halted: {st.get('halted')}",
        f"Open positions: {len(pos)}",
    ]
    for p in pos:
        side = "LONG" if p["side"] == 1 else "SHORT"
        lines.append(f"  - {side} {p['symbol']} @ {p['entry']} stop {p['stop']} target {p['target']} ({p['reason']})")
    s = st.get("stats", {})
    lines.append(f"All-time: {s.get('trades')} trades, {s.get('win_rate_pct')}% win, "
                 f"PF {s.get('profit_factor')}, net ${s.get('net_pnl')}")
    ms = st.get("market_sentiment", {})
    lines.append(f"News sentiment: {ms.get('score')} (risk_off={ms.get('risk_off')})")
    fcs = [f for f in st.get("forecasts", []) if f.get("model_ready", True)]
    if fcs:
        lines.append("Next-move forecast (CALIBRATED probabilities, ~52-53% accurate "
                     "out-of-sample - a small tilt, NOT a prediction):")
        for f in fcs[:5]:
            lines.append(f"  - {f['symbol']}: {f['direction']} {f['prob']}% over {f['horizon']}, "
                         f"expected {f['move_pct']:+}%, order-flow bias {f['ms_bias']:+}")
    ex = st.get("exchange") or {}
    if st.get("mode") == "bybit":
        lines.append(f"Execution: Bybit {ex.get('env', '?')} "
                     f"{'CONNECTED' if ex.get('authenticated') else 'NOT CONNECTED'}")
    lessons = st.get("lessons", [])
    if lessons:
        lines.append("Recent lessons learned:")
        lines += [f"  - {l}" for l in lessons]
    return "\n".join(lines)


def _ollama(question: str) -> str | None:
    prompt = (
        "You are the voice of a systematic crypto trading bot. Answer the user's "
        "question briefly and concretely using ONLY the live state below. If asked "
        "in Hindi/Hinglish, reply in Hinglish.\n\n"
        f"=== LIVE BOT STATE ===\n{_context_blob()}\n\n"
        f"User: {question}\nAnswer:"
    )
    try:
        r = requests.post(OLLAMA_URL, json={"model": OLLAMA_MODEL, "prompt": prompt,
                                            "stream": False}, timeout=30)
        if r.status_code == 200:
            return r.json().get("response", "").strip()
    except Exception:  # noqa: BLE001
        return None
    return None


def _fallback(question: str) -> str:
    q = question.lower()
    st = _state()
    pos = st.get("open_positions", [])
    if any(w in q for w in ["position", "trade", "kya kar", "konsa", "open", "abhi"]):
        if not pos:
            import config
            n = len(config.UNIVERSE)
            return (f"Abhi koi open position nahi hai. Main {n} CRYPTO coins (BTC/ETH/SOL...) "
                    "har 3 min scan kar raha hoon. Trade tabhi lunga jab koi A+ setup mile — "
                    "yaani strategy + ML win-prob + news + risk, chaaron pass ho. Abhi tak "
                    "koi qualifying setup nahi mila, isliye wait kar raha hoon (selective rehna hi edge hai).")
        out = ["Abhi open positions:"]
        for p in pos:
            side = "LONG" if p["side"] == 1 else "SHORT"
            out.append(f"- {side} {p['symbol']} @ {p['entry']}, stop {p['stop']}, "
                       f"target {p['target']} — {p['reason']}")
        return "\n".join(out)
    if any(w in q for w in ["next move", "agla", "aage", "forecast", "predict", "kya hoga",
                            "upar", "niche", "up ya down", "pump", "dump"]):
        fcs = [f for f in st.get("forecasts", []) if f.get("model_ready", True)]
        if not fcs:
            return ("Forecast abhi ready nahi hai. Ek scan complete hone do, ya model train karo: "
                    "`python train_forecast.py 15m 5000 4`.")
        out = ["Agle ~1 ghante ka read (ye PROBABILITY hai, prediction nahi):"]
        for f in fcs[:5]:
            out.append(f"- {f['symbol']}: {f['direction']} {f['prob']}% | expected move "
                       f"{f['move_pct']:+}% | order-flow {f['ms_bias']:+}")
        out.append("")
        out.append("Sach yeh hai: model ki measured accuracy 52.3% hai (out-of-sample, "
                   "77k bars). Yaani 100 me se ~52 baar sahi — ek trade pe ye coin-toss "
                   "jaisa hi hai, edge sirf bahut saare trades pe dikhta hai. Isliye stop-loss "
                   "aur position size hi asli bachaav hai, forecast nahi.")
        return "\n".join(out)
    if any(w in q for w in ["pnl", "profit", "loss", "balance", "equity", "paisa", "kitna"]):
        return (f"Equity ${st.get('equity')} (start ${st.get('start_equity')}, "
                f"PnL ${st.get('pnl')}). All-time: {st.get('stats', {})}.")
    if any(w in q for w in ["lesson", "seekh", "galti", "mistake", "why", "kyu"]):
        ls = st.get("lessons", [])
        return "Recent lessons:\n" + "\n".join(f"- {l}" for l in ls) if ls else \
            "Abhi tak koi loss-lesson record nahi hua (ya bot abhi start nahi hua)."
    if any(w in q for w in ["news", "sentiment", "market"]):
        ms = st.get("market_sentiment", {})
        return f"Market news sentiment: {ms.get('score')} (risk_off={ms.get('risk_off')}, {ms.get('n')} headlines)."
    if any(w in q for w in ["running", "start", "chal", "status", "halt"]):
        return (f"Bot running: {st.get('running')}, halted: {st.get('halted')}, "
                f"daily DD {st.get('daily_dd_pct')}%. Last: {st.get('last_note')}")
    return ("Main poochh sakte ho: 'abhi konsa trade le raha hai', 'pnl kitna hai', "
            "'kya galti se seekha', 'news kya hai', 'status'. "
            "(Full natural-language chat ke liye Ollama install karo: ollama pull llama3.2)")


def ask(question: str) -> dict:
    ans = _ollama(question)
    if ans:
        return {"answer": ans, "engine": "ollama"}
    return {"answer": _fallback(question), "engine": "rule-based"}


if __name__ == "__main__":
    print(ask("abhi konsa trade le raha hai?"))
