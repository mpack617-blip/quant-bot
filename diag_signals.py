"""One-off diagnostic: run the multi-angle scanner on the live market and show,
per symbol, the setup found + opportunity score, then how many pass ALL gates
(signal -> ML win-prob -> news -> risk). Read-only, no orders."""
from __future__ import annotations
import config
from data.bybit import get_klines_cached
from features.indicators import feature_frame, atr, trend_label
from strategies import multi_angle, opportunity_snapshot
from ml.meta import predict_proba, FEATURE_COLS
import news
from risk import RiskManager

INTERVAL, ML_MIN = "15m", 0.50
risk = RiskManager(10.56, {"max_risk_per_trade_pct": 8.0, "max_concurrent_positions": 2,
                           "max_daily_drawdown_pct": 15.0, "default_leverage": 5})
sent = news.market_sentiment()
print(f"news score {sent.score} risk_off={sent.risk_off}\n")
print(f"{'SYM':<7}{'regime':<12}{'score':>6}  setup            gates")
print("-" * 80)
signals = tradable = 0
for sym in config.UNIVERSE:
    try:
        df = get_klines_cached(sym, INTERVAL, bars=500, max_age_min=10)
        f = feature_frame(df)
        a = atr(df, 14)
        snap = opportunity_snapshot(df, f, a)
        sig = multi_angle(df, f, a)[-1]
        line = f"{sym.replace('USDT',''):<7}{trend_label(df):<12}{snap['score']:>6}  "
        if sig is None:
            print(line + f"{'(no signal)':<16}")
            continue
        signals += 1
        setup = sig.reason.split(':')[0]
        row = f.iloc[-1][FEATURE_COLS].to_dict()
        prob = predict_proba(row, sig.side)
        ok_news, _ = news.agrees_with(sym, sig.side)
        dec = risk.evaluate(side=sig.side, entry=float(df['close'].iloc[-1]), stop=sig.stop, open_positions=[])
        gates = []
        gates.append(f"ML {prob:.2f}{'OK' if prob>=ML_MIN else 'X'}")
        gates.append("news" + ("OK" if (ok_news and not sent.risk_off) else "X"))
        gates.append("risk" + ("OK" if dec.approved else "X"))
        passed = prob >= ML_MIN and ok_news and not sent.risk_off and dec.approved
        if passed:
            tradable += 1
        print(line + f"{setup:<16} {' '.join(gates)}  {'==> TRADE' if passed else ''}")
    except Exception as e:
        print(f"{sym:<7} ERROR {e}")
print(f"\nsignals found: {signals}/{len(config.UNIVERSE)}   would actually TRADE (all gates): {tradable}")
