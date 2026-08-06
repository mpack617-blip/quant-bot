"""Where do trades die? A gate-by-gate funnel over recent history.

"The bot isn't trading" has many possible causes and guessing between them is how
you end up loosening the wrong knob. This replays the LIVE entry logic bar by bar
over the last N days and counts how many signals each gate removes, so the real
bottleneck is a number rather than a hunch.

    python diag_funnel.py [days] [interval]

Places no orders and touches no journal — pure measurement.
"""
from __future__ import annotations

import sys
from collections import Counter

import config
import runner
import market
import news
import conviction
from data.bybit import get_klines_cached
from features.indicators import feature_frame, atr
from ml.meta import predict_proba, FEATURE_COLS


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    interval = sys.argv[2] if len(sys.argv) > 2 else runner.DECISION_INTERVAL
    bars_per_day = {"15m": 96, "30m": 48, "1h": 24, "4h": 6}.get(interval, 24)
    bars = days * bars_per_day + 300           # +300 warmup for the indicators

    print(f"replaying {days} days on {interval} across {len(config.UNIVERSE)} coins\n")

    counts = Counter()
    # A live equity/risk snapshot so sizing rejections are realistic.
    from risk import RiskManager
    rm = RiskManager(200.0, runner.BYBIT_RISK)

    for sym in config.UNIVERSE:
        try:
            df = get_klines_cached(sym, interval, bars=bars, max_age_min=120)
            feats = feature_frame(df)
            a = atr(df, 14)
            sigs = runner._gen_signals(df, feats, a)
        except Exception as e:  # noqa: BLE001
            print(f"  {sym}: {e}")
            continue

        start = max(len(sigs) - days * bars_per_day, 0)
        for i in range(start, len(sigs)):
            sig = sigs[i]
            if sig is None:
                continue
            counts["1_raw_signals"] += 1
            price = float(df["close"].iloc[i])
            row = feats.iloc[i]

            # --- gate: BTC market regime (uses the CURRENT bias as a stand-in;
            # historical BTC bias would need a second pass, so treat this as
            # indicative of how often the gate bites in today's regime) ---
            if market.blocks(runner.CURRENT_BIAS_FOR_DIAG, sig.side):
                counts["2_blocked_market_bias"] += 1
                continue
            counts["3_passed_market_bias"] += 1

            # --- gate: ML meta-label ---
            try:
                prob = predict_proba(row[FEATURE_COLS].to_dict(), sig.side)
            except Exception:  # noqa: BLE001
                prob = 0.5
            if prob < runner.ML_MIN_PROB:
                counts["4_blocked_ml"] += 1
                continue
            counts["5_passed_ml"] += 1

            # --- gate: conviction sizing + risk manager ---
            stop_d = abs(price - sig.stop)
            if not stop_d:
                counts["6_blocked_zero_stop"] += 1
                continue
            natural_rr = abs(sig.target - price) / stop_d
            plan = conviction.assess(prob=prob, adx=float(row["adx14"]),
                                     natural_rr=natural_rr, side=sig.side,
                                     entry=price, stop=sig.stop, fwd_agree=0.0)
            d = rm.evaluate(side=sig.side, entry=price, stop=plan.stop,
                            open_positions=[], risk_usd=plan.risk_usd)
            if not d.approved:
                counts["7_blocked_risk"] += 1
                continue

            # --- gate: exchange minimum order value (the $5 floor) ---
            notional = d.qty * price
            if notional < 5.0:
                counts["8_blocked_min_notional"] += 1
                continue
            counts["9_WOULD_TRADE"] += 1

    print(f"{'gate':32} {'count':>8}   per day")
    print("-" * 54)
    for k in sorted(counts):
        print(f"{k:32} {counts[k]:>8}   {counts[k]/days:>6.1f}")

    tradeable = counts["9_WOULD_TRADE"]
    print(f"\n==> {tradeable} trades over {days} days = {tradeable/days:.2f} per day")
    if counts["1_raw_signals"]:
        print(f"    {tradeable/counts['1_raw_signals']*100:.0f}% of raw signals survive every gate")
    print("\nNOTE: news and the forward-forecast gates are NOT replayed here (both need")
    print("live data that has no history), so the LIVE trade rate will be a little lower")
    print("than this figure, not higher.")


if __name__ == "__main__":
    runner.CURRENT_BIAS_FOR_DIAG = market.current_bias(runner.DECISION_INTERVAL)
    print(f"current BTC bias: {runner.CURRENT_BIAS_FOR_DIAG} "
          f"({'UP - shorts blocked' if runner.CURRENT_BIAS_FOR_DIAG == 1 else 'DOWN - longs blocked' if runner.CURRENT_BIAS_FOR_DIAG == -1 else 'flat - nothing blocked'})")
    main()
