# Quant-Bot — Systematic Crypto Trading Engine

A professional, fully-custom systematic trading bot. **No paid AI APIs** (no OpenAI/Claude).
Pure quant + machine-learning, the way real trading desks build it. Crypto first (Bybit),
forex-ready architecture. Testnet/paper before any real money.

## Design principles
1. **Edge comes from data + rigorous backtesting + risk control** — not from hype or video volume.
2. **Everything is testable.** No strategy goes live without out-of-sample + walk-forward proof.
3. **Risk first.** Position sizing, max-drawdown kill-switch, and per-trade limits are non-negotiable.
4. **Every decision is logged.** Full audit trail of why each trade happened.

## Architecture
```
[ Data Layer ]      Bybit v5 REST (public, no key) — historical + live klines, funding, OI
        |
[ Feature Engine ]  custom indicators (EMA/RSI/ATR/MACD/BB...), regime, volatility, returns
        |
[ Strategy Engine ] rule-based signals + ML models (sklearn HistGradientBoosting) trained on YOUR data
        |
[ Backtester ]      vectorised, with fees+slippage, walk-forward, out-of-sample, Monte-Carlo
        |
[ Playbook ]        quality vetoes (stop width, HTF alignment, volatility, session)
                    + the DAILY LOSS BUDGET — see playbook.py
        |
[ Risk Manager ]    position sizing, max DD limit, kill-switch, exposure caps
        |
[ Execution ]       Bybit API (TESTNET first), order/position management, 24/7 runner
        |
[ Dashboard/Logs ]  equity curve, live P&L, trade journal
```

## Tech stack
- Python 3.14 · numpy · pandas · requests · scikit-learn
- Custom indicators (no pandas-ta dependency — full control, no version breakage)
- Bybit v5 public REST for data (no API key needed for historical/market data)

## Build roadmap (phased)
- [x] **Phase 0** — Environment + scaffold
- [x] **Phase 1** — Data layer: Bybit historical klines (paginated, cached) + live quotes
- [x] **Phase 2** — Feature engine: indicator library + feature matrix builder
- [x] **Phase 3** — Backtester: bar-by-bar engine with fees/slippage, ATR stops, BE+trailing, full metrics (`backtest/`, `run_backtest.py`)
- [x] **Phase 4** — Strategy v1: trend-pullback + filters (`strategies/trend_pullback.py`). Backtest PF ~1.12, 35% win, +ve expectancy
- [x] **Phase 5** — ML meta-labeling (`ml/meta.py`, `train_ml.py`) + self-learning journal (`journal.py`): learns which signals win, auto post-mortems every loss
- [x] **Phase 6** — Risk manager (`risk.py`): per-trade sizing, exposure cap, daily-drawdown kill-switch
- [x] **News** — free RSS sentiment (`news.py`): per-symbol + market risk-off filter
- [x] **Phase 7a** — Standalone 24/7 paper runner (`runner.py`): own thread, paper fills, logs to journal
- [x] **Cockpit** — web dashboard (`cockpit.py`): START/STOP, live equity/positions/history, Ollama-or-fallback chat (`chat.py`)
- [x] **Phase 7b** — Live execution on the **TradingView paper account** (`brokers/tradingview.py`): runner `mode="tradingview"` places/closes real orders via CDP, syncs equity from the account. Round-trip tested.
- [x] **Phase 7c** — **Bybit account execution** (`brokers/bybit.py`): signed v5 REST, runner `mode="bybit"`, demo/testnet/mainnet. Stops and targets live ON the exchange, not just in Python.
- [x] **Forward-looking layer** — next-move forecaster (`ml/forecast.py`) + live microstructure (`features/microstructure.py`)
- [ ] **Phase 8** — Forward-test on demo, then graduate to small real capital

## Connect to Bybit
```
python setup_bybit.py           # creates bybit_keys.json, then verifies the connection
python setup_bybit.py --test    # also places + closes one minimum-size order (demo/testnet only)
start_bybit.bat                 # run the bot against Bybit (cockpit on http://localhost:8787)
```
`bybit_keys.json` holds `{"api_key", "api_secret", "env"}` where env is `demo` (fake money,
real prices — start here), `testnet`, or `mainnet` (REAL MONEY). The file is gitignored.

Bybit mode needs no chart, no browser and no CDP port — orders go straight to the exchange.
**Every order carries its stop-loss and take-profit to Bybit's servers**, so a position stays
protected if this bot, the PC, or the internet goes down between ticks. Each tick reconciles
the local journal against the exchange, which is always treated as the source of truth.

## The forward-looking layer ("what's the next move?")
Two parts, kept separate on purpose:
- `ml/forecast.py` — a **trained** direction + magnitude model. Measured strictly out-of-sample
  on 77.5k bars across 24 coins: **52.3% accuracy, AUC 0.529**, stable across both halves of the
  sample (0.535 / 0.527). Its probabilities are **calibrated**, so a displayed 55% means 55%.
  The edge sits at **15m bars, ~1 hour ahead**; sweeping longer horizons found nothing
  (1h/6-bars scored AUC 0.492, i.e. a coin flip — that result is in the code comments, not hidden).
- `features/microstructure.py` — **untrained** live reads: open interest vs price, funding rate,
  order-book imbalance, taker flow, multi-timeframe alignment. No usable history exists to train
  these, so they nudge and veto but never drive.

Neither picks trades. The rule strategies still do; this layer vetoes clear disagreement and
nudges conviction (size/target). An honest ceiling: a 52-53% directional tilt is a real edge over
hundreds of trades and is **worthless on any single trade**. Risk control is still what keeps the
account alive.

## Run the cockpit
```
python cockpit.py                  # default: TradingView paper account
start_bybit.bat                    # trade the Bybit account instead
python runner.py 180 bybit         # 24/7 loop headless on Bybit
python runner.py 120 tradingview   # 24/7 loop headless on the TradingView account
python runner.py 120 paper         # or internal $1000 simulator
python run_backtest.py ALL 1h 1500     # backtest the strategy
python train_ml.py 1h 3000 0.55        # (re)train the meta-model
python train_forecast.py 15m 5000 4    # (re)train the next-move forecaster
```
For natural-language chat install Ollama + `ollama pull llama3.2` (else rule-based chat works).

## How it keeps the losing days small (playbook.py)

Three layers, measured together over 197 days and 7,257 signals with a walk-forward
gate (`python research_manage.py`), on identical entries:

| | win rate | R / trade | days ending ≤1 loss | max drawdown |
|---|---|---|---|---|
| before (hold to target, BE stop at +1R, no day rules) | 29% | −0.018R | 24% | −175R |
| after (playbook + partial + loss budget) | 80% | +0.321R | 98% | −11R |

1. **Quality vetoes** — a stop closer than 1.25% of price, a trade against the
   higher-timeframe trend, the bottom half of the volatility range, and entries after
   17:00 UTC are each a measured negative-expectancy group. They are refused.
2. **Bank a third at +0.5R, stop to breakeven** — the trade only has to be right once
   instead of twice, and the reversal after that costs nothing instead of a full R.
   This is what moves the win rate; the expectancy is unchanged.
3. **The daily loss budget** — after the day's second loss, no new entries until
   00:00 UTC; after any loss, a 4-hour cooldown. Losses cluster, so the trade right
   after a loss is a measurably worse bet. This is the single largest effect in the
   whole study.

Every threshold is an env var (`QUANT_MAX_DAILY_LOSSES`, `QUANT_LOSS_COOLDOWN_H`,
`QUANT_EV_MIN`, `QUANT_PARTIAL_R`, `QUANT_PARTIAL_FRAC`) so none of it needs a redeploy.

## Where the equity number comes from

The bot runs on a host with an **ephemeral disk**: every restart restores `journal.db`
and `paper_state.json` to the copies in git. So neither the balance nor the trade
history may be trusted to local state. Instead, on every tick the runner asks Bybit for
every trade closed since `book_start_ms` (a fixed timestamp, the one thing that must
survive a restart), refills the journal with anything missing, and computes

    book equity = QUANT_CAPITAL + realised PnL since the anchor + unrealised PnL now

The previous design anchored to a *balance* read at boot, which is why a losing account
kept reporting exactly $100.00 with an empty trade table.

## Status
Phases 0-3 done; Phase 4 (strategy) in progress. Run a backtest:
`python run_backtest.py ALL 1h 1500`  (or `python run_backtest.py BTCUSDT 4h 2000`)
```
quant-bot/
  config.py                settings (universe, risk, fees)
  data/bybit.py            Bybit v5 data fetcher (public)
  features/indicators.py   custom indicator library + feature_frame
  strategies/              trading strategies (trend_pullback v1)
  playbook.py              entry vetoes + the daily loss budget / cooldown
  research_manage.py       the study behind them (walk-forward, both halves)
  backtest/engine.py       bar-by-bar sim: ATR stops, BE+trailing, fees
  backtest/metrics.py      Sharpe, maxDD, win-rate, profit-factor, expectancy
  run_backtest.py          CLI runner (single symbol or whole universe)
  brokers/bybit.py         signed Bybit v5 execution (demo/testnet/mainnet)
  brokers/tradingview.py   TradingView paper execution via CDP
  features/microstructure.py  OI / funding / order book / taker flow / MTF reads
  ml/forecast.py           next-move direction + magnitude model (calibrated)
  train_forecast.py        trainer + walk-forward report for the forecaster
  setup_bybit.py           connect + verify the Bybit account
  cache/                   downloaded data (gitignored)
  bybit_keys.json          API credentials (gitignored — never commit)
```
