"""Cockpit — the operator dashboard. Run: python cockpit.py  then open
http://localhost:8787

  - START / STOP button controls the 24/7 runner (in-process thread).
  - Live equity, PnL, open positions, daily drawdown, news sentiment.
  - Trade history + the lessons the bot has learned from its losses.
  - Chat box: ask the bot what it's doing (Ollama if installed, else rule-based).
"""
from __future__ import annotations

from flask import Flask, jsonify, request, send_file, abort

import journal
import chat as chat_brain
from runner import QuantRunner

import os

app = Flask(__name__)
# Default to the live TradingView paper account (the user attached the bot to it).
# Set QUANT_MODE=paper to use the internal $1000 simulator instead.
RUNNER = QuantRunner(mode=os.environ.get("QUANT_MODE", "tradingview"))

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Quant-Bot Cockpit</title>
<style>
 body{font-family:Segoe UI,Arial;margin:0;background:#0d1117;color:#e6edf3}
 header{background:#161b22;padding:14px 22px;border-bottom:1px solid #30363d;display:flex;
   align-items:center;justify-content:space-between}
 h1{font-size:18px;margin:0}
 .pill{padding:4px 10px;border-radius:12px;font-size:12px;font-weight:600}
 .on{background:#1a7f37;color:#fff}.off{background:#6e2630;color:#fff}
 button{cursor:pointer;border:0;border-radius:6px;padding:9px 18px;font-weight:600;font-size:14px}
 .start{background:#238636;color:#fff}.stop{background:#da3633;color:#fff}
 .wrap{display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:16px;max-width:1100px;margin:auto}
 .card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px}
 .card h2{font-size:13px;text-transform:uppercase;color:#8b949e;margin:0 0 10px}
 .big{font-size:26px;font-weight:700}.green{color:#3fb950}.red{color:#f85149}
 table{width:100%;border-collapse:collapse;font-size:13px}
 td,th{text-align:left;padding:5px 6px;border-bottom:1px solid #21262d}
 .full{grid-column:1/3}
 #log{height:120px;overflow:auto;font-size:12px;color:#8b949e}
 #chat{height:180px;overflow:auto;background:#0d1117;border:1px solid #30363d;border-radius:6px;
   padding:8px;font-size:13px;margin-bottom:8px}
 .u{color:#58a6ff}.b{color:#3fb950}
 input{width:75%;padding:9px;border-radius:6px;border:1px solid #30363d;background:#0d1117;color:#e6edf3}
</style></head><body>
<header><h1>⚡ Quant-Bot Cockpit <span id="modetag" style="color:#8b949e;font-size:12px">…</span></h1>
 <div><span id="rstate" class="pill off">STOPPED</span>
 <button id="toggle" class="start" onclick="toggle()">START</button></div></header>
<div class="wrap">
 <div class="card"><h2>Equity</h2><div class="big" id="equity">--</div>
   <div id="pnl">--</div><div style="color:#8b949e" id="dd">--</div></div>
 <div class="card"><h2>All-time stats</h2><div id="stats">--</div>
   <div style="margin-top:8px;color:#8b949e" id="senti">--</div></div>
 <div class="card full"><h2>🔮 Next move — the bot's forward read</h2>
   <div style="color:#8b949e;font-size:11px;margin-bottom:8px" id="fcnote">
     Probabilities, not predictions. Measured out-of-sample: 52.3% directional accuracy,
     AUC 0.529 — a small real tilt, shown honestly (calibrated, so 55% means 55%).</div>
   <table id="fc"><thead><tr><th>Coin</th><th>Next move</th><th>Odds</th><th>Exp. move</th>
     <th>Conviction</th><th>Order flow</th><th>Why</th></tr></thead><tbody></tbody></table></div>
 <div class="card full"><h2>Open positions</h2><table id="pos"><tbody></tbody></table></div>
 <div class="card full"><h2>🖥️ Bot's live TradingView chart</h2>
   <div style="color:#8b949e;font-size:12px;margin-bottom:6px" id="tvinfo">warming up…</div>
   <img id="tvimg" style="width:100%;border-radius:8px;border:1px solid #30363d;display:none" alt="bot chart"></div>
 <div class="card full"><h2>🔬 Live activity (what the bot is doing now)</h2>
   <div style="color:#8b949e;font-size:12px;margin-bottom:6px" id="scaninfo">--</div>
   <div id="activity" style="height:150px;overflow:auto;font-size:12px;font-family:Consolas,monospace"></div></div>
 <div class="card"><h2>👀 Watching now (crypto scan)</h2>
   <table id="watch"><thead><tr><th>Coin</th><th>Regime</th><th>Score</th><th>Setup</th><th>RSI</th><th>ADX</th><th>Signal</th><th>ML</th></tr></thead><tbody></tbody></table></div>
 <div class="card"><h2>📰 Crypto news (learning input)</h2><div id="news" style="height:200px;overflow:auto;font-size:12px"></div></div>
 <div class="card full"><h2>Trade history</h2><table id="hist"><thead><tr>
   <th>Time</th><th>Sym</th><th>Side</th><th>Entry</th><th>Exit</th><th>PnL</th><th>R</th><th>Why / Lesson</th>
   </tr></thead><tbody></tbody></table></div>
 <div class="card full"><h2>Lessons learned</h2><div id="log"></div></div>
 <div class="card full"><h2>💬 Chat with the bot</h2><div id="chat"></div>
   <input id="q" placeholder="abhi konsa trade le raha hai? pnl kitna hai?"
     onkeydown="if(event.key=='Enter')send()">
   <button class="start" onclick="send()">Ask</button></div>
</div>
<script>
async function refresh(){
 let s=await (await fetch('/api/status')).json();
 rstate.textContent=s.running?'RUNNING':'STOPPED';
 rstate.className='pill '+(s.running?'on':'off');
 toggle.textContent=s.running?'STOP':'START';toggle.className=s.running?'stop':'start';
 equity.textContent='$'+(s.equity||0).toFixed(2);
 let p=s.pnl||0;pnl.innerHTML='PnL: <b class="'+(p>=0?'green':'red')+'">$'+p.toFixed(2)+'</b>';
 dd.textContent='Daily DD: '+(s.daily_dd_pct||0)+'%'+(s.halted?' — HALTED':'');
 if(s.trading_capital){
   // Two different numbers, both true: the book the bot sizes against, and what the
   // exchange actually holds. Showing only one of them would be misleading.
   dd.innerHTML+='<br><span style="font-size:11px">Trading capital $'+s.trading_capital+
     ' &middot; exchange balance $'+(s.real_balance||0).toLocaleString()+'</span>';
 }
 let st=s.stats||{};let lr=s.learning||{};
 stats.innerHTML='Trades '+st.trades+' &middot; Win '+st.win_rate_pct+'% &middot; PF '+st.profit_factor+' &middot; Net $'+st.net_pnl+
   '<br><span style="color:#3fb950">🧠 Learning: '+(lr.live_trades_logged||0)+' trades logged, '+(lr.lessons_count||0)+' lessons</span>';
 let ex=s.exchange||{};let mt=document.getElementById('modetag');
 if(s.mode=='bybit'){
   let env=(ex.env||'?').toUpperCase();
   // A mainnet session must be impossible to mistake for a paper one.
   mt.innerHTML=ex.authenticated
     ? 'Bybit '+env+(env=='MAINNET'?' <b style="color:#f85149">REAL MONEY</b>':' (paper funds)')
     : '<b style="color:#f85149">Bybit NOT CONNECTED</b> — run setup_bybit.py';
   document.body.style.borderTop=(env=='MAINNET'&&ex.authenticated)?'4px solid #f85149':'';
 } else mt.textContent=s.mode=='tradingview'?'TradingView paper account':'internal simulator';
 let ms=s.market_sentiment||{};senti.textContent='News sentiment '+ms.score+(ms.risk_off?' (RISK-OFF)':'');
 scaninfo.textContent='Scans done: '+(s.scans||0)+'  |  crypto-only  |  '+(s.last_note||'');
 let af=document.getElementById('activity');af.innerHTML=(s.activity||['(warming up...)']).map(a=>'• '+a).join('<br>');
 let tv=s.tv_view||{};let ti=document.getElementById('tvinfo');let tg=document.getElementById('tvimg');
 if(tv.shot){let k={TRADE:'🟢 IN TRADE',SETUP:'🎯 SETUP',WATCH:'👁️ WATCHING'}[tv.kind]||'';
   ti.innerHTML=k+' <b>'+(tv.symbol||'')+'</b> '+(tv.side?'<b>'+tv.side+'</b> ':'')+'— '+(tv.label||'')+
     ' <span style="color:#586069">('+(tv.updated||'').slice(11,19)+' UTC)</span>';
   tg.src='/tv_chart.png?t='+Date.now();tg.style.display='block';}
 else{ti.textContent='Bot will drive the TradingView chart once it picks a focus coin (needs TradingView open w/ debug port).';}
 let wb=document.querySelector('#watch tbody');wb.innerHTML='';
 (s.watching||[]).forEach(w=>{let col=w.regime.indexOf('UP')>=0?'green':(w.regime.indexOf('DOWN')>=0?'red':'');
   let sc=w.score||0;let scol=sc>=70?'green':(sc>=40?'':'#8b949e');
   wb.innerHTML+='<tr><td>'+w.symbol+'</td><td class="'+col+'">'+w.regime+'</td>'+
   '<td style="color:'+(scol||'inherit')+';font-weight:600">'+sc+'</td>'+
   '<td style="font-size:11px">'+(w.setup||'-')+'</td><td>'+w.rsi+'</td><td>'+w.adx+'</td><td>'+
   (w.signal?'<b>'+w.signal+'</b>':'-')+'</td><td>'+(w.ml!=null?w.ml:'-')+'</td></tr>'});
 if(!s.watching||!s.watching.length)wb.innerHTML='<tr><td colspan=8>warming up…</td></tr>';
 let nb=document.getElementById('news');nb.innerHTML=(s.headlines||[]).map(h=>{
   let c=h.tag=='bull'?'green':(h.tag=='bear'?'red':'');return '<div style="margin-bottom:5px"><span class="'+c+'">['+h.tag+']</span> '+h.title+'</div>'}).join('');
 if(!s.headlines||!s.headlines.length)nb.innerHTML='(loading crypto news…)';
 let fb=document.querySelector('#fc tbody');fb.innerHTML='';
 (s.forecasts||[]).forEach(f=>{
   if(f.model_ready===false){fb.innerHTML='<tr><td colspan=7>'+(f.reasons||[''])[0]+'</td></tr>';return}
   let dc=f.direction=='UP'?'green':(f.direction=='DOWN'?'red':'');
   // conviction bar: how far from a coin flip, not a promise
   let w=Math.round((f.confidence||0)*100);
   let mc=f.ms_bias>0.15?'green':(f.ms_bias<-0.15?'red':'');
   fb.innerHTML+='<tr><td><b>'+f.symbol+'</b></td><td class="'+dc+'"><b>'+f.direction+'</b></td>'+
     '<td>'+f.prob+'%</td><td class="'+(f.move_pct>=0?'green':'red')+'">'+(f.move_pct>=0?'+':'')+f.move_pct+'%</td>'+
     '<td><div style="background:#21262d;border-radius:3px;height:8px;width:60px">'+
     '<div style="background:#58a6ff;height:8px;border-radius:3px;width:'+w+'%"></div></div></td>'+
     '<td class="'+mc+'">'+(f.ms_bias>0?'+':'')+f.ms_bias+'</td>'+
     '<td style="font-size:11px;color:#8b949e">'+((f.reasons||[]).slice(0,2).join(' · '))+'</td></tr>'});
 if(!s.forecasts||!s.forecasts.length)fb.innerHTML='<tr><td colspan=7>warming up — forecasts appear after the first scan…</td></tr>';
 let pb=document.querySelector('#pos tbody');pb.innerHTML='';
 (s.open_positions||[]).forEach(o=>{pb.innerHTML+='<tr><td>'+(o.side==1?'LONG':'SHORT')+' '+o.symbol+
   '</td><td>@'+o.entry+'</td><td>stop '+o.stop+'</td><td>tgt '+o.target+'</td><td>'+o.reason+'</td></tr>'});
 if(!s.open_positions||!s.open_positions.length)pb.innerHTML='<tr><td>No open positions — scanning…</td></tr>';
 log.innerHTML=(s.lessons||['(no losses yet)']).map(l=>'• '+l).join('<br>');
 let h=await (await fetch('/api/trades')).json();let hb=document.querySelector('#hist tbody');hb.innerHTML='';
 h.forEach(t=>{hb.innerHTML+='<tr><td>'+(t.closed_utc||'').slice(5,16)+'</td><td>'+t.symbol+'</td><td>'+
   (t.side==1?'L':'S')+'</td><td>'+(+t.entry).toFixed(4)+'</td><td>'+(+t.exit).toFixed(4)+'</td><td class="'+
   (t.pnl>=0?'green':'red')+'">$'+(+t.pnl).toFixed(2)+'</td><td>'+(t.r_multiple||'')+'</td><td>'+
   (t.postmortem||t.exit_reason||'')+'</td></tr>'});
 if(!h.length)hb.innerHTML='<tr><td colspan=8>No trades yet.</td></tr>';
}
async function toggle(){let s=await (await fetch('/api/status')).json();
 await fetch(s.running?'/api/stop':'/api/start',{method:'POST'});setTimeout(refresh,400)}
async function send(){let q=document.getElementById('q');if(!q.value)return;
 chat.innerHTML+='<div class="u">You: '+q.value+'</div>';let question=q.value;q.value='';
 let r=await (await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({q:question})})).json();
 chat.innerHTML+='<div class="b">Bot ('+r.engine+'): '+r.answer.replace(/\\n/g,'<br>')+'</div>';
 chat.scrollTop=chat.scrollHeight}
refresh();setInterval(refresh,5000);
</script></body></html>"""


@app.route("/")
def index():
    return PAGE


@app.route("/api/status")
def status():
    RUNNER.risk.update_equity(RUNNER.equity)
    return jsonify({
        "running": RUNNER.running,
        "mode": RUNNER.mode,
        "equity": round(RUNNER.equity, 2),
        "start_equity": RUNNER.risk.start_equity,
        "pnl": round(RUNNER.equity - RUNNER.risk.start_equity, 2),
        "daily_dd_pct": RUNNER.risk.daily_drawdown_pct(),
        "halted": RUNNER.risk.halted,
        "open_positions": journal.open_positions(),
        "stats": journal.stats(),
        "lessons": journal.lessons(6),
        "market_sentiment": _senti(),
        "watching": RUNNER.watching[:12],
        "activity": RUNNER.activity_log[-15:][::-1],
        "scans": RUNNER.scans,
        "headlines": _headlines(),
        "learning": {
            "live_trades_logged": journal.stats().get("trades", 0),
            "lessons_count": len(journal.lessons(99)),
        },
        "last_note": RUNNER.last_note,
        "tv_view": RUNNER.tv_view,
        "forecasts": RUNNER.forecasts,
        "exchange": RUNNER.exchange_info,
        "trading_capital": _capital(),
        "manages_full_account": (RUNNER.mode == "bybit"
                                 and __import__("runner").BYBIT_CAPITAL is None),
        "real_balance": round(RUNNER.real_equity, 2) if RUNNER.mode == "bybit" else None,
    })


@app.route("/tv_chart.png")
def tv_chart():
    shot = (RUNNER.tv_view or {}).get("shot")
    if not shot or not os.path.exists(shot):
        abort(404)
    return send_file(shot, mimetype="image/png")


def _capital():
    """What the bot sizes against: the configured notional book, or — when it runs
    the whole account — the live balance itself."""
    if RUNNER.mode != "bybit":
        return None
    cap = __import__("runner").BYBIT_CAPITAL
    return cap if cap is not None else round(RUNNER.real_equity, 2)


def _senti():
    try:
        import news
        s = news.market_sentiment()
        return {"score": s.score, "risk_off": s.risk_off, "n": s.n_headlines}
    except Exception:  # noqa: BLE001
        return {"score": 0, "risk_off": False, "n": 0}


def _headlines():
    try:
        import news
        return news.recent_headlines(8)
    except Exception:  # noqa: BLE001
        return []


@app.route("/api/trades")
def trades():
    return jsonify(journal.recent_trades(40))


@app.route("/api/start", methods=["POST"])
def start():
    period = int(request.args.get("period", 120))
    started = RUNNER.start(period)
    return jsonify({"started": started, "running": RUNNER.running})


@app.route("/api/stop", methods=["POST"])
def stop():
    RUNNER.stop()
    return jsonify({"running": RUNNER.running})


@app.route("/api/chat", methods=["POST"])
def chat():
    q = (request.get_json(silent=True) or {}).get("q", "")
    return jsonify(chat_brain.ask(q))


def _open_browser():
    """Wait until the cockpit is serving, then open it in Chrome (fallback: default browser).
    Set QUANT_NOBROWSER=1 to disable."""
    import threading, time, urllib.request, webbrowser, glob

    def _worker():
        url = "http://localhost:8787/"
        # Poll the server until it answers (or ~30s pass), so we don't open a dead page.
        for _ in range(60):
            try:
                urllib.request.urlopen(url, timeout=1)
                break
            except Exception:
                time.sleep(0.5)
        # Prefer Chrome explicitly; fall back to the default browser.
        candidates = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        ]
        candidates += glob.glob(os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"))
        chrome = next((p for p in candidates if p and os.path.exists(p)), None)
        try:
            if chrome:
                webbrowser.register("chrome", None, webbrowser.BackgroundBrowser(chrome))
                webbrowser.get("chrome").open(url)
            else:
                webbrowser.open(url)
            print(f"opened cockpit in browser: {url}")
        except Exception as e:
            print(f"could not auto-open browser ({e}); open {url} manually")

    threading.Thread(target=_worker, daemon=True).start()


def _keep_awake() -> None:
    """Render's free tier suspends a web service after ~15 minutes with no inbound
    HTTP traffic — which would stop the bot every night. Hitting our own public URL
    on a timer counts as traffic and keeps the instance alive. Costs one request
    every 10 minutes and needs no external uptime service.

    Only runs when the platform actually gives us a public URL, so it is a no-op
    locally.
    """
    url = os.environ.get("RENDER_EXTERNAL_URL")
    if not url:
        return
    import threading, time, urllib.request

    def _worker():
        while True:
            time.sleep(600)
            try:
                urllib.request.urlopen(f"{url.rstrip('/')}/api/status", timeout=20).read(1)
            except Exception as e:  # noqa: BLE001
                print(f"[keep-awake] ping failed: {e}")

    threading.Thread(target=_worker, daemon=True).start()
    print(f"keep-awake pinging {url} every 10 min")


if __name__ == "__main__":
    # For unattended 24/7: auto-start the trading loop on launch (set QUANT_AUTOSTART=0 to disable).
    if os.environ.get("QUANT_AUTOSTART", "1") != "0":
        period = int(os.environ.get("QUANT_PERIOD", "180"))
        RUNNER.start(period)
        print(f"runner auto-started ({RUNNER.mode}, {period}s/tick)")
    if os.environ.get("QUANT_NOBROWSER", "0") != "1":
        _open_browser()
    _keep_awake()
    # On a cloud host (Render/Fly/etc) the platform assigns the port via $PORT and
    # requires binding 0.0.0.0 — 127.0.0.1 would be unreachable and the deploy would
    # be marked failed. Locally both default back to the usual localhost:8787.
    port = int(os.environ.get("PORT", "8787"))
    host = os.environ.get("QUANT_HOST", "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1")
    print(f"Cockpit -> http://{host}:{port}")
    app.run(host=host, port=port, threaded=True)
