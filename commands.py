"""Operator commands — the only way a human changes what the bot is doing.

The bot trades on its own. This module lets the operator TELL it to do something
specific from the cockpit chat ("sab position close kar do", "capital 100 kar do")
without touching code or restarting anything.

Two design rules, both deliberate:

1. **A question is never a command.** "kitni positions open hain?" must not close
   anything. So an instruction is recognised only when an action VERB and a TARGET
   are both present, and anything that looks like a question is refused outright.
   The cost of a false positive here is a real position closed by accident.

2. **Commands are explicit, never inferred.** Nothing in the trading loop calls
   into here; only an operator message does.

Every recognised command is returned as a `Command` and executed against the live
runner, and the result goes into the activity feed so there is an audit trail of
who changed what.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Words that mean "I am asking", not "do this". Checked BEFORE any verb match.
_QUESTION = re.compile(
    r"\b(kya|kaun|kitna|kitni|kitne|konsa|kaunsa|batao|bataa|bata|dikha|dikhao|"
    r"what|which|how|why|when|kyu|kyun|kab|hain\?|status)\b", re.I)

_CLOSE_VERB = re.compile(
    r"\b(close|band|cut|exit|flatten|square[\s-]?off|nikaal|nikal|hata|hatao|bech)\b", re.I)
_ALL = re.compile(r"\b(all|sab|sabhi|saari|sari|saare|sare|har|everything|"
                  r"poori|puri|pura|sb)\b", re.I)
_POSITIONS = re.compile(r"\b(position|positions|trade|trades|order|orders)\b", re.I)
_BOT_WORD = re.compile(r"\b(bot|trading|scan|scanning|loop|runner)\b", re.I)

# "balance 100 usdt kar do", "capital ko 100 set karo", "trade with $100"
_CAPITAL = re.compile(
    r"\b(balance|capital|equity|funds?|account|paisa|paise|amount|size)\b", re.I)
_NUMBER = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(?:usdt|usd|dollars?|\$)?", re.I)
_FULL_ACCOUNT = re.compile(r"\b(full|pura|poora|whole|entire|sara|saara)\b", re.I)

_START = re.compile(r"\b(start|resume|chalu|chalao|shuru|on)\b", re.I)
_STOP = re.compile(r"\b(stop|pause|halt|ruk|roko|rok)\b", re.I)
_HELP = re.compile(r"\b(help|commands?|kya kar sakta|options)\b", re.I)
# "abhi close mat karna" is the opposite of an order to close — treat any negation
# as "do nothing", because acting on a negated instruction is the worst failure here.
_NEGATED = re.compile(r"\b(mat|nahi|nhi|na karna|don'?t|do ?not|never|avoid)\b", re.I)


@dataclass
class Command:
    kind: str          # close | close_all | capital | start | stop | help
    arg: object = None
    raw: str = ""


def parse(text: str) -> Command | None:
    """Return a Command if the message is an unambiguous instruction, else None
    (in which case the caller should answer it as a normal question)."""
    t = (text or "").strip()
    if not t:
        return None
    if _HELP.search(t):
        return Command("help", raw=t)
    if _QUESTION.search(t) or t.rstrip().endswith("?"):
        return None                      # asking, not ordering
    if _NEGATED.search(t):
        return None                      # "close mat karna" — never act on a negation

    # --- capital / balance ---
    if _CAPITAL.search(t):
        if _FULL_ACCOUNT.search(t) and not _NUMBER.search(t):
            return Command("capital", None, raw=t)
        m = _NUMBER.search(t)
        if m:
            return Command("capital", float(m.group(1).replace(",", "")), raw=t)

    # --- close ---
    if _CLOSE_VERB.search(t):
        # "trading band kar do" = pause the bot, not close positions.
        if _BOT_WORD.search(t) and not _POSITIONS.search(t) and not _ALL.search(t):
            return Command("stop", raw=t)
        if _ALL.search(t) or _POSITIONS.search(t):
            sym = _symbol_in(t)
            return Command("close", sym, raw=t) if sym else Command("close_all", raw=t)
        sym = _symbol_in(t)
        if sym:
            return Command("close", sym, raw=t)
        return None                      # a verb with no target is too vague to act on

    # --- run state ---
    if _BOT_WORD.search(t):
        if _STOP.search(t):
            return Command("stop", raw=t)
        if _START.search(t):
            return Command("start", raw=t)
    return None


def _symbol_in(text: str) -> str | None:
    """Pick a coin ticker out of the message, if there is one. Matched against the
    live universe so an ordinary word can never be mistaken for a symbol."""
    import config
    words = set(re.findall(r"[A-Za-z][A-Za-z0-9]{1,11}", text.upper()))
    for sym in config.UNIVERSE:
        base = sym.replace("USDT", "")
        if sym in words or base in words:
            return sym
    return None


def execute(cmd: Command, runner) -> str:
    """Run the command against the live runner and describe what happened."""
    if cmd.kind == "help":
        return HELP_TEXT

    if cmd.kind == "close_all":
        results = runner.close_all()
        if not results:
            return "Koi open position hi nahi hai — kuch close karne ko nahi tha."
        ok = [r for r in results if r["ok"]]
        bad = [r for r in results if not r["ok"]]
        total = sum(r.get("pnl", 0) or 0 for r in ok)
        out = [f"✅ {len(ok)} position close kar di (net ${total:+.2f}):"]
        out += [f"  • {r['symbol']}: ${r.get('pnl', 0):+.2f}" for r in ok]
        if bad:
            out.append(f"⚠️ {len(bad)} close nahi hui:")
            out += [f"  • {r['msg']}" for r in bad]
        out.append("Bot ab bhi chalu hai aur naye setups scan kar raha hai.")
        return "\n".join(out)

    if cmd.kind == "close":
        r = runner.close_now(cmd.arg)
        return ("✅ " + r["msg"]) if r["ok"] else ("⚠️ " + r["msg"])

    if cmd.kind == "capital":
        r = runner.set_capital(cmd.arg)
        return ("✅ " + r["msg"]) if r["ok"] else ("⚠️ " + r["msg"])

    if cmd.kind == "stop":
        runner.stop()
        return ("⏸️ Bot rok diya — naye trades nahi lega. Open positions ke stop/target "
                "Bybit ke server pe hain, wo phir bhi lagenge. 'bot start karo' bolke wapas chalu karo.")

    if cmd.kind == "start":
        runner.start(int(__import__("os").environ.get("QUANT_PERIOD", "180")))
        return "▶️ Bot chalu — scanning shuru."

    return "Ye command samajh nahi aaya."


HELP_TEXT = """Main ye commands maan leta hoon (baaki time khud trade karta rehta hoon):

  • "sab position close kar do"     — saari open positions band
  • "ZRO close kar do"              — sirf ek coin ki position band
  • "balance 100 usdt kar do"       — bot $100 ke account jaisa size/risk lega
                                      (exchange ka balance nahi badalta — Bybit demo
                                       balance API se set nahi hota)
  • "capital full kar do"           — wapas pura account manage karo
  • "bot band kar do" / "bot start karo" — naye trades rokna / chalu karna

Sawaal poochhne pe kuch change nahi karta — sirf jawab deta hoon."""
