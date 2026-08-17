#!/usr/bin/env python
"""LLM wedge detector — sends a trivial completion to each local port and alarms on
completion_tokens == 0. HTTP-200-with-empty-body is the exact failure a process-liveness
check (watchdog.sh) cannot see, so this fills that gap. Runs from cron every few minutes.

Alarm = LOUD line to stdout (cron appends to runs/llm_health.log) + touch runs/LLM_WEDGED
so any job / a human can see the flag. Clears the flag when all ports are healthy again.
Uses raw HTTP (no LLMBackend) so it never contends for the single-slot lock beyond one probe.
"""
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

PORTS = (8000, 8001, 8002)
FLAG = Path(__file__).resolve().parent.parent / "runs" / "LLM_WEDGED"

# Template-leakage markers that must NEVER appear in clean decoded output. Their
# presence means the wrong chat template (e.g. Mistral [INST] on a Llama-3 GGUF) —
# the 2026-07-15 regression, which produced degraded-but-nonempty output on
# no-system calls that a token-count check alone would pass.
_LEAK = re.compile(r"\[/?INST\]|</?s>|<\|start_header_id\|>|<\|eot_id\|>|<\|end_of_text\|>")


def probe(port, timeout=30):
    # Use a SYSTEM role on purpose: system-role handling is exactly what the
    # template regression broke (system -> 0 tokens). This probe would catch it.
    body = json.dumps({"model": "default", "temperature": 0.0, "max_tokens": 16,
                       "messages": [{"role": "system", "content": "You are a terse assistant."},
                                    {"role": "user", "content": "Reply with the single word OK."}]}).encode()
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
        tok = (d.get("usage") or {}).get("completion_tokens")
        content = (d["choices"][0]["message"]["content"] or "").strip()
        leak = bool(_LEAK.search(content))
        ok = bool(content) and (tok or 0) > 0 and not leak
        tag = content[:40] + (" <<TEMPLATE-LEAK>>" if leak else "")
        return ok, tok, tag
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {str(e)[:60]}"


def main():
    ts = time.strftime("%F %T")
    results = {p: probe(p) for p in PORTS}
    wedged = [p for p, (ok, _, _) in results.items() if not ok]
    if wedged:
        FLAG.parent.mkdir(exist_ok=True)
        FLAG.write_text(f"wedged_ports={wedged} at {ts}\n")
        print(f"[llm_health {ts}] *** ALARM: WEDGED/EMPTY ports={wedged} ***")
        for p, (ok, tok, info) in results.items():
            print(f"    :{p} ok={ok} completion_tokens={tok} sample={info!r}")
        sys.exit(1)
    else:
        if FLAG.exists():
            FLAG.unlink()
            print(f"[llm_health {ts}] RECOVERED: all ports healthy "
                  f"({ {p: results[p][1] for p in PORTS} })")
        # else: quiet on steady-state health (avoid log spam)


if __name__ == "__main__":
    main()
