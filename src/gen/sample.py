# -*- coding: utf-8 -*-
"""Stage C formal sampling (frozen protocol v1, 2026-10-23).

Design (see docs/experiment_plan.md S3/S9):
  - 3 flagship families x 2 cultural conditions (CN zh / EU en), n=1000 each;
  - temperature 1.0, prompt template v1, max_tokens 8000 (reasoning models
    otherwise return empty content); glm-5.3 thinking budget_tokens=1000;
  - empty-content replies (reasoning overflow) are retried up to 3 times and
    otherwise logged as failed samples;
  - resumable: JSONL per (family, condition), existing indices skipped;
  - all replies kept raw; parsing/encoding happens in a later stage so this
    script never decides compliance;
  - protocol fingerprint (this file's config) written to runs/c/protocol_v1.json
    with MD5s.

Launch (detached, survives the shell):
  cmd /c start "sampler" /min python sample.py
2026-10-23. Stage C of the seven-stage pipeline.
"""
import json, os, sys, time, hashlib, threading
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from client import chat, APIError
from gate import PROMPT_CN, PROMPT_EU

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
RUNS = os.path.join(ROOT, "runs", "c")

PROTOCOL = {
    "version": "v2",
    "date": "2026-10-23",
    "v1_abandoned_reason": (
        "v1 flagship snapshots are reasoning models whose chain-of-thought "
        "consumed ~90% of spend with no contribution to the measured object "
        "(the output distribution Q). v2 uses non-reasoning snapshots of the "
        "same three families -- a purer sampling channel for the same theory. "
        "v1 partial data (373 deepseek-v4-pro CN samples) discarded: mixed "
        "instruments within a cell are not permitted. Gate compliance of all "
        "three v2 snapshots verified 2026-10-23 (0 reasoning tokens, 100% "
        "ABC parse)."),
    "n_per_cell": 1000,
    "temperature": 1.0,
    "max_tokens": 2000,
    "prompt_versions": {"CN": "v1", "EU": "v1"},
    "cells": {
        "deepseek": dict(model="deepseek-chat", workers=8, extra=None,
                         retries_empty=3,
                         price=dict(in_per_m=2.0, out_per_m=8.0), cap_cny=45),
        "dashscope": dict(model="qwen-flash", workers=8, extra=None,
                          retries_empty=3,
                          price=dict(in_per_m=0.5, out_per_m=2.0), cap_cny=28),
        "zhipu": dict(model="glm-4-air", workers=8, extra=None,
                      retries_empty=3,
                      price=dict(in_per_m=1.0, out_per_m=1.0), cap_cny=28),
    },
    "order": ["deepseek", "dashscope", "zhipu"],
}

WORST_IN_TOK = 250     # prompt tokens upper bound for cap arithmetic
_COUNTERS = {}
_BUDGET_LOCK = threading.Lock()


def budget_guard(fam):
    """True while the family's worst-case spend stays under its cap."""
    cfg = PROTOCOL["cells"][fam]
    with _BUDGET_LOCK:
        n = _COUNTERS.get(fam, 0)
    per_call = (WORST_IN_TOK * cfg["price"]["in_per_m"]
                + PROTOCOL["max_tokens"] * cfg["price"]["out_per_m"]) / 1e6
    return n * per_call < cfg["cap_cny"]

PROMPTS = {"CN": PROMPT_CN, "EU": PROMPT_EU}
LOCK = threading.Lock()


def cell_path(fam, cond):
    return os.path.join(RUNS, f"sample_{fam}_{cond}.jsonl")


def done_indices(path):
    """Indices of SUCCESSFUL samples only -- failed records (e.g. v1's 402s)
    must be re-attempted on relaunch."""
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for ln in f:
                try:
                    rec = json.loads(ln)
                except Exception:
                    continue
                if not rec.get("failed"):
                    done.add(rec["i"])
    return done


def one_sample(fam, cond, cfg, i):
    prompt = PROMPTS[cond]
    last_err = None
    for attempt in range(cfg["retries_empty"] + 1):
        try:
            if not budget_guard(fam):
                return dict(i=i, model=cfg["model"], condition=cond,
                            family=fam, failed=True,
                            error="budget cap reached"), "budget_cap"
            with _BUDGET_LOCK:
                _COUNTERS[fam] = _COUNTERS.get(fam, 0) + 1
            ts = time.strftime("%Y-%m-%dT%H:%M:%S%z") or time.strftime("%Y-%m-%dT%H:%M:%S")
            msg, payload = chat(fam, cfg["model"],
                                [{"role": "user", "content": prompt}],
                                temperature=PROTOCOL["temperature"],
                                max_tokens=PROTOCOL["max_tokens"],
                                extra=cfg["extra"], timeout=280)
            text = msg.get("content") or ""
            fin = payload["choices"][0].get("finish_reason")
            if not text.strip() and attempt < cfg["retries_empty"]:
                last_err = "empty_content(finish=%s)" % fin
                continue
            rec = dict(i=i, model=cfg["model"], condition=cond,
                       family=fam, prompt_version="v1",
                       temperature=PROTOCOL["temperature"],
                       requested_at=ts, attempt=attempt,
                       finish_reason=fin, text=text,
                       usage=payload.get("usage"))
            return rec, None
        except APIError as e:
            last_err = str(e)[:160]
            time.sleep(5 * (attempt + 1))
    return dict(i=i, model=cfg["model"], condition=cond, family=fam,
                failed=True, error=last_err), last_err


def run_cell(fam):
    cfg = PROTOCOL["cells"][fam]
    for cond in ("CN", "EU"):
        path = cell_path(fam, cond)
        done = done_indices(path)
        todo = [i for i in range(PROTOCOL["n_per_cell"]) if i not in done]
        print(f"[sample] {fam}/{cfg['model']} {cond}: "
              f"{len(todo)} remaining of {PROTOCOL['n_per_cell']}", flush=True)
        t0 = time.time()
        n_ok = n_fail = 0
        with open(path, "a", encoding="utf-8") as w, \
                ThreadPoolExecutor(max_workers=cfg["workers"]) as ex:
            futures = {ex.submit(one_sample, fam, cond, cfg, i): i
                       for i in todo}
            done_ct = 0
            for fut in futures:
                rec, err = fut.result()
                with LOCK:
                    w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    w.flush()   # real-time progress monitoring
                done_ct += 1
                if rec.get("failed"):
                    n_fail += 1
                else:
                    n_ok += 1
                if done_ct % 50 == 0:
                    el = time.time() - t0
                    rate = done_ct / max(el, 1) * 3600
                    print(f"[sample] {fam} {cond}: {done_ct}/{len(todo)} "
                          f"({rate:.0f}/h, fail={n_fail})", flush=True)
        print(f"[sample] {fam} {cond} DONE ok={n_ok} fail={n_fail} "
              f"in {(time.time()-t0)/60:.1f} min", flush=True)


def write_protocol_manifest():
    man = dict(PROTOCOL, prompt_CN=PROMPT_CN, prompt_EU=PROMPT_EU,
               code_md5={})
    here = os.path.dirname(os.path.abspath(__file__))
    for fn in ("client.py", "gate.py", "abc_parse.py", "sample.py"):
        h = hashlib.md5(open(os.path.join(here, fn), "rb").read())
        man["code_md5"][fn] = h.hexdigest()
    out = os.path.join(RUNS, f"protocol_{PROTOCOL['version']}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    print("[sample] protocol manifest ->", out, flush=True)


def ensure_single_instance():
    """Lock-file guard with stale-PID takeover (no process-name races)."""
    import subprocess
    lock_path = os.path.join(RUNS, ".sampler.lock")
    my_pid = os.getpid()
    if os.path.exists(lock_path):
        try:
            old_pid = int(open(lock_path).read().strip())
        except Exception:
            old_pid = -1
        alive = False
        if old_pid > 0:
            r = subprocess.run(
                ["tasklist", "/FI", f"PID eq {old_pid}", "/FO", "CSV"],
                capture_output=True, text=True).stdout
            alive = f",{old_pid}," in r or f'"{old_pid}"' in r
        if alive and old_pid != my_pid:
            print(f"[sample] another sampler alive (pid={old_pid}); exit",
                  flush=True)
            sys.exit(1)
    with open(lock_path, "w") as f:
        f.write(str(my_pid))


def main():
    ensure_single_instance()
    write_protocol_manifest()
    todo_fams = [f for f in PROTOCOL["order"]]
    only = sys.argv[1:]
    for fam in todo_fams:
        if only and fam not in only:
            continue
        run_cell(fam)
    print("[sample] ALL DONE", flush=True)


if __name__ == '__main__':
    main()
