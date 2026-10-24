# -*- coding: utf-8 -*-
"""Stage F step 2: seeded sampling (protocol 6a / amendment F-1).

Cells: per family x mechanism x arm x condition x dose c in {0.1,0.25,0.5,1.0},
n=100 each; c=0 reuses stage-C pools (no new calls). Mechanisms:
  M1 continuation priming  (own seeds only): seeds rendered as the opening,
                             model continues the body;
  M2 notation injection    (own and cross): motifs listed as mandatory;
  M3 rejection sampling    is NOT sampled here (stage F step 4 selects from
                             the frozen stage-C pools).

Resumable per cell (runs/f/sample_<cell>.jsonl), budget guard per family,
single-instance lock. Protocol manifest -> runs/f/protocol_f.json.

Launch (detached):  cmd /c start "samplerF" /min python sample_f.py

2026-10-24.
"""
import json, os, sys, time, hashlib, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from client import chat, APIError
from gate import PROMPT_CN, PROMPT_EU

WORK = Path(__file__).resolve().parents[3]
RUNF = WORK / "runs" / "f"
RUNF.mkdir(parents=True, exist_ok=True)

PROTOCOL = {
    "version": "f1", "date": "2026-10-24",
    "design": "docs/experiment_plan.md S6a (amendment F-1)",
    "n_per_cell": 100, "temperature": 1.0, "max_tokens": 1200,
    "doses": (0.1, 0.25, 0.5, 1.0),
    "mechanisms": ("M1", "M2"),
    "cells_cfg": {
        "deepseek": dict(model="deepseek-chat", workers=8, extra=None,
                         retries_empty=3,
                         price=dict(in_per_m=2.0, out_per_m=8.0), cap_cny=32),
        "dashscope": dict(model="qwen-flash", workers=8, extra=None,
                          retries_empty=3,
                          price=dict(in_per_m=0.5, out_per_m=2.0), cap_cny=12),
        "zhipu": dict(model="glm-4-air", workers=8, extra=None,
                      retries_empty=3,
                      price=dict(in_per_m=1.0, out_per_m=1.0), cap_cny=12),
    },
    "order": ["deepseek", "dashscope", "zhipu"],
}

PROMPTS = {"CN": PROMPT_CN, "EU": PROMPT_EU}
WORST_IN_TOK = 1200
_COUNTERS = {}
_BUDGET_LOCK = threading.Lock()
LOCK = threading.Lock()


def budget_guard(fam):
    cfg = PROTOCOL["cells_cfg"][fam]
    with _BUDGET_LOCK:
        n = _COUNTERS.get(fam, 0)
    per_call = (WORST_IN_TOK * cfg["price"]["in_per_m"]
                + PROTOCOL["max_tokens"] * cfg["price"]["out_per_m"]) / 1e6
    return n * per_call < cfg["cap_cny"]


def load_seeds():
    seeds = {}
    for culture in ("CN", "EU"):
        p = RUNF / f"seeds_{culture}.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        seeds[culture] = {c: v for c, v in d["cells"].items()}
    return seeds


SEEDS = load_seeds()


def build_prompt(mech, arm, cond, c):
    seed_culture = cond if arm == "own" else ("EU" if cond == "CN" else "CN")
    cell = SEEDS[seed_culture][str(c)]
    frags = cell["abc_fragments"]
    if mech == "M1":       # continuation priming (own only)
        opening = " | ".join(frags)
        prompt = (PROMPTS[cond]
                  + "\n\nHere is the opening of the tune, written for you. "
                    "Continue it in exactly the same style, for another 8 to "
                    "12 bars. Output ONLY the ABC body lines of the "
                    "continuation -- no header lines (X:, T:, M:, L:, K:), "
                    "no explanation, no Markdown:\n\n"
                    "X:1\nT:Seeded Opening\nM:4/4\nL:1/4\nK:C\n" + opening + " |")
    else:                  # M2 notation injection (own + cross)
        listing = "\n".join(f"  {i+1}. {f}" for i, f in enumerate(frags))
        prompt = (PROMPTS[cond]
                  + "\n\nMandatory motifs: the melody MUST contain each of the "
                    "following motifs at least once. Transposition to any pitch "
                    "level is allowed; keep the given note values.\n" + listing
                  + "\nNow output the complete tune as instructed above (ABC "
                    "text only).")
    return prompt


def cell_name(fam, mech, arm, cond, c):
    return f"{fam}_{mech}_{arm}_{cond}_{c}"


def cell_path(cell):
    return RUNF / f"sample_{cell}.jsonl"


def done_indices(path):
    done = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for ln in f:
                try:
                    rec = json.loads(ln)
                except Exception:
                    continue
                if not rec.get("failed"):
                    done.add(rec["i"])
    return done


def one_sample(fam, cell, mech, arm, cond, c, i):
    cfg = PROTOCOL["cells_cfg"][fam]
    prompt = build_prompt(mech, arm, cond, c)
    last_err = None
    for attempt in range(cfg["retries_empty"] + 1):
        try:
            if not budget_guard(fam):
                return dict(i=i, cell=cell, model=cfg["model"], family=fam,
                            mech=mech, arm=arm, condition=cond, dose=c,
                            failed=True, error="budget cap reached"), "budget_cap"
            with _BUDGET_LOCK:
                _COUNTERS[fam] = _COUNTERS.get(fam, 0) + 1
            ts = time.strftime("%Y-%m-%dT%H:%M:%S")
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
            return dict(i=i, cell=cell, model=cfg["model"], family=fam,
                        mech=mech, arm=arm, condition=cond, dose=c,
                        temperature=PROTOCOL["temperature"],
                        requested_at=ts, attempt=attempt,
                        finish_reason=fin, text=text,
                        usage=payload.get("usage")), None
        except APIError as e:
            last_err = str(e)[:160]
            time.sleep(5 * (attempt + 1))
    return dict(i=i, cell=cell, model=cfg["model"], family=fam, mech=mech,
                arm=arm, condition=cond, dose=c, failed=True,
                error=last_err), last_err


def run_cell(fam, mech, arm, cond, c):
    cfg = PROTOCOL["cells_cfg"][fam]
    cell = cell_name(fam, mech, arm, cond, c)
    path = cell_path(cell)
    done = done_indices(path)
    todo = [i for i in range(PROTOCOL["n_per_cell"]) if i not in done]
    if not todo:
        print(f"[F] {cell} complete", flush=True)
        return
    print(f"[F] {cell}: {len(todo)} remaining", flush=True)
    t0 = time.time()
    n_ok = n_fail = 0
    with open(path, "a", encoding="utf-8") as w, \
            ThreadPoolExecutor(max_workers=cfg["workers"]) as ex:
        futures = {ex.submit(one_sample, fam, cell, mech, arm, cond, c, i): i
                   for i in todo}
        for fut in futures:
            rec, err = fut.result()
            with LOCK:
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                w.flush()
            if rec.get("failed"):
                n_fail += 1
            else:
                n_ok += 1
    print(f"[F] {cell} DONE ok={n_ok} fail={n_fail} "
          f"in {(time.time()-t0)/60:.1f} min", flush=True)


def write_manifest():
    man = dict(PROTOCOL, prompt_CN=PROMPT_CN, prompt_EU=PROMPT_EU,
               seed_md5={c: hashlib.md5((RUNF / f"seeds_{c}.json")
                                        .read_bytes()).hexdigest()
                         for c in ("CN", "EU")},
               code_md5={})
    for fn in ("client.py", "gate.py", "abc_parse.py", "sample_f.py"):
        h = hashlib.md5((Path(HERE) / fn).read_bytes())
        man["code_md5"][fn] = h.hexdigest()
    (RUNF / "protocol_f.json").write_text(
        json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[F] protocol manifest ->", RUNF / "protocol_f.json", flush=True)


def ensure_single_instance():
    import subprocess
    lock_path = RUNF / ".sampler_f.lock"
    my_pid = os.getpid()
    if lock_path.exists():
        try:
            old_pid = int(lock_path.read_text().strip())
        except Exception:
            old_pid = -1
        alive = False
        if old_pid > 0:
            r = subprocess.run(["tasklist", "/FI", f"PID eq {old_pid}", "/FO", "CSV"],
                               capture_output=True, text=True).stdout
            alive = f",{old_pid}," in r or f'"{old_pid}"' in r
        if alive and old_pid != my_pid:
            print(f"[F] another sampler alive (pid={old_pid}); exit", flush=True)
            sys.exit(1)
    lock_path.write_text(str(my_pid))


def main():
    ensure_single_instance()
    write_manifest()
    for fam in PROTOCOL["order"]:
        for mech in PROTOCOL["mechanisms"]:
            arms = ("own",) if mech == "M1" else ("own", "cross")
            for arm in arms:
                for cond in ("CN", "EU"):
                    for c in PROTOCOL["doses"]:
                        run_cell(fam, mech, arm, cond, c)
    print("[F] ALL CELLS DONE", flush=True)


if __name__ == "__main__":
    main()
