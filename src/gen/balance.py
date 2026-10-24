# -*- coding: utf-8 -*-
"""Balance / spend check for the three API families.

DeepSeek has a queryable balance endpoint; for DashScope and Zhipu no public
balance API exists, so we report cumulative token spend from the sampling
JSONL (which maps to console billing) and flag when the user should check
the Aliyun / BigModel consoles.

Usage: python balance.py
2026-10-23. Stage C monitoring helper.
"""
import json, os, sys, glob, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from client import load_env

RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))), "runs", "c")

# published CNY list prices (per 1M tokens) -- update if the console differs
PRICE = {
    "deepseek": dict(in_cached=0.5, in_miss=2.0, out=8.0),   # deepseek-v4-pro
    "dashscope": dict(in_cached=0.0, in_miss=2.0, out=6.0),  # qwen3.8-max (est.)
    "zhipu": dict(in_cached=0.0, in_miss=2.0, out=8.0),     # glm-5.3 (est.)
}


def deepseek_balance():
    key = load_env()["DEEPSEEK_API_KEY"]
    req = urllib.request.Request(
        "https://api.deepseek.com/user/balance",
        headers={"Authorization": "Bearer " + key})
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.loads(r.read().decode())
    return d["balance_infos"][0]


def spend_from_jsonl():
    tot = {}
    for fam in ("deepseek", "dashscope", "zhipu"):
        tin = tout = tcache = 0
        n = fail = 0
        for f in glob.glob(os.path.join(RUNS, f"sample_{fam}_*.jsonl")):
            for ln in open(f, encoding="utf-8"):
                try:
                    rec = json.loads(ln)
                except Exception:
                    continue
                if rec.get("failed"):
                    fail += 1
                    continue
                u = rec.get("usage") or {}
                tin += u.get("prompt_tokens", 0)
                tout += u.get("completion_tokens", 0)
                det = u.get("prompt_tokens_details") or {}
                tcache += det.get("cached_tokens", 0)
                n += 1
        p = PRICE[fam]
        cost = ((tin - tcache) * p["in_miss"] + tcache * p["in_cached"]
                + tout * p["out"]) / 1e6
        tot[fam] = dict(samples=n, failed=fail, prompt_tokens=tin,
                      completion_tokens=tout, cached=tcache,
                      est_cost_cny=round(cost, 2))
    return tot


def main():
    print("== balance ==")
    try:
        b = deepseek_balance()
        print(f"deepseek: CNY {b['total_balance']} "
              f"(topped_up {b['topped_up_balance']}, granted {b['granted_balance']})")
    except Exception as e:
        print("deepseek balance query failed:", str(e)[:120])
    print()
    print("== spend from sampling JSONL ==")
    for fam, s in spend_from_jsonl().items():
        print(f"{fam:<10} samples={s['samples']:<5} failed={s['failed']:<4} "
              f"in={s['prompt_tokens']:>9,} out={s['completion_tokens']:>10,} "
              f"est_cost=CNY {s['est_cost_cny']}")
    print()
    print("note: dashscope/zhipu have no public balance API; check the")
    print("Aliyun and BigModel consoles when est_cost approaches your budget.")


if __name__ == '__main__':
    main()
