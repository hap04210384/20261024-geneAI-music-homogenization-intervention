# -*- coding: utf-8 -*-
"""Stage C go/no-go gate: ABC-notation compliance rate of cheap-tier models.

Two cultural conditions, one prompt template each (frozen below, v1):
  CN : Chinese descriptor (zh) -- DeepSeek/Qwen native-language advantage
  EU : European descriptor (en) -- same ABC container, culture via style words
Gate rule (experiment_plan.md S3): model-condition combos with parse failure
rate > 20% get post-processing or are excluded from formal sampling.

Outputs raw model replies to runs/c/gate/ and a compliance report JSON.
2026-10-23. Stage C of the seven-stage pipeline.
"""
import json, os, re, sys, time, collections
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from client import chat, APIError, API_KEYS
from abc_parse import parse_abc, ABCError

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
RUNS = os.path.join(ROOT, "runs", "c")
os.makedirs(RUNS, exist_ok=True)

N_PER_CELL = 20   # gate sample size per model x condition

CANDIDATE_CHEAP = {
    "deepseek": ["deepseek-v4-flash", "deepseek-flash", "deepseek-chat"],
    "dashscope": ["qwen3.8-flash", "qwen-flash", "qwen-turbo"],
    "zhipu": ["glm-5.3-flash", "glm-4-flash", "glm-4-air"],
}

PROMPT_CN = (
    "请创作一段中国风格的单声部旋律（传统民歌或五声音阶流行风格），用 ABC 记谱法输出。"
    "严格要求：只输出 ABC 文本本身，以 X:、T:、M:、L:、K: 五行为头部；"
    "必须是单旋律线，禁止和弦、歌词、伴奏声部；长度约 12 到 16 小节；"
    "调性与拍号自选。不要输出任何解释或 Markdown 标记。"
)
PROMPT_EU = (
    "Compose a single-voice melody in the style of a traditional European "
    "(e.g. Celtic, German, Balkan) folk song, written in ABC notation. "
    "Strict requirements: output ONLY the ABC text itself, starting with "
    "header lines X:, T:, M:, L:, K:; it must be a single melodic line -- "
    "no chords, no lyrics, no accompaniment staves; about 12 to 16 bars; "
    "choose any key and metre. No explanations, no Markdown fences."
)

FENCE_RE = re.compile(r"^```(?:abc|ABC)?\s*|\s*```$", re.MULTILINE)


def extract_abc(text):
    """Strip code fences / leading chatter conservatively, then normalize
    the common LLM ABC dialect 'F#' -> '^F', 'Bb' -> '_B' in body lines
    (header lines are left untouched; K:F#m is parsed natively)."""
    t = text.strip()
    m = re.search(r"X:\s*\d", t)
    if m:
        t = t[m.start():]
    # cut trailing fence / chatter after the music ends
    lines = t.splitlines()
    while lines and lines[-1].strip().startswith("```"):
        lines.pop()
    while lines and not lines[-1].strip():
        lines.pop()
    out = []
    for ln in lines:
        if re.match(r"^[A-Za-z]:\s*", ln):
            out.append(ln)     # header line: keep verbatim
        else:
            ln = re.sub(r"([A-Ga-g])#", r"^\1", ln)
            ln = re.sub(r"([A-Ga-g])b", r"_\1", ln)
            out.append(ln)
    return "\n".join(out)


def probe_models():
    """Find the first working cheap model per family. Returns {fam: model}."""
    out = {}
    for fam, cands in CANDIDATE_CHEAP.items():
        for mdl in cands:
            try:
                msg, _ = chat(fam, mdl, [{"role": "user", "content": "hi"}],
                              max_tokens=4, max_retries=1)
                out[fam] = mdl
                print(f"[probe] {fam}: {mdl} OK")
                break
            except APIError as e:
                print(f"[probe] {fam}: {mdl} FAIL ({str(e)[:90]})")
        if fam not in out:
            print(f"[probe] {fam}: no cheap model reachable")
    return out


def run_cell(fam, model, cond, prompt, workers=8, count=None, start=0):
    """n samples -> compliance stats; raw replies saved. `start` offsets the
    sample index so a slow cell can be finished across several runs; a
    merge step then aggregates batches of the same fam/cond/date."""
    if count is None:
        count = N_PER_CELL
    # Zhipu glm-5.x are reasoning models that would burn the whole budget on
    # thinking and return empty content; cap the thinking budget (frozen).
    extra = ({"thinking": {"type": "enabled", "budget_tokens": 1000}}
             if fam == "zhipu" else None)
    def one(i):
        try:
            msg, payload = chat(fam, model,
                                [{"role": "user", "content": prompt}],
                                temperature=1.0, max_tokens=8000,
                                extra=extra)
            text = msg.get("content") or ""
        except APIError as e:
            return dict(i=i, error=str(e)[:200]), f"api:{str(e)[:40]}"
        try:
            parse_abc(extract_abc(text))
            err = None
        except ABCError as e:
            err = str(e)[:60]
        return dict(i=i, text=text, usage=payload.get("usage")), err

    ok, fail_reasons, replies = 0, collections.Counter(), []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for rec, err in ex.map(one, range(start, start + count)):
            replies.append(rec)
            if err is None:
                ok += 1
            else:
                fail_reasons[err] += 1
    replies.sort(key=lambda r: r["i"])
    return dict(model=model, condition=cond, n=count, start=start,
                compliant=ok, rate=round(ok / count, 3),
                failures=dict(fail_reasons), replies=replies)


def main():
    argv = sys.argv[1:]
    # usage: gate.py [--start S] [--count C] [family condition ...]
    # (default: all six cells, full N_PER_CELL each)
    start = int(argv[argv.index("--start") + 1]) if "--start" in argv else 0
    count = int(argv[argv.index("--count") + 1]) if "--count" in argv else None
    argv = [a for a in argv if a not in ("--start", "--count")
            and not a.isdigit()]
    only = set(tuple(argv[i:i + 2]) for i in range(0, len(argv) - 1, 2))
    print("[gate] probing cheap-tier models ...")
    models = probe_models()
    if len(models) < 3:
        print("[gate] WARNING: not all families have a cheap model")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    report = dict(date=time.strftime("%Y-%m-%d"), gate_n=N_PER_CELL,
                  prompt_versions=dict(CN="v1", EU="v1"), cells=[])
    for fam, model in models.items():
        for cond, prompt in (("CN", PROMPT_CN), ("EU", PROMPT_EU)):
            if only and (fam, cond) not in only:
                continue
            print(f"[gate] {fam}/{model} {cond} [{start}:{start+(count or N_PER_CELL)}] ...")
            cell = run_cell(fam, model, cond, prompt, count=count, start=start)
            report["cells"].append({k: v for k, v in cell.items()
                                    if k != "replies"})
            raw_path = os.path.join(RUNS, f"gate_{fam}_{cond}_{stamp}_s{start}.json")
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(cell, f, ensure_ascii=False, indent=1)
            print(f"       rate={cell['rate']} failures={cell['failures']}")
    rep_path = os.path.join(RUNS, f"gate_report_{stamp}.json")
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("[gate] report ->", rep_path)
    print(json.dumps(report["cells"], ensure_ascii=False, indent=1))

if __name__ == '__main__':
    main()
