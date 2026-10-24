# -*- coding: utf-8 -*-
"""
Stage A4: mine the reference pattern libraries (P, pi) x 3 alphabets x 2 corpora.

Protocol (frozen):
  * Floor semantics: a pattern survives iff its WINDOW-level frequency >= 2
    (present in >= 2 windows). support passed to the engine is 2/N x (1-1e-12)
    so that the engine's real-valued comparison freq >= N*support admits
    freq == 2 exactly (the A-paper adapter documented the strict-> boundary).
  * Engine: TensorFIM CoParaCG_baseline.exe (CPU bitmap), called UNMODIFIED;
    its output MFIs are independently cross-verified against a certified
    ground truth obtained by exhaustive per-row subset enumeration
    (windows carry <= ~12 items, so subset enumeration is exact and cheap).
  * pi (document frequency) is PIECE-level: a pattern scores once per piece
    containing it in >=1 window. Window freq is discovery support only.
  * Determinism check: engine run twice on mtc.a1, outputs compared.

Outputs (data_backup/derived/, derivative data -- NOT pushed to GitHub):
  reflib.<corpus>.<alph>.json.gz  {meta, items, mfi[[ids,win_freq,piece_freq]],
                                   census{"i,j,k":[win_freq,piece_freq]}}
Runs and manifests: repo/runs/a4/<corpus>.<alph>/ (runs/ is gitignored).

2026-10-22.
"""
import gzip, json, os, re, subprocess, sys, time, hashlib, collections
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]          # git repo root (…/Manuscript_work/repo)
WORK = REPO.parent                                   # …/Manuscript_work
DATA = WORK / "data_backup"
DERIVED = DATA / "derived"
RUNS = REPO / "runs" / "a4"
EXE = REPO / "engines" / "tensorfim" / "run" / "CoParaCG_baseline.exe"

LINE = re.compile(r"^\d+th:\s+(\d+)\s+\{\s*([^}]*)\}\s+support:\s+\S+\s+frequency:\s+(\d+)")

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()

def load_windows(corpus, alph):
    src = DERIVED / f"{corpus}.windows.jsonl.gz"
    mp = DERIVED / "fimi" / f"{corpus}.{alph}.items.json"
    id_of = {t: int(i) for i, t in
             json.loads(mp.read_text(encoding="utf-8"))["token_of_id"].items()}
    pieces = []
    with gzip.open(src, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if not r.get("kept", True):
                continue          # short (<8 events) or exact duplicate
            pieces.append((r["id"], [[id_of[t] for t in w] for w in r[alph]]))
    return pieces

def certified_truth(pieces, key="id"):
    """Exhaustive subset enumeration with int bitmasks (ids 1..182 -> bits).
    Returns (win_freq{mask:n}, piece_freq{mask:n}, maximal[mask]) for ALL
    patterns with window frequency >= 2. Exact by construction.

    Maximality criterion used: k (frequent) is NON-maximal iff adding any
    single item keeps frequency >= 2 (extension monotonicity: if a frequent
    proper superset exists, some single-item extension of k is frequent too).
    Check cost O(|freq| x vocab) instead of pairwise subset tests."""
    win_freq = collections.Counter()
    piece_freq = collections.Counter()
    for pid, wins in pieces:
        ps = set()
        for w in wins:
            subs = [0]
            for b in (1 << i for i in w):
                subs += [x | b for x in subs]
            for k in subs[1:]:
                win_freq[k] += 1
                ps.add(k)
        for k in ps:
            piece_freq[k] += 1
    freqset = {k for k, c in win_freq.items() if c >= 2}
    maximal = []
    vocab_mask = 0
    for k in freqset:
        vocab_mask |= k
    for k in freqset:
        missing = vocab_mask ^ (k & vocab_mask)
        sup = False
        while missing:
            lb = missing & -missing
            if (k | lb) in freqset:
                sup = True
                break
            missing ^= lb
        if not sup:
            maximal.append(k)
    freq = {k: win_freq[k] for k in freqset}
    return freq, piece_freq, maximal

def run_engine(fimi, support, run_dir, tag=""):
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    local = run_dir / "input.fimi"
    if not local.exists() or sha256(local) != sha256(fimi):
        import shutil; shutil.copyfile(fimi, local)
    result = run_dir / f"{local.name}-{support:.6f}=Results.txt"
    if result.exists():
        result.unlink()
    t0 = time.time()
    with open(run_dir / "stdout.log", "wb") as lf:
        subprocess.run([str(EXE), str(local), repr(support)],
                       cwd=run_dir, stdout=lf, stderr=subprocess.STDOUT, timeout=3600)
    wall = time.time() - t0
    if not result.exists():
        raise RuntimeError(f"engine produced no result; see {run_dir}/stdout.log")
    mfis = []
    with open(result, encoding="utf-8", errors="replace") as f:
        for ln in f:
            m = LINE.match(ln.strip())
            if m:
                mfis.append((frozenset(int(x) for x in m.group(2).split()),
                             int(m.group(3))))
    man = dict(engine="tensorfim_bitmap CoParaCG_baseline.exe",
               exe_sha256=sha256(EXE), input_sha256=sha256(local),
               support=support, wall_secs=round(wall, 3), tag=tag,
               result=str(result.name), created=time.strftime("%Y-%m-%dT%H:%M:%S"))
    (run_dir / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    return mfis, man

def mine(corpus, alph, verify=True):
    print(f"[A4] {corpus}.{alph}")
    pieces = load_windows(corpus, alph)
    n_win = sum(len(w) for _, w in pieces)
    n_pieces = len(pieces)
    items = json.loads((DERIVED / "fimi" / f"{corpus}.{alph}.items.json").read_text(encoding="utf-8"))
    tok_of = {int(k): v for k, v in items["token_of_id"].items()}
    fimi = DERIVED / "fimi" / f"{corpus}.{alph}.fimi"
    support = (2.0 / n_win) * (1 - 1e-12)
    rdir = RUNS / f"{corpus}.{alph}"
    mfis, man = run_engine(fimi, support, rdir)
    print(f"     engine: {len(mfis)} MFIs  ({man['wall_secs']}s)")
    # determinism: second run on mtc.a1
    if (corpus, alph) == ("mtc", "a1"):
        mfis2, _ = run_engine(fimi, support, RUNS / "mtc.a1.repeat", tag="repeat")
        same = sorted((sorted(s), c) for s, c in mfis) == sorted((sorted(s), c) for s, c in mfis2)
        print(f"     determinism re-run identical: {same}")
        (rdir / "determinism.json").write_text(json.dumps({"identical": same}), encoding="utf-8")
    truth = None
    if verify:
        freq, pfreq, maximal = certified_truth(pieces)
        truth = (freq, pfreq, maximal)
        eng = {}
        for s, c in mfis:
            k = 0
            for i in s:
                k |= 1 << i
            eng[k] = c
        tru = {k: freq[k] for k in maximal}
        same_sets = set(eng) == set(tru)
        freq_ok = all(eng[k] == tru[k] for k in eng) if same_sets else False
        print(f"     truth: {len(tru)} maximal / {len(freq)} frequent sets >=2; "
              f"engine==truth: {same_sets and freq_ok}")
        (rdir / "verify.json").write_text(json.dumps({
            "engine_n": len(eng), "truth_maximal_n": len(tru),
            "truth_frequent_n": len(freq), "sets_equal": same_sets,
            "freqs_equal": bool(freq_ok)}), encoding="utf-8")
        out_freq, out_pfreq, out_max = freq, pfreq, maximal
    else:
        out_freq = {}
        out_pfreq = collections.Counter()
        out_max = []

    def ids_of(k):
        return [i for i in range(1, 200) if k >> i & 1]

    # emit reflib (ids everywhere; decode via items); verify is always on
    mfi_rows = [[ids_of(k), sorted(tok_of[i] for i in ids_of(k)),
                 out_freq.get(k, c), truth[1].get(k, 0)]
                for k, c in eng.items()]
    census = {"|".join(map(str, ids_of(k))): [c, truth[1].get(k, 0)]
              for k, c in sorted(out_freq.items())}
    rec = dict(corpus=corpus, alphabet=alph, n_windows=n_win, n_pieces=n_pieces,
               floor="window_freq>=2", support_arg=support,
               engine_manifest=man, items={str(k): v for k, v in sorted(tok_of.items())},
               mfi=mfi_rows, census=census)
    out = DERIVED / f"reflib.{corpus}.{alph}.json.gz"
    with gzip.open(out, "wt", encoding="utf-8") as w:
        json.dump(rec, w, ensure_ascii=False)
    print(f"     -> {out.name}  mfi={len(mfi_rows)} census={len(census)}")
    return dict(corpus=corpus, alph=alph, n_mfi=len(mfi_rows), n_census=len(census),
                wall=man["wall_secs"])

if __name__ == "__main__":
    todo = [("mtc", "a1"), ("mtc", "a2"), ("mtc", "a3"),
            ("pop909", "a1"), ("pop909", "a2"), ("pop909", "a3")]
    if len(sys.argv) == 3:
        todo = [(sys.argv[1], sys.argv[2])]
    summary = [mine(c, a) for c, a in todo]
    print(json.dumps(summary, indent=1))
