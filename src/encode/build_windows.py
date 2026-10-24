# -*- coding: utf-8 -*-
"""
Stage A2/A3: event-stream construction and three-alphabet window encoding.

Corpora
-------
POP909  : 909 Chinese popular-song piano-vocal scores (MELODY track).
          Events rebuilt from MIDI ticks (tpb=480): onset_q = ticks/480.
          Key : dataset annotation key_audio.txt (verified: melody in-scale
          rate 0.96 vs label -> label applies to MIDI content; first segment
          used for multi-segment songs).
          Bars: conductor-track time signatures (numerator = beats per bar).
MTC-FS  : 18,109 Dutch/Flemish folk-song monophonic sequences (Zenodo
          release MTC-FS-INST-2.0_sequences-1.1, CC BY-NC-SA 3.0).
          Events taken from note-level features: onset_q = onsettick/4
          (quarter = 4 ticks in this release); scaledegree+specifier used
          directly for the tonal alphabet.
          Bars: beatinsong (string fraction) modulo timesignature numerator.
          freemeter=True pieces: no bar grid -> fixed-length windows.

Alphabets (frozen rules, manuscript S3.2)
-----------------------------------------
A1 tonal set    : scale-degree+specifier tokens ("3M", "1P", ...). POP909
                  mapping (mode, chromatic offset from tonic) -> token is
                  learned from the MTC convention for cross-corpus
                  consistency.
A2 interval set : signed chromatic interval tokens of consecutive notes
                  *within* a window ("+2", "-7", "0"). Cross-window pairs
                  are never counted (windows only partition the stream).
A3 compound set : f"{interval}|{durClass}", durClass of the second note of
                  the pair in {Whole,Half,Quarter,Eighth,Sixteenth,Other}.

Empty windows are skipped at mining time but counted here (piece stats).

Output: data_backup/derived/<corpus>.windows.jsonl.gz  (derivative data,
NOT pushed to GitHub; see repo README). Manifest: data_backup/manifests/derived.md5.

2026-10-22. Stage A2/A3 of the seven-stage pipeline.
"""
import gzip, json, os, glob, hashlib, collections, bisect
from fractions import Fraction

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
DATA = os.path.join(_ROOT, "data_backup")
POP909 = os.path.join(DATA, "pop909", "POP909")
MTC = os.path.join(DATA, "mtc", "MTC-FS-INST-2.0_sequences-1.1.jsonl.gz")
OUT = os.path.join(DATA, "derived")
os.makedirs(OUT, exist_ok=True)

NOTE_PC = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4,
           'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9,
           'A#': 10, 'Bb': 10, 'B': 11}

DUR_CLASSES = [(4.0, 'Whole'), (2.0, 'Half'), (1.0, 'Quarter'),
               (0.5, 'Eighth'), (0.25, 'Sixteenth')]

def dur_class(dur_q):
    for th, name in DUR_CLASSES:
        if dur_q >= th - 1e-9:
            return name
    return 'Other'

# ------------------------------------------------------------- bar grid ----

def bar_grid(upto, segs):
    """Bar-start positions up to and including `upto`.
    segs: [(start_pos, numerator), ...] time-signature segments, positions in
    quarter notes. A segment switch takes effect at its exact onset (time-
    signature changes are assumed to fall on bar lines in these corpora)."""
    segs = sorted((Fraction(str(s)), int(n)) for s, n in segs)
    grid, pos, si = [Fraction(0)], Fraction(0), 0
    num = segs[0][1]
    while grid[-1] <= upto:
        nxt = segs[si + 1][0] if si + 1 < len(segs) else None
        if nxt is not None and pos + num > nxt:
            pos = nxt
            si += 1
            num = segs[si][1]
        else:
            pos += num
        grid.append(pos)
    return grid

def split_windows(onsets, grid):
    """Partition note indices into windows by bar (or fixed grid)."""
    wins, cur, bid = [], [], 0
    for i, o in enumerate(onsets):
        b = bisect.bisect_right(grid, o) - 1
        if b != bid or not cur:
            if cur:
                wins.append(cur)
            cur, bid = [], b
        cur.append(i)
    if cur:
        wins.append(cur)
    return wins

# --------------------------------------------------------------- alphabets ----

def code_window(idx, notes):
    """One window -> (A1 tokens, A2 tokens, A3 tokens) as sorted lists."""
    t1, t2, t3 = set(), set(), set()
    prev = None
    for i in idx:
        nt = notes[i]
        if nt['token'] is not None:
            t1.add(nt['token'])
        if prev is not None:
            iv = f"{nt['pitch'] - prev['pitch']:+d}"
            t2.add(iv)
            t3.add(f"{iv}|{dur_class(nt['dur_q'])}")
        prev = nt
    return sorted(t1), sorted(t2), sorted(t3)

# ------------------------------------------------------------------ MTC ----

def mtc_records():
    with gzip.open(MTC, 'rt', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)

def learn_mtc_tonal_map(sample=4000):
    """(mode, chromatic offset from tonic) -> 'degree+specifier', learned
    from MTC itself so POP909 coding follows the same convention."""
    tbl = collections.defaultdict(collections.Counter)
    for k, rec in enumerate(mtc_records()):
        if k >= sample:
            break
        ft = rec['features']
        for i in range(len(ft['midipitch'])):
            sd, sp, tn, md = (ft['scaledegree'][i], ft['scaledegreespecifier'][i],
                              ft['tonic'][i], ft['mode'][i])
            if sd is None or tn not in NOTE_PC:
                continue
            off = (ft['midipitch'][i] - NOTE_PC[tn]) % 12
            tbl[(md, off)][f"{sd}{sp}"] += 1
    return {k: v.most_common(1)[0][0] for k, v in tbl.items()}

def mtc_notes(rec):
    ft = rec['features']
    n = len(ft['midipitch'])
    notes = []
    for i in range(n):
        df = ft['duration_frac'][i]
        notes.append(dict(
            onset=Fraction(ft['onsettick'][i]) / 4,
            pitch=int(ft['midipitch'][i]),
            dur_q=(float(Fraction(df)) * 4 if isinstance(df, str)
                   else float(ft['duration'][i])),
            token=(None if ft['scaledegree'][i] is None
                   else f"{ft['scaledegree'][i]}{ft['scaledegreespecifier'][i]}"),
        ))
    segs = []
    cur_ts, seg_start, ts_missing = None, Fraction(0), False
    for i in range(n):
        ts = ft['timesignature'][i]
        if ts is None:
            ts_missing = True
            continue
        num = int(ts.split('/')[0])
        o = Fraction(ft['onsettick'][i]) / 4
        if cur_ts is None:
            cur_ts, seg_start = num, o
        elif num != cur_ts:
            segs.append((seg_start, cur_ts))
            cur_ts, seg_start = num, o
    if ts_missing or not segs:
        freemeter = True
        segs = [(Fraction(0), 4)]
    else:
        segs.append((seg_start, cur_ts))
        freemeter = bool(rec.get('freemeter'))
    return notes, segs, freemeter

# ---------------------------------------------------------------- POP909 ----

def pop909_files():
    files = sorted(glob.glob(os.path.join(POP909, '*', '*.mid')))
    return [f for f in files
            if os.path.basename(f)[:-4] == os.path.basename(os.path.dirname(f))]

def pop909_key(song_dir):
    ka = os.path.join(song_dir, 'key_audio.txt')
    line = open(ka, encoding='utf-8', errors='replace').read().strip().splitlines()[0]
    tonic, mode = line.split()[-1].split(':')
    return NOTE_PC[tonic] % 12, ('major' if mode == 'maj' else 'minor')

def pop909_notes(path):
    import mido
    m = mido.MidiFile(path)
    tpb = m.ticks_per_beat
    segs, t = [], 0
    for x in m.tracks[0]:
        t += x.time
        if x.type == 'time_signature':
            segs.append((t / tpb, x.numerator))
    if not segs:
        segs = [(0.0, 4)]
    tr = [tr for tr in m.tracks if any(
        e.type == 'track_name' and e.name == 'MELODY' for e in tr)][0]
    raw, t, active = [], 0, {}
    for x in tr:
        t += x.time
        if x.type == 'note_on' and x.velocity > 0:
            if x.note in active:
                s = active.pop(x.note)
                raw.append((s, x.note, t - s))
            active[x.note] = t
        elif x.type in ('note_off', 'note_on') and (x.type == 'note_off' or x.velocity == 0):
            nn = getattr(x, 'note', None)
            if nn in active:
                s = active.pop(nn)
                raw.append((s, nn, t - s))
    for nn, s in active.items():
        raw.append((s, nn, t - s))
    raw.sort(key=lambda z: (z[0], z[1]))
    notes = [dict(onset=Fraction(s, tpb), pitch=int(p), dur_q=du / tpb,
                  token=None) for s, p, du in raw]
    return notes, segs

# ------------------------------------------------------------------ main ----

def encode_corpus(out_path, pieces, filt=None):
    """filt(notes) -> (kept: bool, reason: str|None). Excluded pieces are
    still written (kept=false) for provenance; downstream stages filter."""
    stats = collections.Counter()
    with gzip.open(out_path, 'wt', encoding='utf-8') as w:
        for p in pieces:
            notes, segs, freemeter, meta = p
            kept, reason = (True, None) if filt is None else filt(notes)
            if kept:
                upto = max((nt['onset'] for nt in notes), default=Fraction(0))
                grid = ([Fraction(4 * i) for i in range(int(upto / 4) + 2)]
                        if freemeter else bar_grid(upto, segs))
                wins = split_windows([nt['onset'] for nt in notes], grid)
                a1, a2, a3 = [], [], []
                for idx in wins:
                    x1, x2, x3 = code_window(idx, notes)
                    a1.append(x1); a2.append(x2); a3.append(x3)
                stats['pieces'] += 1
                stats['windows'] += len(wins)
                n_empty = sum(1 for x in a1 if not x)
                stats['empty'] += n_empty
                rec = dict(meta, freemeter=freemeter, n_events=len(notes),
                           n_windows=len(wins), n_empty=n_empty, kept=True,
                           a1=a1, a2=a2, a3=a3)
            else:
                stats['excluded_' + reason] += 1
                rec = dict(meta, freemeter=freemeter, n_events=len(notes),
                           n_windows=0, n_empty=0, kept=False, reason=reason,
                           a1=[], a2=[], a3=[])
            w.write(json.dumps(rec) + "\n")
    return stats

def sig_of(notes):
    """Exact event-stream signature (onset, chromatic pitch, duration).
    Used for the manuscript's conservative dedup rule (within corpus only;
    cross-corpus duplicates are retained and counted)."""
    return tuple((str(nt['onset']), nt['pitch'], nt['dur_q']) for nt in notes)

def mtc_pieces(tonal_map_unused=None):
    for rec in mtc_records():
        notes, segs, freemeter = mtc_notes(rec)
        meta = dict(id=rec.get('id'), family=rec.get('tunefamily'),
                    year=rec.get('year'), type=rec.get('type'))
        yield notes, segs, freemeter, meta

def pop_pieces(tonal_map):
    unmapped = collections.Counter()
    for f in pop909_files():
        d = os.path.dirname(f)
        pid = os.path.basename(f)[:-4]
        tonic_pc, mode = pop909_key(d)
        notes, segs = pop909_notes(f)
        def tok(nt, tonic_pc=tonic_pc, mode=mode):
            t = tonal_map.get((mode, (nt['pitch'] - tonic_pc) % 12))
            if t is None:
                unmapped[(mode, (nt['pitch'] - tonic_pc) % 12)] += 1
            return t
        for nt in notes:
            nt['token'] = tok(nt)
        meta = dict(id=pid, key=tonic_pc, mode=mode)
        yield notes, segs, False, meta
    if unmapped:
        print("          POP909 UNMAPPED (mode,offset):",
              dict(unmapped.most_common(10)))

def main():
    print("[A2/A3] learning MTC tonal convention ...")
    tonal_map = learn_mtc_tonal_map()
    print(f"          {len(tonal_map)} (mode,offset) conventions")

    def make_filt():
        seen = {}
        def filt(notes):
            if len(notes) < 8:
                return False, 'short'
            sig = sig_of(notes)
            if sig in seen:
                return False, 'dup'
            seen[sig] = True
            return True, None
        return filt

    print("[A2/A3] encoding MTC ...")
    s = encode_corpus(os.path.join(OUT, 'mtc.windows.jsonl.gz'),
                      mtc_pieces(), make_filt())
    print(f"          kept={s['pieces']} windows={s['windows']} "
          f"excluded={ {k: v for k, v in s.items() if k.startswith('excluded')} }")

    print("[A2/A3] encoding POP909 ...")
    s = encode_corpus(os.path.join(OUT, 'pop909.windows.jsonl.gz'),
                      pop_pieces(tonal_map), make_filt())
    print(f"          kept={s['pieces']} windows={s['windows']} "
          f"excluded={ {k: v for k, v in s.items() if k.startswith('excluded')} }")

    man = os.path.join(DATA, 'manifests', 'derived.md5')
    os.makedirs(os.path.dirname(man), exist_ok=True)
    with open(man, 'w', encoding='utf-8') as w:
        for fn in sorted(os.listdir(OUT)):
            p = os.path.join(OUT, fn)
            if not os.path.isfile(p):
                continue
            h = hashlib.md5()
            with open(p, 'rb') as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b''):
                    h.update(chunk)
            w.write(f"{h.hexdigest()}  {fn}\n")
    print("[A2/A3] manifest ->", man)
    print("[A2/A3] DONE")

if __name__ == '__main__':
    main()
