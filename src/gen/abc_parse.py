# -*- coding: utf-8 -*-
"""ABC notation -> event stream, consistent with Stage A2 conventions.

Output piece record (compatible with build_windows.encode_corpus):
  notes : [dict(onset=Fraction(q), pitch=MIDI int, dur_q=float, token=None)]
  segs  : [(Fraction(0), numerator)]          metre segments
  freemeter : False
  meta  : dict(id=..., key=tonic_pc, mode='major'|'minor'|church mode,
               abc_len=int)

Conventions (mirroring A2):
  - single melodic line only; chords [CEG], multiple voices V:, lyrics w:,
    multi-staff are REJECTED (single-violin requirement);
  - grace notes {..} are absorbed (skipped), as in A2;
  - explicit accidentals persist to the bar line (ABC v1.7);
  - repeats are read as written (no expansion) -- uniform across models;
  - durations quantize onto the same metre grid downstream (A2/A3 reuse).

Compliance = parse OK AND >=8 events AND <=4000 events AND all pitches in
[21,108] AND a K: header AND an M: header.

2026-10-23. Stage C of the seven-stage pipeline.
"""
import re
from fractions import Fraction

NOTE_BASE = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
SHARP_ORDER = ['F', 'C', 'G', 'D', 'A', 'E', 'B']      # +1..+7
FLAT_ORDER = ['B', 'E', 'A', 'D', 'G', 'C', 'F']       # -1..-7

# K: -> (tonic letter, accidental, sharps count); minor handled separately
_MAJOR_SHARPS = {'C': 0, 'G': 1, 'D': 2, 'A': 3, 'E': 4, 'B': 5,
                 'F#': 6, 'C#': 7}
_MAJOR_FLATS = {'F': -1, 'Bb': -2, 'Eb': -3, 'Ab': -4, 'Db': -5,
                'Gb': -6, 'Cb': -7}

_MODE_MINOR = {'m', 'min', 'minor', 'aeo', 'aeolian',
               'dor', 'dorian', 'phr', 'phrygian'}
_MODE_WORD = [('major', 'major'), ('ionian', 'major'), ('mixolydian', 'mixolydian'),
              ('minor', 'minor'), ('aeolian', 'minor'), ('dorian', 'dorian'),
              ('phrygian', 'phrygian'), ('lydian', 'lydian'), ('locrian', 'locrian'),
              ('m', 'minor')]


def parse_key(ktxt):
    """'D' / 'Em' / 'F#m' / 'Bb dorian' / 'C major' -> (tonic_pc, n_acc,
    mode_norm). Raises ValueError if unmappable."""
    ktxt = ktxt.strip()
    m = re.match(r'^([A-Ga-g])([#b]?)(.*)$', ktxt)
    if not m:
        raise ValueError(f"bad key: {ktxt!r}")
    letter, acc, rest = m.group(1).upper(), m.group(2), m.group(3).strip()
    rest_l = rest.lower()
    mode_norm = 'major'
    for word, norm in _MODE_WORD:
        if rest_l.startswith(word):
            mode_norm = norm
            break
    tonic_name = letter + acc
    name = tonic_name
    if mode_norm in _MODE_MINOR:
        # minor keys: signature = major key a minor third up
        rel = {'C': 'Eb', 'C#': 'E', 'Db': 'Fb', 'D': 'F', 'D#': 'F#',
               'Eb': 'Gb', 'E': 'G', 'F': 'Ab', 'F#': 'A', 'Gb': 'Bb',
               'G': 'Bb', 'G#': 'B', 'Ab': 'Cb', 'A': 'C', 'A#': 'C#',
               'Bb': 'Db', 'B': 'D'}
        name = rel.get(tonic_name)
        if name is None:
            raise ValueError(f"unsupported minor key: {ktxt!r}")
        tonic_pc = NOTE_BASE[letter] + (1 if acc == '#' else -1 if acc == 'b' else 0)
        tonic_pc %= 12
    else:
        tonic_pc = NOTE_BASE[letter] + (1 if acc == '#' else -1 if acc == 'b' else 0)
        tonic_pc %= 12
    if name in _MAJOR_SHARPS:
        n_acc = _MAJOR_SHARPS[name]
    elif name in _MAJOR_FLATS:
        n_acc = _MAJOR_FLATS[name]
    else:
        raise ValueError(f"unsupported key: {ktxt!r}")
    if mode_norm in _MODE_MINOR:
        mode_norm = 'minor'
    return tonic_pc, n_acc, mode_norm


def sig_accidentals(n_acc):
    """letter -> +1 sharp / -1 flat from the key signature."""
    acc = {}
    if n_acc > 0:
        for l in SHARP_ORDER[:n_acc]:
            acc[l] = 1
    elif n_acc < 0:
        for l in FLAT_ORDER[:-n_acc]:
            acc[l] = -1
    return acc


TOKEN_RE = re.compile(r"""
    (?P<grace>\{[^}]*\})                      # grace notes -> absorbed
  | (?P<chordsym>\"[^\"]*\")                  # chord symbols -> skipped
  | (?P<acc>__|_|=|\^\^|\^)?(?P<note>[A-Ga-g])
    (?P<oct>[',]*) (?P<dur>(\d+/\d+|\d+/\d*|/\d*|\d+)?)
    (?P<tie>-)?
  | (?P<rest>[zx])(?P<rdur>(\d+/\d+|\d+/\d*|/\d*|\d+)?)(?P<rtie>-)?
  | (?P<brk>[<>])
  | (?P<bar>\|(:)?(:)?\|?|\|?\]|:\|:?)
  | (?P<brack>\[(?:\d|:[^\]]*|\|[^\]]*|[^\]\d|:*][^\]]*\])?) # [1 endings / [K:]
  | (?P<space>\s+)
  | (?P<comment>%[^\n]*)
""", re.VERBOSE)


class ABCError(ValueError):
    pass


def _dur_to_q(durtxt, unit_q):
    """ABC duration text relative to L: -> quarter notes (Fraction)."""
    if not durtxt:
        return unit_q
    t = durtxt
    if '/' in t:
        num, den = t.split('/', 1)
        f = Fraction(1)
        if num:
            f *= int(num)
        if den:
            f /= int(den)
        return unit_q * f
    return unit_q * int(t)


def parse_abc(text):
    """-> (notes, segs, freemeter, meta). Raises ABCError on non-compliance."""
    if not isinstance(text, str) or not text.strip():
        raise ABCError("empty text")
    lines = text.splitlines()

    headers = {}
    body_start = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s or s.startswith('%'):
            continue
        m = re.match(r'^([A-Za-z]):\s*(.*)$', s)
        if m and body_start is None:
            headers[m.group(1).upper()] = m.group(2).strip()
        else:
            body_start = i
            break
    if body_start is None:
        body_start = len(lines)

    for req in ('X', 'M', 'L', 'K'):
        if req not in headers:
            raise ABCError(f"missing header {req}:")
    if headers['M'].upper() in ('FREEMETER', 'NONE', ''):
        raise ABCError("freemeter not supported for generated pieces")
    mm = re.match(r'^(\d+)\s*/\s*(\d+)$', headers['M'])
    if not mm:
        if headers['M'].upper() == 'C':
            numerator = 4
        elif headers['M'].upper() == 'C|':
            numerator = 2
        else:
            raise ABCError(f"unsupported metre: {headers['M']!r}")
    else:
        numerator = int(mm.group(1))

    lt = headers['L']
    ml = re.match(r'^(\d+)\s*/\s*(\d+)$', lt)
    if not ml:
        raise ABCError(f"unsupported L: {lt!r}")
    unit_q = Fraction(4) * Fraction(int(ml.group(1)), int(ml.group(2)))

    tonic_pc, n_acc, mode_norm = parse_key(headers['K'])
    sig = sig_accidentals(n_acc)

    body = "\n".join(lines[body_start:])
    notes = []
    onset = Fraction(0)
    pending_tie = False
    pending_mult = Fraction(1)   # broken rhythm: next note duration factor
    bar_acc = {}       # explicit accidentals within current bar
    prev_end, prev_kind = None, None
    for mo in TOKEN_RE.finditer(body):
        adjacent = (prev_end is not None and mo.start() == prev_end
                    and prev_kind not in ('space', 'comment', None))
        prev_end = mo.end()
        g = mo.group
        # lastgroup cannot be used: trailing empty groups (tie/oct) always
        # "participate". Dispatch on which named group actually matched.
        if g('note') is not None:
            kind = 'note'
        elif g('rest') is not None:
            kind = 'rest'
        elif g('brk') is not None:
            kind = 'brk'
        elif g('bar') is not None:
            kind = 'bar'
        elif g('brack') is not None:
            kind = 'brack'
        elif g('grace') is not None:
            kind = 'grace'
        elif g('chordsym') is not None:
            kind = 'chordsym'
        elif g('comment') is not None:
            kind = 'comment'
        elif g('space') is not None:
            kind = 'space'
        else:
            raise ABCError(f"unparseable token: {mo.group(0)!r}")
        prev_kind = kind
        if kind in ('space', 'comment', 'chordsym', 'grace'):
            continue
        if kind == 'bar':
            bar_acc = {}
            pending_tie = False
            continue
        if kind == 'brack':
            b = mo.group('brack')
            inline = re.match(r'^\[([A-Za-z]):\s*([^\]]*)\]$', b)
            if not inline and re.search(r'[A-Ga-g]', b):
                raise ABCError(f"chord symbol {b!r}: single melodic line required")
            if inline:
                f, v = inline.group(1).upper(), inline.group(2).strip()
                if f == 'K':
                    tonic_pc, n_acc, mode_norm = parse_key(v)
                    sig = sig_accidentals(n_acc)
                    bar_acc = {}
                elif f == 'M':
                    mm = re.match(r'^(\d+)\s*/\s*(\d+)$', v)
                    if not mm:
                        raise ABCError(f"unsupported inline metre {v!r}")
                    # metre change: new segment from current onset
                    _ = (onset, int(mm.group(1)))
                    raise ABCError("inline metre change not supported")
            # [1 / [2 / [|: etc: skip token (music inside is read as written)
            continue
        if kind == 'brk':
            if not notes or not adjacent:
                raise ABCError("broken rhythm must directly follow a note")
            d_last = Fraction(str(notes[-1]['dur_q']))
            if mo.group('brk') == '>':
                notes[-1]['dur_q'] = float(d_last * Fraction(3, 2))
                onset += d_last / 2
                pending_mult = Fraction(1, 2)
            else:
                notes[-1]['dur_q'] = float(d_last / 2)
                onset -= d_last / 2
                pending_mult = Fraction(3, 2)
            continue
        if kind == 'rest':
            dq = _dur_to_q(mo.group('rdur'), unit_q) * pending_mult
            pending_mult = Fraction(1)
            onset += dq
            if mo.group('rtie'):
                pass
            pending_tie = False
            continue
        if kind == 'note':
            acc_t = mo.group('acc')
            letter = mo.group('note')
            upper = letter.isupper()
            letter = letter.upper()
            acc = sig.get(letter, 0)
            if acc_t:
                if acc_t in ('^', '^^'):
                    acc = len(acc_t)
                elif acc_t == '_':
                    acc = -1
                elif acc_t == '__':
                    acc = -2
                elif acc_t == '=':
                    acc = 0
                bar_acc[(letter, upper)] = acc
            else:
                if (letter, upper) in bar_acc:
                    acc = bar_acc[(letter, upper)]
            # ABC standard: uppercase = octave 4 (C = 60), lowercase +12
            base = (60 if upper else 72) + NOTE_BASE[letter]
            for ch in mo.group('oct'):
                base += 12 if ch == "'" else -12
            pitch = base + acc
            dq = _dur_to_q(mo.group('dur'), unit_q) * pending_mult
            pending_mult = Fraction(1)
            if dq <= 0:
                raise ABCError("zero/negative duration")
            if pending_tie and notes and notes[-1]['pitch'] == pitch:
                notes[-1]['dur_q'] = float(notes[-1]['dur_q']) + float(dq)
                notes[-1]['onset'] = Fraction(notes[-1]['onset'])
            else:
                notes.append(dict(onset=Fraction(onset), pitch=int(pitch),
                                  dur_q=float(dq), token=None))
            onset += dq
            pending_tie = bool(mo.group('tie'))
            continue
        raise ABCError(f"unparseable token: {mo.group(0)!r}")

    # strips unterminated garbage? TOKEN_RE finditer skips unmatched chars:
    covered = ''.join(mo.group(0) for mo in TOKEN_RE.finditer(body))
    if re.sub(r'[\s%]', '', body) != re.sub(r'[\s%]', '', covered):
        raise ABCError("unparseable content in body")

    if len(notes) < 8:
        raise ABCError(f"too few events ({len(notes)})")
    if len(notes) > 4000:
        raise ABCError(f"too many events ({len(notes)})")
    if any(nt['pitch'] < 21 or nt['pitch'] > 108 for nt in notes):
        raise ABCError("pitch out of piano range")

    segs = [(Fraction(0), numerator)]
    meta = dict(id=None, key=tonic_pc, mode=mode_norm,
                metre=headers['M'], L=headers['L'], K=headers['K'])
    return notes, segs, False, meta
