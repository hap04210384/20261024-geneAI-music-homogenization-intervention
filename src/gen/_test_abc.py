# -*- coding: utf-8 -*-
"""Quick self-test for abc_parse (not part of pipeline)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fractions import Fraction
from abc_parse import parse_abc, ABCError

GOOD = """X:1
T:Test Tune
M:4/4
L:1/8
K:D
D2 F2 A2 d2 | c2 A2 F2 D2 | E2 G2 B2 e2 | d6 z2 |
^G2 =G2 F2 E2 | D8 | {a}f2 e2 d2 c2 | B2 A2 G2 F2 | E8 |]
"""

def main():
    notes, segs, fm, meta = parse_abc(GOOD)
    print("OK:", len(notes), "events; meta:", meta, "; segs:", segs)
    print("first 6:", [(str(n['onset']), n['pitch'], n['dur_q']) for n in notes[:6]])
    assert notes[0]['pitch'] == 62, "D4 = 62"
    assert notes[1]['pitch'] == 66, "F#4 = 66 (K:D signature)"
    assert notes[4]['pitch'] == 73, "C#5 = 73 (K:D signature)"
    assert segs == [(Fraction(0), 4)]
    # grace note {a} absorbed: bar 7 starts at quarter 24 with f2 -> F#5=78
    b7 = [n for n in notes if Fraction(24) <= n['onset'] < Fraction(28)]
    print("bar7 pitches:", [n['pitch'] for n in b7])  # F#5 E5 D5 C#5
    assert b7[0]['pitch'] == 78 and b7[-1]['pitch'] == 73
    # bar 2: c2 A2 F2 D2 -> C#5 A4 F#4 D4
    b2 = [n for n in notes if Fraction(4) <= n['onset'] < Fraction(8)]
    assert [n['pitch'] for n in b2] == [73, 69, 66, 62]
    # accidental bar-scope: ^G then =G -> G#4 G4 F#4 E4
    b5 = [n for n in notes if Fraction(16) <= n['onset'] < Fraction(20)]
    print("bar5 pitches:", [n['pitch'] for n in b5])
    assert [n['pitch'] for n in b5] == [68, 67, 66, 64]

    for bad_name, bad in [
        ("no key", "X:1\nM:4/4\nL:1/8\nCDEF GABc|\n"),
        ("chord", "X:1\nM:4/4\nL:1/8\nK:C\n[CEG]2 [DFA]2 |\n"),
        ("garbage", "X:1\nM:4/4\nL:1/8\nK:C\nC D E <<<>>> F G A B |\n"),
        ("short", "X:1\nM:4/4\nL:1/8\nK:C\nC D E |\n"),
        ("freemeter", "X:1\nM:none\nL:1/8\nK:C\nCDEF GABc defg abc'd'|\n"),
    ]:
        try:
            parse_abc(bad)
            print("!! SHOULD HAVE FAILED:", bad_name)
        except ABCError as e:
            print(f"rejected {bad_name}: {e}")
    print("self-test done")

if __name__ == '__main__':
    main()
