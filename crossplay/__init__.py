"""Cross-play study — see crossplay/README.md.

The stdout reconfigure below is not cosmetic. On a Korean Windows install the
console defaults to cp949, and every table in this package prints a '±' or an
em dash, so the analysis scripts die on the first line with a
UnicodeEncodeError. (The same locale makes `tests/test_smoke.py` fail on
`configs/team.json` unless it is run with PYTHONUTF8=1 — that one is in shared
code and is left alone here.)
"""

import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # already wrapped, or not a TextIO
        pass
