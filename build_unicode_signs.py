"""Build static/unicode_signs.json: Gardiner/JSesh sign code -> Unicode hieroglyph.

Usage: python build_unicode_signs.py Unikemet.txt FONT.ttf

Unikemet.txt is from the Unicode Character Database
(https://www.unicode.org/Public/UCD/latest/ucd/Unikemet.txt). Its kEH_JSesh and
kEH_HG fields give the sign codes for the basic list and for the extended list
added in Unicode 16 (e.g. U+13948 = D115). Only characters the font can draw are
kept. Keys are upper case ("D115", "AA1", "N35A").
"""
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict

from fontTools.ttLib import TTFont

OUT = os.path.join(os.path.dirname(__file__), "static", "unicode_signs.json")
FIELDS = ("kEH_JSesh", "kEH_HG")   # in order of preference
NAME_RE = re.compile(r"^EGYPTIAN HIEROGLYPH ([A-Z]+)0*(\d+)([A-Z]*)$")


def main():
    unikemet, font = sys.argv[1:3]
    cmap = TTFont(font).getBestCmap()
    codes = defaultdict(dict)   # codepoint -> {field: [codes]}
    for line in open(unikemet, encoding="utf-8"):
        if not line.startswith("U+"):
            continue
        cp, field, value = line.rstrip("\n").split("\t")
        if field in FIELDS:
            codes[int(cp[2:], 16)][field] = value.split()

    signs = {}
    for field in FIELDS:
        # Lower code points first, so a code shared by several characters
        # maps to the basic-list sign
        for cp in sorted(codes):
            if cp not in cmap:
                continue
            for code in codes[cp].get(field, []):
                signs.setdefault(code.upper(), chr(cp))

    # Basic-list signs are also named by their Gardiner code ("... A001")
    for cp in sorted(cmap):
        m = NAME_RE.match(unicodedata.name(chr(cp), ""))
        if m:
            signs.setdefault("".join(m.groups()), chr(cp))

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(dict(sorted(signs.items())), f, ensure_ascii=False, indent=0)
    print(f"{len(signs)} sign codes -> {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
