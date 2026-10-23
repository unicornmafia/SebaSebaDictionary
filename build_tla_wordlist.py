"""Build a word list from the TLA sentence corpus.

Usage: python build_tla_wordlist.py SENTENCES_EN.jsonl OUT.jsonl

SENTENCES_EN.jsonl is train.jsonl with an added English "translation" and the
original German moved to "translation_de" (output of translate_tla.py).

One output line per TLA lemma. Hieroglyph, transliteration, lemma, UPOS and
glossing columns are all space-separated with one item per word, so word i of
a sentence is item i of each column.
"""
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict

TLA_LEMMA_URL = "https://thesaurus-linguae-aegyptiae.de/lemma/{}"
MAX_EXAMPLES = 3
NAME_RE = re.compile(r"^EGYPTIAN HIEROGLYPH ([A-Z]{1,2})0*(\d+)([A-Z]*)$")
TAG_RE = re.compile(r"<g>(.*?)</g>")
CODE_TEXT_RE = re.compile(r"^(?:Aa|[A-Z])\d+[A-Za-z]*$")


def gardiner_code(ch: str) -> str | None:
    """'𓀀' -> 'A1', '𓌅' -> 'U1a'... Returns None for non-signs (joiners, variation selectors)."""
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return None
    m = NAME_RE.match(name)
    if not m:
        return None
    series, num, suffix = m.groups()
    if series == "AA":
        series = "Aa"
    return f"{series}{num}{suffix.lower()}"


def to_gardiner(group: str) -> str:
    """Convert one word's hieroglyphs to space-separated Gardiner codes.
    <g>CODE</g> tags (signs outside Unicode) keep their TLA code; unidentified signs become '?'."""
    codes = []
    pos = 0
    for m in TAG_RE.finditer(group):
        codes += [c for c in map(gardiner_code, group[pos:m.start()]) if c]
        tag = m.group(1)
        codes.append("?" if tag == "(unidentified)" else tag)
        pos = m.end()
    codes += [c for c in map(gardiner_code, group[pos:]) if c]
    # A few words in the source carry a bare Gardiner code instead of signs
    if not codes and CODE_TEXT_RE.match(group):
        codes.append(group)
    return " ".join(codes)


def main():
    src, dst = sys.argv[1:3]
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]

    words = defaultdict(lambda: {
        "count": 0, "lemma_forms": Counter(), "upos": Counter(), "glossing": Counter(),
        "spellings": Counter(), "transliterations": Counter(), "examples": [],
        "single_word_translations": [], "dates": [],
    })
    skipped_rows = 0
    for r in rows:
        hiero = r["hieroglyphs"].split()
        translit = r["transliteration"].split()
        lemmas = r["lemmatization"].split()
        upos = r["UPOS"].split()
        gloss = r["glossing"].split()
        if not (len(hiero) == len(translit) == len(lemmas) == len(upos) == len(gloss)):
            skipped_rows += 1
            continue
        example = {
            "hieroglyphs": r["hieroglyphs"],
            "transliteration": r["transliteration"],
            "translation": r["translation"],
            "translation_de": r["translation_de"],
            "dateNotBefore": int(r["dateNotBefore"]) if r.get("dateNotBefore") else None,
            "dateNotAfter": int(r["dateNotAfter"]) if r.get("dateNotAfter") else None,
        }
        for h, t, l, u, g in zip(hiero, translit, lemmas, upos, gloss):
            lemma_id, _, lemma_form = l.partition("|")
            # A few names are lemmatized as just their suffix pronoun (sꜣi̯-ꞽb=f -> =f)
            if lemma_form.startswith("=") and not t.startswith("="):
                continue
            w = words[lemma_id]
            w["count"] += 1
            w["lemma_forms"][lemma_form] += 1
            w["upos"][u] += 1
            w["glossing"][g] += 1
            w["spellings"][h] += 1
            w["transliterations"][t] += 1
            if example["dateNotBefore"] is not None:
                w["dates"].append((example["dateNotBefore"], example["dateNotAfter"]))
            if len(lemmas) == 1:
                w["single_word_translations"].append((r["translation"], r["translation_de"]))
            if example not in w["examples"]:
                w["examples"].append(example)

    out = []
    for lemma_id, w in words.items():
        # Prefer short examples: they show the word with the least surrounding noise
        examples = sorted(w["examples"], key=lambda e: len(e["transliteration"]))[:MAX_EXAMPLES]
        meanings = Counter(w["single_word_translations"]).most_common(5)
        out.append({
            "lemma_id": lemma_id,
            "lemma": w["lemma_forms"].most_common(1)[0][0],
            "upos": w["upos"].most_common(1)[0][0],
            "glossing": [g for g, _ in w["glossing"].most_common()],
            "count": w["count"],
            "spellings": [
                {"hieroglyphs": h, "gardiner": to_gardiner(h), "count": n}
                for h, n in w["spellings"].most_common()
            ],
            "transliterations": [t for t, _ in w["transliterations"].most_common()],
            "translations": [en for (en, de), _ in meanings],
            "translations_de": [de for (en, de), _ in meanings],
            "dateNotBefore": min((d[0] for d in w["dates"]), default=None),
            "dateNotAfter": max((d[1] for d in w["dates"] if d[1] is not None), default=None),
            "tla_url": TLA_LEMMA_URL.format(lemma_id),
            "examples": examples,
        })
    out.sort(key=lambda e: (-e["count"], e["lemma"]))

    with open(dst, "w", encoding="utf-8") as f:
        for e in out:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"{len(out)} lemmas from {len(rows)} sentences ({skipped_rows} skipped: column lengths differ)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
