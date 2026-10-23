"""Translate the German sentence translations in train.jsonl to English.

Usage: python translate_tla.py IN.jsonl OUT.jsonl CACHE.json [--sample N]

- Inline transliterations ($...$) are swapped for placeholders before
  translation and restored afterwards.
- Multi-line translations are translated line by line, keeping line breaks.
- Identical German lines are translated once; results are cached in CACHE.json
  so an interrupted run can resume.
"""
import json
import re
import sys

import torch
from transformers import MarianMTModel, MarianTokenizer

MODEL = "Helsinki-NLP/opus-mt-de-en"
BATCH = 32
DOLLAR = re.compile(r"\$([^$]*)\$")
PLACEHOLDER = re.compile(r"X(\d+)Q")


HIEROGLYPHS = re.compile(r"[\U00013000-\U0001345F︀-️]+")

# Egyptological terms the general-purpose model gets wrong, with the usual
# English rendering. Matched as whole words, including common inflections.
GLOSSARY = [
    (r"Horusauges?", "Eye of Horus"),
    (r"Totenpriesterinnen", "funerary priestesses"),
    (r"Totenpriesterin", "funerary priestess"),
    (r"Totenpriestern?s?", "funerary priest"),
    (r"Totenopfern?s?", "funerary offering"),
    (r"Totenstiftung(?:en)?", "funerary endowment"),
    (r"Vorlesepriesters?", "lector priest"),
    (r"Gottessieglers?", "god's sealer"),
    (r"Aktenschreibers?", "document scribe"),
    (r"Langhornrind(?:es|er|ern)?", "long-horned cattle"),
    (r"Säbelantilopen?", "oryx"),
    (r"Sedfest(?:es)?", "Sed festival"),
    (r"Neunheiten", "Enneads"),
    (r"Neunheit", "Ennead"),
    (r"Balsamierers?", "embalmer"),
    (r"Personenbeischrift", "caption"),
    (r"Rangtitels?", "rank title"),
    (r"Gutsvorstehers?", "estate overseer"),
    (r"Gutsverwalters?", "estate manager"),
    (r"Kammerdieners?", "chamberlain"),
    (r"Vorsteherin", "overseer"),
    (r"Vorstehern?s?", "overseer"),
    (r"Versorgte[nr]?", "revered one"),
    (r"Gotteshalle", "god's hall"),
    (r"Jungtier(?:e|es)?", "young animal"),
    (r"Einzige[nrs]? Freund(?:es)?", "sole companion"),
    (r"Worte sprechen", "Words to be spoken"),
    (r"Wab-Priester(?:s|n)?", "wab-priest"),
    (r"Großer? der Räucherung", "great one of censing"),
]
GLOSSARY_RE = [(re.compile(r"\b" + pat + r"\b"), en) for pat, en in GLOSSARY]
GLOSSARY_RE.append((re.compile(r"\bwörtl\."), "lit."))

# "Ein Vorlesepriester" comes out as "An lector priest": the model picks the
# article before it sees the glossary term. Fix the article to suit the term.
_TERMS = sorted({en for _, en in GLOSSARY}, key=len, reverse=True)
_A_BEFORE_TERM = re.compile(r"\b([Aa])n? (?=(" + "|".join(map(re.escape, _TERMS)) + r")\b)")


def fix_article(text):
    def sub(m):
        return m.group(1) + ("n " if m.group(2)[0].lower() in "aeiou" else " ")
    return _A_BEFORE_TERM.sub(sub, text)


def protect(line):
    """Replace text the model must not translate (or translates badly) with
    placeholders. spans[i] is what placeholder i is restored to."""
    spans = []
    def keep(text):
        spans.append(text)
        return f"X{len(spans) - 1}Q"
    line = DOLLAR.sub(lambda m: keep(m.group(1)), line)
    line = HIEROGLYPHS.sub(lambda m: keep(m.group(0)), line)
    for pat, en in GLOSSARY_RE:
        line = pat.sub(lambda m, en=en: keep(en), line)
    return line, spans


def restore(text, spans):
    def sub(m):
        i = int(m.group(1))
        return spans[i] if i < len(spans) else m.group(0)
    out = PLACEHOLDER.sub(sub, text)
    # Any span the model dropped is appended in brackets rather than lost
    missing = [s for i, s in enumerate(spans) if f"X{i}Q" not in text]
    if missing:
        out += " [" + ", ".join(missing) + "]"
    return out


def main():
    src, dst, cache_path = sys.argv[1:4]
    sample = int(sys.argv[sys.argv.index("--sample") + 1]) if "--sample" in sys.argv else None

    rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    if sample:
        rows = rows[:sample]

    try:
        cache = json.load(open(cache_path, encoding="utf-8"))
    except (OSError, ValueError):
        cache = {}

    lines = set()
    for r in rows:
        for line in r["translation"].replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            if line.strip():
                lines.add(line.strip())
    todo = sorted(l for l in lines if l not in cache)
    print(f"{len(rows)} rows, {len(lines)} distinct lines, {len(todo)} to translate", file=sys.stderr)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = MarianTokenizer.from_pretrained(MODEL)
    model = MarianMTModel.from_pretrained(MODEL).to(device).eval()

    todo.sort(key=len)  # similar lengths per batch = less padding
    for start in range(0, len(todo), BATCH):
        batch = todo[start:start + BATCH]
        prot = [protect(l) for l in batch]
        enc = tok([p[0] for p in prot], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model.generate(**enc, num_beams=4, max_new_tokens=512)
        for line, (_, spans), en in zip(batch, prot, tok.batch_decode(out, skip_special_tokens=True)):
            en = fix_article(restore(en, spans))
            if en and line[0].isupper() and en[0].islower():
                en = en[0].upper() + en[1:]
            cache[line] = en
        if (start // BATCH) % 20 == 0:
            json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False)
            print(f"  {min(start + BATCH, len(todo))}/{len(todo)}", file=sys.stderr, flush=True)
    json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False)

    with open(dst, "w", encoding="utf-8") as f:
        for r in rows:
            parts = r["translation"].replace("\r\n", "\n").replace("\r", "\n").split("\n")
            en = "\n".join(fix_article(cache.get(p.strip(), "")) if p.strip() else "" for p in parts)
            out = dict(r)
            out["translation_de"] = out.pop("translation")
            out["translation"] = en
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
