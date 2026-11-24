"""Add each lemma's dictionary meaning to the TLA word list.

Usage: python add_tla_meanings.py DICTIONARY.xml WORDLIST.jsonl

The TLA sentence corpus only says which lemma each word is, not what the lemma
means. The meanings come from the TLA lemma list published as the AED
(Ancient Egyptian Dictionary) word list, keyed by the same lemma ids:

    https://github.com/simondschweitzer/aed-tei  files/dictionary.xml
    Simon D. Schweitzer, Corpus of Egyptian Texts for the AED, CC BY-SA 4.0,
    https://doi.org/10.5281/zenodo.3580939

Adds "meaning" (English), "meaning_de" and "meaning_source" to each lemma in
WORDLIST.jsonl, in place. meaning_source is "aed" for the AED's English,
"aed-mt" for the AED's German machine-translated to English, or null.
Machine translations are cached in tla_meanings_mt_cache.json next to the
word list, so re-running doesn't translate again.
"""
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

TEI = "{http://www.tei-c.org/ns/1.0}"
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"
BATCH = 32


def load_dictionary(path):
    """lemma id -> (english, german); either may be ''."""
    out = {}
    for entry in ET.parse(path).getroot().iter(TEI + "entry"):
        lemma_id = entry.get(XML_ID, "").removeprefix("tla")
        quotes = {"en": [], "de": []}
        for cit in entry.iter(TEI + "cit"):
            lang = cit.get(XML_LANG)
            if lang in quotes:
                quotes[lang] += [q.text.strip() for q in cit.iter(TEI + "quote") if q.text and q.text.strip()]
        out[lemma_id] = ("; ".join(quotes["en"]), "; ".join(quotes["de"]))
    return out


def split_senses(text):
    """'geben; veranlassen' -> ['geben', 'veranlassen'], ignoring ';' inside brackets.
    The model drops parts of a ';' list, so each sense is translated alone."""
    parts, depth, cur = [], 0, ""
    for ch in text:
        depth += ch in "([" and 1 or (ch in ")]" and -1 or 0)
        if ch == ";" and depth <= 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    parts.append(cur.strip())
    return [p for p in parts if p]


# For a name only the explanation in brackets is German: "Wadju (ein Weideplatz)".
# The name itself must not go through the model, which turns "Bu-nefer" into
# "Citrus fruit".
NAME_NOTE = re.compile(r"\(([^()]*[a-zäöüß][^()]*)\)")


def looks_like_name(upos, de):
    """A bare name ("Bu-nefer", "Wadju (ein Weideplatz)"), not a German
    description such as "Die große Neunheit" or "[Goldname Dewens]"."""
    if upos != "PROPN" or "[" in de:
        return False
    words = NAME_NOTE.sub("", de).split()
    return bool(words) and all(w[0].isupper() and w not in {"Die", "Der", "Das"} for w in words)


def to_translate(de, is_name):
    if is_name:
        return [m.group(1) for m in NAME_NOTE.finditer(de)]
    return split_senses(de)


def translate_meaning(de, is_name, cache):
    if is_name:
        return NAME_NOTE.sub(lambda m: "(" + cache.get(m.group(1), m.group(1)) + ")", de)
    return "; ".join(cache.get(p, "") for p in split_senses(de))


def machine_translate(meanings, cache):
    lines = [p for de, is_name in meanings for p in to_translate(de, is_name)]
    todo = sorted({l for l in lines if l not in cache}, key=len)
    if not todo:
        return
    import torch
    from transformers import MarianMTModel, MarianTokenizer
    from translate_tla import MODEL, fix_article, protect, restore

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = MarianTokenizer.from_pretrained(MODEL)
    model = MarianMTModel.from_pretrained(MODEL).to(device).eval()
    print(f"Machine-translating {len(todo)} German meanings", file=sys.stderr)
    for start in range(0, len(todo), BATCH):
        batch = todo[start:start + BATCH]
        prot = [protect(l) for l in batch]
        enc = tok([p[0] for p in prot], return_tensors="pt", padding=True, truncation=True,
                  max_length=256).to(device)
        with torch.no_grad():
            out = model.generate(**enc, num_beams=4, max_new_tokens=256)
        for line, (_, spans), en in zip(batch, prot, tok.batch_decode(out, skip_special_tokens=True)):
            en = fix_article(restore(en, spans)).strip().rstrip(";.")
            # Dictionary meanings are lower case unless the German is a name
            if en and line[0].islower() and en[0].isupper() and not en.startswith("I "):
                en = en[0].lower() + en[1:]
            cache[line] = en


def main():
    dict_path, wordlist_path = sys.argv[1:3]
    cache_path = os.path.join(os.path.dirname(os.path.abspath(wordlist_path)), "tla_meanings_mt_cache.json")

    meanings = load_dictionary(dict_path)
    words = [json.loads(l) for l in open(wordlist_path, encoding="utf-8")]
    try:
        cache = json.load(open(cache_path, encoding="utf-8"))
    except (OSError, ValueError):
        cache = {}

    machine_translate([(meanings[w["lemma_id"]][1], looks_like_name(w["upos"], meanings[w["lemma_id"]][1])) for w in words
                       if w["lemma_id"] in meanings and not meanings[w["lemma_id"]][0]
                       and meanings[w["lemma_id"]][1]], cache)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=0, sort_keys=True)

    counts = {"aed": 0, "aed-mt": 0, None: 0}
    for w in words:
        en, de = meanings.get(w["lemma_id"], ("", ""))
        if en:
            source = "aed"
        elif de and translate_meaning(de, looks_like_name(w["upos"], de), cache):
            en, source = translate_meaning(de, looks_like_name(w["upos"], de), cache), "aed-mt"
        else:
            source = None
        w["meaning"], w["meaning_de"], w["meaning_source"] = en or None, de or None, source
        counts[source] += 1

    with open(wordlist_path, "w", encoding="utf-8") as f:
        for w in words:
            f.write(json.dumps(w, ensure_ascii=False) + "\n")
    print(f"{len(words)} lemmas: {counts['aed']} English meanings, {counts['aed-mt']} machine-translated, "
          f"{counts[None]} without a meaning", file=sys.stderr)


if __name__ == "__main__":
    main()
