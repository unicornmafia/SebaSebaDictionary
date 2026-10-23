#!/usr/bin/env python3
"""
Import the TLA word list (DictionaryName=13) into MongoDB.

dictionaries/tla_wordlist.jsonl is built from the Thesaurus Linguae
Aegyptiae "Original Earlier Egyptian sentences, corpus v18, premium" dataset
by translate_tla.py and build_tla_wordlist.py.

Each attested spelling of a TLA lemma becomes one (Transliteration,
GardinerSigns) pair, merged into an existing entry with the same key or
created as a new entry. The TLA has no per-word meanings in this dataset, so
the translation text is the lemma's single-word sentence translations where
there are any; every TLA translation also carries the shortest example
sentence and a link to the lemma on the TLA website.

Re-running is safe: all TLA data is removed first, then imported again. The
KeywordSearch index is rebuilt afterwards.

Usage:
    python3 import_tla.py [--dry-run]
"""

import argparse
import json
import os
import sys

from pymongo import DeleteOne, InsertOne, MongoClient, ReplaceOne

from parse_dictionaries import TLA, _normalise_gard, build_keyword_index

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB  = os.environ.get("MONGO_DB", "MiddleEgyptianDictionary")

WORDLIST_PATH = os.path.join(os.path.dirname(__file__), "dictionaries", "tla_wordlist.jsonl")

UPOS_NAMES = {
    "NOUN": "noun", "PROPN": "proper noun", "VERB": "verb", "ADJ": "adjective",
    "ADV": "adverb", "ADP": "preposition", "PRON": "pronoun", "PART": "particle",
    "NUM": "numeral", "INTJ": "interjection",
}

# Leiden Unified Transliteration -> the Manuel de Codage letters used in the DB
_TRANSLIT_MAP = {
    "ꜣ": "A", "Ꜣ": "A", "ꜥ": "a", "Ꜥ": "a", "ꞽ": "i", "Ꞽ": "i", "j": "i", "J": "i",
    "ḥ": "H", "Ḥ": "H", "ḫ": "x", "Ḫ": "x", "ẖ": "X", "š": "S", "Š": "S",
    "ṯ": "T", "Ṯ": "T", "ḏ": "D", "Ḏ": "D", "ḳ": "q", "=": ".",
    "ʾ": "", "̯": "",   # aleph-like hamza, inverted breve below (w̯)
}


def to_mdc_translit(lui: str) -> str:
    out = []
    for ch in lui:
        if ch == "̱":   # h + macron below = ẖ
            if out and out[-1] == "h":
                out[-1] = "X"
            continue
        if ch in _TRANSLIT_MAP:
            out.append(_TRANSLIT_MAP[ch])
        elif ch.isascii() and ch.isalpha():
            out.append(ch.lower())
        else:
            out.append(ch)
    return "".join(out)


def _key(translit: str, signs: str) -> tuple:
    return (translit.strip(), _normalise_gard(signs))


def _strip_tla(entry: dict) -> bool:
    """Remove TLA translations in place. Returns True if anything changed."""
    kept = []
    for t in entry.get("Translations", []):
        meta = t.get("TranslationMetadata", [])
        other = [m for m in meta if m.get("DictionaryName") != TLA]
        if other:
            t["TranslationMetadata"] = other
            kept.append(t)
    changed = len(kept) != len(entry.get("Translations", [])) or any(
        m.get("DictionaryName") == TLA for t in kept for m in t["TranslationMetadata"])
    entry["Translations"] = kept
    return changed


def tla_translation(word: dict) -> dict:
    example = word["examples"][0] if word["examples"] else None
    return {
        "translation": "; ".join(word["translations"]),
        "Example": {
            "transliteration": example["transliteration"],
            # Drop quotes around direct speech; the page adds its own
            "translation": example["translation"].strip().strip('"„“”'),
        } if example else None,
        "TranslationMetadata": [{
            "DictionaryName": TLA,
            "PartOfSpeech": UPOS_NAMES.get(word["upos"], word["upos"].lower()),
            "Page": None,
            "IndexOnPage": None,
            "Url": word["tla_url"],
        }],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without modifying the database")
    args = parser.parse_args()

    words = [json.loads(l) for l in open(WORDLIST_PATH, encoding="utf-8")]
    print(f"{len(words)} TLA lemmas", file=sys.stderr)

    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client[MONGO_DB]
    entries_col = db["DictionaryEntry"]
    keywords_col = db["KeywordSearch"]

    entries = list(entries_col.find())
    print(f"Loaded {len(entries)} entries", file=sys.stderr)

    by_key = {}
    changed_ids = set()
    for e in entries:
        by_key.setdefault(_key(e["Transliteration"], e.get("GardinerSigns") or ""), e)
        if _strip_tla(e):
            changed_ids.add(e["_id"])

    new_entries = []
    merged = skipped = 0
    for w in words:
        translit = to_mdc_translit(w["lemma"])
        for spelling in w["spellings"]:
            signs = spelling["gardiner"]
            # Spellings with a sign the TLA marks as unidentified can't be shown or searched
            if not signs or "?" in signs.split():
                skipped += 1
                continue
            k = _key(translit, signs)
            entry = by_key.get(k)
            if entry is None:
                entry = {
                    "Transliteration": translit,
                    "GardinerSigns":   _normalise_gard(signs),
                    "Res":             None,
                    "ManuelDeCodage":  None,
                    "Translations":    [],
                }
                by_key[k] = entry
                new_entries.append(entry)
            elif "_id" in entry:
                changed_ids.add(entry["_id"])
                merged += 1
            urls = {m.get("Url") for t in entry["Translations"] for m in t["TranslationMetadata"]}
            if w["tla_url"] not in urls:
                entry["Translations"].append(tla_translation(w))

    empty = [e for e in entries if not e["Translations"]]
    empty_ids = {e["_id"] for e in empty}
    updated = [e for e in entries if e["_id"] in changed_ids and e["_id"] not in empty_ids]

    print(f"  Existing entries given TLA data: {merged}", file=sys.stderr)
    print(f"  Entries updated:                 {len(updated)}", file=sys.stderr)
    print(f"  Entries added:                   {len(new_entries)}", file=sys.stderr)
    print(f"  Entries removed:                 {len(empty)} (TLA-only, from an earlier import)",
          file=sys.stderr)
    print(f"  Spellings skipped:               {skipped} (unidentified signs)", file=sys.stderr)
    for e in new_entries[:5]:
        print(f"    + {e['Transliteration']}  {e['GardinerSigns']}  "
              f"{e['Translations'][0]['translation'][:40]!r}", file=sys.stderr)

    if args.dry_run:
        print("Dry run — database not modified.", file=sys.stderr)
        return

    ops = [ReplaceOne({"_id": e["_id"]}, e) for e in updated]
    ops += [DeleteOne({"_id": e["_id"]}) for e in empty]
    ops += [InsertOne(e) for e in new_entries]   # pymongo assigns _id in place
    if ops:
        entries_col.bulk_write(ops, ordered=False)

    print("Rebuilding keyword index…", file=sys.stderr)
    final = list(entries_col.find({}, {"Translations": 1}))
    kw_index = build_keyword_index(final)
    keywords_col.drop()
    keywords_col.insert_many([{"Keyword": kw, "EntryIds": ids} for kw, ids in kw_index.items()])
    keywords_col.create_index("Keyword")

    print(f"Done. {len(final)} entries, {len(kw_index)} keywords.", file=sys.stderr)
    client.close()


if __name__ == "__main__":
    main()
