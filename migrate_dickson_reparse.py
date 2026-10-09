#!/usr/bin/env python3
"""
Replace Dickson (DictionaryName=0) data with a fresh parse of the Dickson PDF.

The original Dickson parser read whole pages, which interleaved the PDF's two
columns line by line. It only kept entries that fit on one extracted line, and
glued neighbouring entries together, producing sign lists like
"S29 D58 [SBH] (N.) CRY {S29 D58 V28 F18 A2". parse_dictionaries.parse_dickson
now reads each column separately and handles wrapped entries.

This script:
  1. removes every Dickson translation/metadata from the database,
  2. merges the re-parsed Dickson entries back in by (Transliteration,
     GardinerSigns), creating entries where none exist,
  3. deletes entries left with no translations (e.g. the garbled ones),
  4. rebuilds the KeywordSearch index from all entries.

Entries that survive keep their _id. Other sources are untouched.

Usage:
    python3 migrate_dickson_reparse.py [--dry-run]
"""

import argparse
import os
import sys

from pymongo import DeleteOne, InsertOne, MongoClient, ReplaceOne

from parse_dictionaries import (
    DICKSON,
    PDFS_DIR,
    _normalise_gard,
    _normalise_translit,
    build_keyword_index,
    parse_dickson,
)

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB  = os.environ.get("MONGO_DB", "MiddleEgyptianDictionary")


def _key(translit: str, signs: str) -> tuple:
    return (_normalise_translit(translit), _normalise_gard(signs))


def _strip_dickson(entry: dict) -> bool:
    """Remove Dickson metadata in place. Returns True if anything changed."""
    changed = False
    kept = []
    for t in entry.get("Translations", []):
        meta = t.get("TranslationMetadata", [])
        other = [m for m in meta if m.get("DictionaryName") != DICKSON]
        if len(other) != len(meta):
            changed = True
        if other:
            t["TranslationMetadata"] = other
            kept.append(t)
    entry["Translations"] = kept
    return changed


def _add_dickson(entry: dict, translation: str) -> None:
    meta = {"DictionaryName": DICKSON, "PartOfSpeech": None, "Page": None, "IndexOnPage": None}
    for t in entry["Translations"]:
        if t["translation"] == translation:
            if all(m["DictionaryName"] != DICKSON for m in t["TranslationMetadata"]):
                t["TranslationMetadata"].append(meta)
            return
    entry["Translations"].append({"translation": translation, "TranslationMetadata": [meta]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without modifying the database")
    args = parser.parse_args()

    print("Parsing Dickson…", file=sys.stderr)
    dickson = parse_dickson(os.path.join(PDFS_DIR, "dickson_2006.pdf"))

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
        if _strip_dickson(e):
            changed_ids.add(e["_id"])

    new_entries = []
    for d in dickson:
        k = _key(d["transliteration"], d["gardiner_signs"])
        entry = by_key.get(k)
        if entry is None:
            entry = {
                "Transliteration": d["transliteration"],
                "GardinerSigns":   _normalise_gard(d["gardiner_signs"]),
                "Res":             None,
                "ManuelDeCodage":  None,
                "Translations":    [],
            }
            by_key[k] = entry
            new_entries.append(entry)
        elif "_id" in entry:
            changed_ids.add(entry["_id"])
        _add_dickson(entry, d["translation"])

    empty = [e for e in entries if not e["Translations"]]
    empty_ids = {e["_id"] for e in empty}
    updated = [e for e in entries if e["_id"] in changed_ids and e["_id"] not in empty_ids]

    print(f"  Entries updated:  {len(updated)}", file=sys.stderr)
    print(f"  Entries added:    {len(new_entries)}", file=sys.stderr)
    print(f"  Entries removed:  {len(empty)} (no translations left once old Dickson data is gone)",
          file=sys.stderr)
    for e in empty[:10]:
        print(f"    - {e['Transliteration']}  {e.get('GardinerSigns')}", file=sys.stderr)

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
