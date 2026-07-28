#!/usr/bin/env python3
"""
Aggregate keywords by author from articles_keywords.csv.
Handles author formats:
  - 中文名=English Name   (most common)
  - 中文名 only
  - English Name only
Output: author_keywords.csv with separate Chinese and English keyword columns.
"""

import csv
from collections import defaultdict


INPUT_FILE = "articles_keywords.csv"
OUTPUT_FILE = "author_keywords.csv"
CHINESE_KEYWORD_COL = "關鍵字"
ENGLISH_KEYWORD_COL = "英文關鍵字"


def is_chinese_char(c: str) -> bool:
    return '一' <= c <= '鿿'


def has_chinese(s: str) -> bool:
    return any(is_chinese_char(c) for c in s)


def parse_author(raw: str) -> tuple[str, str, str]:
    """
    Parse one author token.
    Returns (key, chinese_name, english_name).
    key is used for deduplication — prefer Chinese name.
    """
    raw = raw.strip()
    if not raw:
        return ("", "", "")

    if "=" in raw:
        left, _, right = raw.partition("=")
        left, right = left.strip(), right.strip()
        # Determine which side is Chinese
        if has_chinese(left):
            zh, en = left, right
        elif has_chinese(right):
            zh, en = right, left
        else:
            zh, en = "", left  # both English, keep as English-only
        return (zh or en, zh, en)

    if has_chinese(raw):
        return (raw, raw, "")

    # Pure English
    return (raw, "", raw)


def parse_authors(author_field: str) -> list[tuple[str, str, str]]:
    """Split multi-author field and parse each."""
    results = []
    for token in author_field.split("|"):
        key, zh, en = parse_author(token)
        if key:
            results.append((key, zh, en))
    return results


def parse_keywords(kw_field: str) -> list[str]:
    """Split keyword field and return non-empty, stripped terms."""
    return [k.strip() for k in kw_field.split("|") if k.strip()]


def split_keywords(row: dict) -> tuple[list[str], list[str]]:
    """Return (Chinese/CJK keywords, English/non-CJK keywords)."""
    zh_keywords = []
    en_keywords = []

    for keyword in parse_keywords(row.get(CHINESE_KEYWORD_COL, "")):
        if has_chinese(keyword):
            zh_keywords.append(keyword)
        else:
            en_keywords.append(keyword)

    for keyword in parse_keywords(row.get(ENGLISH_KEYWORD_COL, "")):
        if has_chinese(keyword):
            zh_keywords.append(keyword)
        else:
            en_keywords.append(keyword)

    return zh_keywords, en_keywords


def main():
    import os
    src = INPUT_FILE
    print(f"Reading {src}...", flush=True)

    # author_key -> {zh, en_set, article_count, zh_keywords, en_keywords}
    authors: dict[str, dict] = defaultdict(lambda: {
        "zh": "",
        "en_set": set(),
        "article_count": 0,
        "zh_keywords": set(),
        "en_keywords": set(),
    })

    with open(src, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            author_field = row.get("作者", "").strip()
            if not author_field:
                continue

            zh_keywords, en_keywords = split_keywords(row)
            parsed = parse_authors(author_field)

            for key, zh, en in parsed:
                rec = authors[key]
                if zh and not rec["zh"]:
                    rec["zh"] = zh
                if en:
                    rec["en_set"].add(en)
                rec["article_count"] += 1
                rec["zh_keywords"].update(zh_keywords)
                rec["en_keywords"].update(en_keywords)

    print(f"Unique authors: {len(authors)}", flush=True)

    # Write output
    with open(OUTPUT_FILE, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["作者", "英文姓名", "文章數", "關鍵字數", "關鍵字", "英文關鍵字"],
        )
        writer.writeheader()

        for key, rec in sorted(authors.items()):
            display_name = rec["zh"] or key
            en_name = " / ".join(sorted(rec["en_set"])) if rec["en_set"] else ""
            zh_kw_list = sorted(rec["zh_keywords"])
            en_kw_list = sorted(rec["en_keywords"])
            keyword_count = len(zh_kw_list) + len(en_kw_list)
            writer.writerow({
                "作者": display_name,
                "英文姓名": en_name,
                "文章數": rec["article_count"],
                "關鍵字數": keyword_count,
                "關鍵字": "|".join(zh_kw_list),
                "英文關鍵字": "|".join(en_kw_list),
            })

    print(f"Done. Output: {OUTPUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
