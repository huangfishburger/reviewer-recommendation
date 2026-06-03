#!/usr/bin/env python3
"""
Randomly pick one eligible F03 source paper and recommend authors by keywords.

Criteria:
1. source_paper.csv row has 來源文獻ID starting with F03
2. articles_updated.csv has the same 主要篇名
3. articles_updated.csv 主要摘要 and 關鍵字 are both non-empty
4. Embed the sampled paper keywords and compare them with author keywords
"""

import argparse
from collections import defaultdict
import csv
import json
import random
from pathlib import Path


SOURCE_FILE = "source_paper.csv"
ARTICLES_FILE = "articles_updated.csv"
AUTHOR_KEYWORDS_FILE = "author_keywords.csv"
OUTPUT_FILE = "sample_author_recommendations.json"
EMBEDDING_CACHE_FILE = "author_keyword_embeddings_cache.json"
EMBEDDING_MODEL = "all-mpnet-base-v2"
KEYWORD_TOP_K = 30
SIMILARITY_THRESHOLD = 0.55
COOCCUR_BONUS = 0.2


def normalize_title(title: str) -> str:
    return " ".join((title or "").strip().split())


def normalize_name(name: str) -> str:
    return " ".join((name or "").strip().lower().split())


def parse_pipe_field(raw: str) -> list[str]:
    return [part.strip() for part in (raw or "").split("|") if part.strip()]


def parse_author_identities(raw: str) -> set[str]:
    """
    Parse author names for exclusion matching.
    Handles fields like 中文名=English Name and pipe-separated authors.
    """
    identities = set()
    for token in parse_pipe_field(raw):
        pieces = [token]
        if "=" in token:
            left, _, right = token.partition("=")
            pieces.extend([left, right])
        if "/" in token:
            pieces.extend(token.split("/"))

        for piece in pieces:
            normalized = normalize_name(piece)
            if normalized:
                identities.add(normalized)
    return identities


def read_f03_source_titles(path: str) -> dict[str, list[str]]:
    """
    Return title -> source ids for source papers whose 來源文獻ID starts with F03.
    """
    titles: dict[str, list[str]] = {}

    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source_id = (row.get("來源文獻ID") or "").strip()
            title = normalize_title(row.get("主要篇名", ""))
            if source_id.startswith("F03") and title:
                titles.setdefault(title, []).append(source_id)

    return titles


def read_matching_articles(path: str, f03_titles: dict[str, list[str]]) -> list[dict]:
    """
    Return articles whose title appears in F03 source papers and whose
    主要摘要 and 關鍵字 are non-empty.
    """
    matches = []

    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = normalize_title(row.get("主要篇名", ""))
            abstract = (row.get("主要摘要") or "").strip()
            keywords = (row.get("關鍵字") or "").strip()

            if title in f03_titles and abstract and keywords:
                row = dict(row)
                row["來源文獻ID"] = "|".join(sorted(set(f03_titles[title])))
                row["主要篇名"] = title
                row["關鍵字清單"] = [kw.strip() for kw in keywords.split("|") if kw.strip()]
                matches.append(row)

    return matches


def select_paper(matches: list[dict], paper_title: str | None, source_id: str | None) -> dict:
    if paper_title:
        target_title = normalize_title(paper_title)
        title_matches = [
            row for row in matches
            if normalize_title(row.get("主要篇名", "")) == target_title
        ]
        if not title_matches:
            raise SystemExit(f"找不到符合指定主要篇名的 F03 文章：{paper_title}")
        if len(title_matches) > 1:
            print(f"指定篇名找到 {len(title_matches)} 筆，使用第一筆。")
        return title_matches[0]

    if source_id:
        target_source_id = source_id.strip()
        source_matches = [
            row for row in matches
            if target_source_id in parse_pipe_field(row.get("來源文獻ID", ""))
        ]
        if not source_matches:
            raise SystemExit(f"找不到符合指定來源文獻ID的 F03 文章：{source_id}")
        if len(source_matches) > 1:
            print(f"指定來源文獻ID找到 {len(source_matches)} 筆，使用第一筆。")
        return source_matches[0]

    return random.choice(matches)


def read_author_keyword_rows(path: str) -> list[dict]:
    authors = []

    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            keywords = parse_pipe_field(row.get("關鍵字", ""))
            if not keywords:
                continue

            author = (row.get("作者") or "").strip()
            english_name = (row.get("英文姓名") or "").strip()
            identities = parse_author_identities(author)
            identities.update(parse_author_identities(english_name))

            authors.append({
                "author": author,
                "english_name": english_name,
                "article_count": int(row.get("文章數") or 0),
                "keyword_count": int(row.get("關鍵字數") or len(keywords)),
                "keywords": keywords,
                "identities": identities,
            })

    return authors


def build_author_keyword_index(authors: list[dict]) -> tuple[list[str], dict[str, list[int]]]:
    """
    Build a keyword vocabulary and an inverted index:
    author keyword -> author row indexes that own this keyword.
    """
    keyword_to_author_indexes = defaultdict(list)

    for author_index, author in enumerate(authors):
        for keyword in author["keywords"]:
            keyword_to_author_indexes[keyword].append(author_index)

    return sorted(keyword_to_author_indexes), keyword_to_author_indexes


def load_embedding_model(model: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "找不到 sentence-transformers 套件，請先執行："
            "python3 -m pip install sentence-transformers"
        ) from exc

    return SentenceTransformer(model)


def import_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit("找不到 numpy 套件，請先執行：python3 -m pip install numpy") from exc

    return np


def vector_to_list(vector) -> list[float]:
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(value) for value in vector]


def normalize_matrix(vectors):
    np = import_numpy()
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / (norms + 1e-9)


def create_embeddings(embedding_model, texts: list[str], batch_size: int, progress: bool = True):
    vectors = embedding_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=progress,
        convert_to_numpy=True,
    )
    return normalize_matrix(vectors)


def cache_vector_path(cache_path: str) -> Path:
    return Path(cache_path).with_suffix(".npy")


def load_or_create_keyword_embeddings(
    keywords: list[str],
    embedding_model,
    model_name: str,
    cache_path: str,
    batch_size: int,
):
    """
    Cache keyword embeddings as a binary matrix instead of JSON floats.
    The JSON file stores metadata and keyword order; the .npy file stores vectors.
    """
    np = import_numpy()
    meta_path = Path(cache_path)
    vector_path = cache_vector_path(cache_path)

    if meta_path.exists() and vector_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        if meta.get("model") == model_name and meta.get("keywords") == keywords:
            vectors = np.load(vector_path)
            if vectors.shape[0] == len(keywords):
                return normalize_matrix(vectors)

    print(f"Building keyword embeddings: {len(keywords)} keywords")
    vectors = create_embeddings(embedding_model, keywords, batch_size=batch_size, progress=True)
    np.save(vector_path, vectors)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model_name,
                "vector_file": vector_path.name,
                "normalized": True,
                "keywords": keywords,
            },
            f,
            ensure_ascii=False,
        )

    print(f"Saved keyword embedding metadata: {meta_path}")
    print(f"Saved keyword embedding matrix: {vector_path}")
    return vectors


def find_similar_author_keywords(
    paper_keywords: list[str],
    paper_vectors,
    author_keywords: list[str],
    author_keyword_vectors,
    keyword_top_k: int,
    similarity_threshold: float,
) -> dict[str, list[dict]]:
    np = import_numpy()
    matches_by_paper_keyword = {}
    candidate_count = min(max(keyword_top_k, 1), len(author_keywords))

    for keyword, keyword_vector in zip(paper_keywords, paper_vectors):
        scores = author_keyword_vectors @ keyword_vector
        if candidate_count == len(author_keywords):
            candidate_indexes = np.arange(len(author_keywords))
        else:
            candidate_indexes = np.argpartition(scores, -candidate_count)[-candidate_count:]

        ranked_indexes = sorted(candidate_indexes, key=lambda idx: float(scores[idx]), reverse=True)
        matches = []
        for idx in ranked_indexes:
            similarity = float(scores[idx])
            if similarity < similarity_threshold:
                continue
            author_keyword = author_keywords[int(idx)]
            matches.append({
                "paper_keyword": keyword,
                "author_keyword": author_keyword,
                "similarity": round(similarity, 6),
                "match_type": "exact" if author_keyword.lower() == keyword.lower() else "similar",
            })

        matches_by_paper_keyword[keyword] = matches

    return matches_by_paper_keyword


def recommend_authors_from_keyword_matches(
    authors: list[dict],
    keyword_to_author_indexes: dict[str, list[int]],
    matches_by_paper_keyword: dict[str, list[dict]],
    top_k: int,
    excluded_identities: set[str],
    mode: str,
    cooccur_bonus: float,
) -> list[dict]:
    per_term_scores = {}
    per_term_exact = {}
    per_term_similar = {}
    per_term_matches = {}

    for paper_keyword, matches in matches_by_paper_keyword.items():
        contrib_score = defaultdict(float)
        contrib_exact = defaultdict(int)
        contrib_similar = defaultdict(int)
        contrib_matches = defaultdict(list)

        for match in matches:
            author_keyword = match["author_keyword"]
            similarity = match["similarity"]
            for author_index in keyword_to_author_indexes.get(author_keyword, []):
                author = authors[author_index]
                if excluded_identities and author["identities"] & excluded_identities:
                    continue

                contrib_score[author_index] += similarity
                if match["match_type"] == "exact":
                    contrib_exact[author_index] += 1
                else:
                    contrib_similar[author_index] += 1
                contrib_matches[author_index].append(match)

        per_term_scores[paper_keyword] = contrib_score
        per_term_exact[paper_keyword] = contrib_exact
        per_term_similar[paper_keyword] = contrib_similar
        per_term_matches[paper_keyword] = contrib_matches

    all_author_indexes = set()
    for scores in per_term_scores.values():
        all_author_indexes.update(scores)

    results = []
    paper_keywords = list(matches_by_paper_keyword)
    for author_index in all_author_indexes:
        scores_by_term = [per_term_scores[kw].get(author_index, 0.0) for kw in paper_keywords]
        source_keyword_hits = sum(1 for score in scores_by_term if score > 0.0)

        if mode == "sum":
            score = sum(scores_by_term)
        elif mode == "avg":
            score = sum(scores_by_term) / max(1, len(paper_keywords))
        elif mode == "min":
            score = min(scores_by_term)
        elif mode == "soft_and":
            score = sum(scores_by_term) + cooccur_bonus * max(0, source_keyword_hits - 1)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        exact_hits = sum(per_term_exact[kw].get(author_index, 0) for kw in paper_keywords)
        similar_hits = sum(per_term_similar[kw].get(author_index, 0) for kw in paper_keywords)
        matched_keywords = []
        seen_matches = set()
        for kw in paper_keywords:
            for match in per_term_matches[kw].get(author_index, []):
                key = (match["paper_keyword"], match["author_keyword"])
                if key in seen_matches:
                    continue
                seen_matches.add(key)
                matched_keywords.append(match)

        matched_keywords.sort(key=lambda item: -item["similarity"])
        author = authors[author_index]
        results.append({
            "author": author["author"],
            "english_name": author["english_name"],
            "article_count": author["article_count"],
            "keyword_count": author["keyword_count"],
            "score": round(float(score), 6),
            "source_keyword_hits": source_keyword_hits,
            "exact_hits": exact_hits,
            "similar_hits": similar_hits,
            "matched_keywords": matched_keywords,
            "author_keywords": author["keywords"],
        })

    return sorted(
        results,
        key=lambda item: (
            -item["score"],
            -item["source_keyword_hits"],
            -item["exact_hits"],
            -item["similar_hits"],
            -item["article_count"],
            item["author"],
        ),
    )[:top_k]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter F03 papers, randomly sample one, and recommend authors by keyword embeddings."
    )
    parser.add_argument("--source-file", default=SOURCE_FILE)
    parser.add_argument("--articles-file", default=ARTICLES_FILE)
    parser.add_argument("--author-file", default=AUTHOR_KEYWORDS_FILE)
    parser.add_argument("--output-file", default=OUTPUT_FILE)
    parser.add_argument("--cache-file", default=EMBEDDING_CACHE_FILE)
    parser.add_argument("--model", default=EMBEDDING_MODEL)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--paper-title", help="Specify a paper by exact 主要篇名 instead of random sampling.")
    parser.add_argument("--source-id", help="Specify a paper by exact 來源文獻ID instead of random sampling.")
    parser.add_argument(
        "--keyword-top-k",
        type=int,
        default=KEYWORD_TOP_K,
        help="For each paper keyword, keep this many nearest author keywords before scoring authors.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=SIMILARITY_THRESHOLD,
        help="Minimum cosine similarity for paper keyword -> author keyword matches.",
    )
    parser.add_argument(
        "--score-mode",
        choices=["sum", "avg", "min", "soft_and"],
        default="soft_and",
        help="How to combine scores across paper keywords.",
    )
    parser.add_argument(
        "--cooccur-bonus",
        type=float,
        default=COOCCUR_BONUS,
        help="Bonus per additional matched paper keyword when --score-mode soft_and is used.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, help="Set a seed to make random sampling reproducible.")
    parser.add_argument(
        "--include-paper-authors",
        action="store_true",
        help="Allow recommending the sampled paper's own authors.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the sampled paper and keyword text; do not create embeddings.",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    f03_titles = read_f03_source_titles(args.source_file)
    matches = read_matching_articles(args.articles_file, f03_titles)

    if not matches:
        raise SystemExit("找不到符合條件的文章。")

    sampled = select_paper(
        matches=matches,
        paper_title=args.paper_title,
        source_id=args.source_id,
    )
    keyword_text = " | ".join(sampled["關鍵字清單"])

    print(f"F03 source titles: {len(f03_titles)}")
    print(f"Matched eligible articles: {len(matches)}")
    print(f"Sampled title: {sampled['主要篇名']}")
    print(f"Source ID: {sampled['來源文獻ID']}")
    print(f"Paper authors: {(sampled.get('作者') or '').strip()}")
    print(f"Keywords: {keyword_text}")

    if args.dry_run:
        print("Dry run only; embeddings were not created.")
        return

    authors = read_author_keyword_rows(args.author_file)
    print(f"Authors with keywords: {len(authors)}")
    author_keywords, keyword_to_author_indexes = build_author_keyword_index(authors)
    print(f"Unique author keywords: {len(author_keywords)}")

    embedding_model = load_embedding_model(args.model)
    paper_keyword_vectors = create_embeddings(
        embedding_model=embedding_model,
        texts=sampled["關鍵字清單"],
        batch_size=args.batch_size,
        progress=False,
    )
    author_keyword_vectors = load_or_create_keyword_embeddings(
        keywords=author_keywords,
        embedding_model=embedding_model,
        model_name=args.model,
        cache_path=args.cache_file,
        batch_size=args.batch_size,
    )

    excluded_identities = set()
    if not args.include_paper_authors:
        excluded_identities = parse_author_identities(sampled.get("作者", ""))

    matches_by_paper_keyword = find_similar_author_keywords(
        paper_keywords=sampled["關鍵字清單"],
        paper_vectors=paper_keyword_vectors,
        author_keywords=author_keywords,
        author_keyword_vectors=author_keyword_vectors,
        keyword_top_k=args.keyword_top_k,
        similarity_threshold=args.similarity_threshold,
    )

    recommendations = recommend_authors_from_keyword_matches(
        authors=authors,
        keyword_to_author_indexes=keyword_to_author_indexes,
        matches_by_paper_keyword=matches_by_paper_keyword,
        top_k=args.top_k,
        excluded_identities=excluded_identities,
        mode=args.score_mode,
        cooccur_bonus=args.cooccur_bonus,
    )

    paper_keyword_embeddings = [
        {
            "keyword": keyword,
            "embedding": vector_to_list(vector),
        }
        for keyword, vector in zip(sampled["關鍵字清單"], paper_keyword_vectors)
    ]

    output = {
        "model": args.model,
        "source_id": sampled["來源文獻ID"],
        "title": sampled["主要篇名"],
        "paper_authors": sampled.get("作者", ""),
        "keywords": sampled["關鍵字清單"],
        "embedding_dimension": int(paper_keyword_vectors.shape[1]),
        "paper_keyword_embeddings": paper_keyword_embeddings,
        "recommendation_settings": {
            "keyword_top_k": args.keyword_top_k,
            "similarity_threshold": args.similarity_threshold,
            "score_mode": args.score_mode,
            "cooccur_bonus": args.cooccur_bonus,
            "include_paper_authors": args.include_paper_authors,
        },
        "paper_keyword_matches": matches_by_paper_keyword,
        "recommendations": recommendations,
    }

    output_path = Path(args.output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Embedding dimension: {int(paper_keyword_vectors.shape[1])}")
    print(f"Top {len(recommendations)} recommended authors:")
    for index, item in enumerate(recommendations, 1):
        top_matches = item["matched_keywords"][:3]
        match_preview = " | ".join(
            f"{m['paper_keyword']}->{m['author_keyword']}({m['similarity']:.3f})"
            for m in top_matches
        ) or "-"
        print(
            f"{index}. {item['author']} "
            f"(score={item['score']:.4f}, "
            f"source_hits={item['source_keyword_hits']}, "
            f"exact={item['exact_hits']}, similar={item['similar_hits']}, "
            f"articles={item['article_count']}, matches={match_preview})"
        )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
