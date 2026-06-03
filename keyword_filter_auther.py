#!/usr/bin/env python3
"""
Pick one or more eligible F03 source papers and recommend authors by keywords.

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
OUTPUT_FILE = "20X20_author_recommendations.csv"
JSON_OUTPUT_FILE = ""
EMBEDDING_CACHE_FILE = "author_keyword_embeddings_cache.json"
EMBEDDING_MODEL = "BAAI/bge-m3"
KEYWORD_TOP_K = 30
SIMILARITY_THRESHOLD = 0.55
COOCCUR_BONUS = 0.2
ABSTRACT_TOP_K = 3
ABSTRACT_THRESHOLD = 0.45
ABSTRACT_BATCH_SIZE = 8
MAX_SEQ_LENGTH = 512


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


def select_papers(
    matches: list[dict],
    paper_title: str | None,
    source_id: str | None,
    count: int,
) -> list[dict]:
    if paper_title or source_id:
        return [select_paper(matches, paper_title, source_id)]

    if count <= 0:
        raise SystemExit("--num-papers 必須大於 0。")

    if count >= len(matches):
        return list(matches)

    return random.sample(matches, count)


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


def read_article_rows(path: str) -> list[dict]:
    articles = []

    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row_index, row in enumerate(reader):
            title = normalize_title(row.get("主要篇名", ""))
            abstract = (row.get("主要摘要") or "").strip()
            author_field = (row.get("作者") or "").strip()
            if not title or not abstract or not author_field:
                continue

            articles.append({
                "article_index": row_index,
                "title": title,
                "english_title": (row.get("英文篇名") or "").strip(),
                "authors": author_field,
                "abstract": abstract,
                "keywords": parse_pipe_field(row.get("關鍵字", "")),
                "identities": parse_author_identities(author_field),
            })

    return articles


def build_author_article_index(articles: list[dict]) -> dict[str, list[int]]:
    identity_to_article_indexes = defaultdict(list)

    for article_index, article in enumerate(articles):
        for identity in article["identities"]:
            identity_to_article_indexes[identity].append(article_index)

    return identity_to_article_indexes


def candidate_article_indexes_for_author(
    author: dict,
    articles: list[dict],
    identity_to_article_indexes: dict[str, list[int]],
    target_title: str,
) -> list[int]:
    article_indexes = set()

    for identity in author["identities"]:
        article_indexes.update(identity_to_article_indexes.get(identity, []))

    return sorted(
        article_index for article_index in article_indexes
        if articles[article_index]["title"] != target_title
    )


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


def load_or_create_text_embeddings(
    texts: list[str],
    embedding_model,
    model_name: str,
    cache_path: str,
    batch_size: int,
    item_key: str,
    item_label: str,
):
    """
    Cache text embeddings as a binary matrix instead of JSON floats.
    The JSON file stores metadata and text order; the .npy file stores vectors.
    """
    np = import_numpy()
    meta_path = Path(cache_path)
    vector_path = cache_vector_path(cache_path)

    if meta_path.exists() and vector_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        if meta.get("model") == model_name and meta.get(item_key) == texts:
            vectors = np.load(vector_path)
            if vectors.shape[0] == len(texts):
                return normalize_matrix(vectors)

    print(f"Building {item_label} embeddings: {len(texts)} {item_label}s")
    vectors = create_embeddings(embedding_model, texts, batch_size=batch_size, progress=True)
    np.save(vector_path, vectors)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model_name,
                "vector_file": vector_path.name,
                "normalized": True,
                item_key: texts,
            },
            f,
            ensure_ascii=False,
        )

    print(f"Saved {item_label} embedding metadata: {meta_path}")
    print(f"Saved {item_label} embedding matrix: {vector_path}")
    return vectors


def load_or_create_keyword_embeddings(
    keywords: list[str],
    embedding_model,
    model_name: str,
    cache_path: str,
    batch_size: int,
):
    return load_or_create_text_embeddings(
        texts=keywords,
        embedding_model=embedding_model,
        model_name=model_name,
        cache_path=cache_path,
        batch_size=batch_size,
        item_key="keywords",
        item_label="keyword",
    )


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


def attach_abstract_evidence(
    recommendations: list[dict],
    authors: list[dict],
    articles: list[dict],
    identity_to_article_indexes: dict[str, list[int]],
    embedding_model,
    target_title: str,
    target_abstract: str,
    abstract_top_k: int,
    abstract_threshold: float,
    abstract_batch_size: int,
) -> list[dict]:
    if not recommendations:
        return recommendations

    author_by_name = {
        (author["author"], author["english_name"]): author
        for author in authors
    }

    recommendation_article_indexes = {}
    all_candidate_indexes = set()
    for recommendation in recommendations:
        author = author_by_name.get((
            recommendation["author"],
            recommendation["english_name"],
        ))
        if not author:
            recommendation_article_indexes[id(recommendation)] = []
            continue

        article_indexes = candidate_article_indexes_for_author(
            author=author,
            articles=articles,
            identity_to_article_indexes=identity_to_article_indexes,
            target_title=target_title,
        )
        recommendation_article_indexes[id(recommendation)] = article_indexes
        all_candidate_indexes.update(article_indexes)

    if not all_candidate_indexes:
        for recommendation in recommendations:
            recommendation["abstract_score"] = 0.0
            recommendation["abstract_matches"] = []
        return recommendations

    sorted_candidate_indexes = sorted(all_candidate_indexes)
    target_vector = create_embeddings(
        embedding_model=embedding_model,
        texts=[target_abstract],
        batch_size=1,
        progress=False,
    )[0]
    candidate_vectors = create_embeddings(
        embedding_model=embedding_model,
        texts=[articles[index]["abstract"] for index in sorted_candidate_indexes],
        batch_size=abstract_batch_size,
        progress=False,
    )
    vector_by_article_index = {
        article_index: candidate_vectors[offset]
        for offset, article_index in enumerate(sorted_candidate_indexes)
    }

    for recommendation in recommendations:
        matches = []
        for article_index in recommendation_article_indexes.get(id(recommendation), []):
            similarity = float(vector_by_article_index[article_index] @ target_vector)
            if similarity < abstract_threshold:
                continue

            article = articles[article_index]
            matches.append({
                "title": article["title"],
                "english_title": article["english_title"],
                "authors": article["authors"],
                "similarity": round(similarity, 6),
                "keywords": article["keywords"],
                "abstract": article["abstract"],
            })

        matches.sort(key=lambda item: -item["similarity"])
        matches = matches[:abstract_top_k]
        abstract_score = 0.0
        if matches:
            abstract_score = sum(match["similarity"] for match in matches) / len(matches)

        recommendation["abstract_score"] = round(float(abstract_score), 6)
        recommendation["abstract_matches"] = matches

    return recommendations


def recommend_for_paper(
    sampled: dict,
    args,
    authors: list[dict],
    author_keywords: list[str],
    keyword_to_author_indexes: dict[str, list[int]],
    author_keyword_vectors,
    embedding_model,
    articles: list[dict],
    identity_to_article_indexes: dict[str, list[int]],
) -> dict:
    paper_keyword_vectors = create_embeddings(
        embedding_model=embedding_model,
        texts=sampled["關鍵字清單"],
        batch_size=args.batch_size,
        progress=False,
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

    for recommendation in recommendations:
        recommendation["keyword_score"] = recommendation["score"]
        recommendation["final_score"] = recommendation["score"]

    recommendations = attach_abstract_evidence(
        recommendations=recommendations,
        authors=authors,
        articles=articles,
        identity_to_article_indexes=identity_to_article_indexes,
        embedding_model=embedding_model,
        target_title=sampled["主要篇名"],
        target_abstract=(sampled.get("主要摘要") or "").strip(),
        abstract_top_k=args.abstract_top_k,
        abstract_threshold=args.abstract_threshold,
        abstract_batch_size=args.abstract_batch_size,
    )

    return {
        "source_id": sampled["來源文獻ID"],
        "title": sampled["主要篇名"],
        "paper_authors": sampled.get("作者", ""),
        "paper_abstract": (sampled.get("主要摘要") or "").strip(),
        "keywords": sampled["關鍵字清單"],
        "recommendations": recommendations,
        "paper_keyword_matches": matches_by_paper_keyword,
    }


def format_keyword_matches(matches: list[dict], limit: int = 5) -> str:
    return " | ".join(
        f"{match['paper_keyword']}->{match['author_keyword']}({match['similarity']:.3f})"
        for match in matches[:limit]
    )


def recommendation_csv_rows(results: list[dict], abstract_columns: int) -> list[dict]:
    rows = []

    for paper_rank, result in enumerate(results, 1):
        for author_rank, recommendation in enumerate(result["recommendations"], 1):
            row = {
                "paper_rank": paper_rank,
                "source_id": result["source_id"],
                "paper_title": result["title"],
                "paper_authors": result["paper_authors"],
                "paper_abstract": result["paper_abstract"],
                "paper_keywords": "|".join(result["keywords"]),
                "recommended_rank": author_rank,
                "author": recommendation["author"],
                "english_name": recommendation["english_name"],
                "keyword_score": recommendation["keyword_score"],
                "abstract_score": recommendation["abstract_score"],
                "final_score": recommendation["final_score"],
                "source_keyword_hits": recommendation["source_keyword_hits"],
                "exact_hits": recommendation["exact_hits"],
                "similar_hits": recommendation["similar_hits"],
                "author_article_count": recommendation["article_count"],
                "author_keyword_count": recommendation["keyword_count"],
                "matched_keywords": format_keyword_matches(recommendation["matched_keywords"]),
                "author_keywords": "|".join(recommendation["author_keywords"]),
            }

            for index in range(abstract_columns):
                prefix = f"abstract_{index + 1}"
                match = (
                    recommendation["abstract_matches"][index]
                    if index < len(recommendation["abstract_matches"])
                    else None
                )
                row[f"{prefix}_similarity"] = match["similarity"] if match else ""
                row[f"{prefix}_title"] = match["title"] if match else ""
                row[f"{prefix}_english_title"] = match["english_title"] if match else ""
                row[f"{prefix}_authors"] = match["authors"] if match else ""
                row[f"{prefix}_keywords"] = "|".join(match["keywords"]) if match else ""
                row[f"{prefix}_abstract"] = match["abstract"] if match else ""

            rows.append(row)

    return rows


def write_recommendations_csv(path: str, rows: list[dict], abstract_columns: int) -> None:
    base_fields = [
        "paper_rank",
        "source_id",
        "paper_title",
        "paper_authors",
        "paper_abstract",
        "paper_keywords",
        "recommended_rank",
        "author",
        "english_name",
        "keyword_score",
        "abstract_score",
        "final_score",
        "source_keyword_hits",
        "exact_hits",
        "similar_hits",
        "author_article_count",
        "author_keyword_count",
        "matched_keywords",
        "author_keywords",
    ]
    abstract_fields = []
    for index in range(1, abstract_columns + 1):
        abstract_fields.extend([
            f"abstract_{index}_similarity",
            f"abstract_{index}_title",
            f"abstract_{index}_english_title",
            f"abstract_{index}_authors",
            f"abstract_{index}_keywords",
            f"abstract_{index}_abstract",
        ])

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=base_fields + abstract_fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter F03 papers and recommend authors by keyword embeddings."
    )
    parser.add_argument("--source-file", default=SOURCE_FILE)
    parser.add_argument("--articles-file", default=ARTICLES_FILE)
    parser.add_argument("--author-file", default=AUTHOR_KEYWORDS_FILE)
    parser.add_argument("--output-file", default=OUTPUT_FILE)
    parser.add_argument(
        "--json-output-file",
        default=JSON_OUTPUT_FILE,
        help="Optional JSON output path. Leave empty to only write CSV.",
    )
    parser.add_argument("--cache-file", default=EMBEDDING_CACHE_FILE)
    parser.add_argument("--model", default=EMBEDDING_MODEL)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--num-papers",
        type=int,
        default=20,
        help="Randomly sample this many eligible papers when --paper-title/--source-id is not provided.",
    )
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
    parser.add_argument(
        "--abstract-top-k",
        type=int,
        default=ABSTRACT_TOP_K,
        help="Keep this many most similar articles per recommended author.",
    )
    parser.add_argument(
        "--abstract-threshold",
        type=float,
        default=ABSTRACT_THRESHOLD,
        help="Minimum target abstract -> recommended author article abstract similarity to keep as evidence.",
    )
    parser.add_argument(
        "--abstract-batch-size",
        type=int,
        default=ABSTRACT_BATCH_SIZE,
        help="Batch size for embedding recommended authors' article abstracts.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=MAX_SEQ_LENGTH,
        help="Maximum token length for the sentence-transformers model.",
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

    sampled_papers = select_papers(
        matches=matches,
        paper_title=args.paper_title,
        source_id=args.source_id,
        count=args.num_papers,
    )

    print(f"F03 source titles: {len(f03_titles)}")
    print(f"Matched eligible articles: {len(matches)}")
    print(f"Selected papers: {len(sampled_papers)}")
    for index, sampled in enumerate(sampled_papers, 1):
        keyword_text = " | ".join(sampled["關鍵字清單"])
        print(f"[{index}] {sampled['來源文獻ID']} | {sampled['主要篇名']}")
        print(f"    Paper authors: {(sampled.get('作者') or '').strip()}")
        print(f"    Keywords: {keyword_text}")

    if args.dry_run:
        print("Dry run only; embeddings were not created.")
        return

    authors = read_author_keyword_rows(args.author_file)
    print(f"Authors with keywords: {len(authors)}")
    author_keywords, keyword_to_author_indexes = build_author_keyword_index(authors)
    print(f"Unique author keywords: {len(author_keywords)}")
    articles = read_article_rows(args.articles_file)
    identity_to_article_indexes = build_author_article_index(articles)
    print(f"Articles with abstracts: {len(articles)}")

    embedding_model = load_embedding_model(args.model)
    if args.max_seq_length:
        embedding_model.max_seq_length = args.max_seq_length
        print(f"Model max_seq_length: {embedding_model.max_seq_length}")
    author_keyword_vectors = load_or_create_keyword_embeddings(
        keywords=author_keywords,
        embedding_model=embedding_model,
        model_name=args.model,
        cache_path=args.cache_file,
        batch_size=args.batch_size,
    )

    results = []
    for index, sampled in enumerate(sampled_papers, 1):
        print(f"\nRunning paper {index}/{len(sampled_papers)}: {sampled['來源文獻ID']} {sampled['主要篇名']}")
        result = recommend_for_paper(
            sampled=sampled,
            args=args,
            authors=authors,
            author_keywords=author_keywords,
            keyword_to_author_indexes=keyword_to_author_indexes,
            author_keyword_vectors=author_keyword_vectors,
            embedding_model=embedding_model,
            articles=articles,
            identity_to_article_indexes=identity_to_article_indexes,
        )
        results.append(result)

        print(f"Top {len(result['recommendations'])} recommended authors:")
        for author_rank, item in enumerate(result["recommendations"], 1):
            top_matches = item["matched_keywords"][:3]
            match_preview = format_keyword_matches(top_matches, limit=3) or "-"
            print(
                f"  {author_rank}. {item['author']} "
                f"(keyword_score={item['keyword_score']:.4f}, "
                f"abstract_score={item['abstract_score']:.4f}, "
                f"final_score={item['final_score']:.4f}, "
                f"keyword_matches={match_preview})"
            )

    csv_rows = recommendation_csv_rows(results, abstract_columns=args.abstract_top_k)
    write_recommendations_csv(args.output_file, csv_rows, abstract_columns=args.abstract_top_k)
    print(f"\nSaved CSV: {args.output_file}")
    print(f"CSV rows: {len(csv_rows)}")

    if args.json_output_file:
        json_output = {
            "model": args.model,
            "papers": results,
            "row_count": len(csv_rows),
            "paper_count": len(results),
            "recommendations_per_paper": args.top_k,
            "recommendation_settings": {
                "keyword_top_k": args.keyword_top_k,
                "similarity_threshold": args.similarity_threshold,
                "score_mode": args.score_mode,
                "cooccur_bonus": args.cooccur_bonus,
                "include_paper_authors": args.include_paper_authors,
                "abstract_top_k": args.abstract_top_k,
                "abstract_threshold": args.abstract_threshold,
            },
        }
        with open(args.json_output_file, "w", encoding="utf-8") as f:
            json.dump(json_output, f, ensure_ascii=False, indent=2)
        print(f"Saved JSON: {args.json_output_file}")


if __name__ == "__main__":
    main()
