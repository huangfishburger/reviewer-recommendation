#!/usr/bin/env python3
"""
Build an undirected author citation graph from ref_pair.csv + articles_keywords.csv.

Node ID   : canonical name (Chinese name preferred; English-only if no Chinese)
Attributes: en_name  — for matching against author_keywords.csv
Edges     : between two authors who have a citation OR co-authorship relationship
Weight    : (# citing/cited paper pairs) + (# co-authored papers)
Output    : author_citation_graph.graphml
"""

import csv
import sys
from collections import defaultdict

import networkx as nx

sys.stdout.reconfigure(encoding="utf-8")

REF_PAIR_FILE  = "ref_pair.csv"
ARTICLES_FILE  = "articles_keywords.csv"
OUTPUT_FILE    = "author_citation_graph.graphml"


def is_chinese_char(c: str) -> bool:
    return "一" <= c <= "鿿"


def has_chinese(s: str) -> bool:
    return any(is_chinese_char(c) for c in s)


def parse_name_token(token: str) -> tuple[str, str]:
    """
    Parse one author token.
    Returns (node_id, en_name).
    node_id = Chinese name if present, else English name.
    en_name = English name if present, else "".
    """
    token = token.strip()
    if not token:
        return ("", "")
    if "=" in token:
        left, _, right = token.partition("=")
        left, right = left.strip(), right.strip()
        if has_chinese(left):
            return (left, right)
        if has_chinese(right):
            return (right, left)
        return (left or right, left or right)
    if has_chinese(token):
        return (token, "")
    return (token, token)


def parse_authors(field: str) -> list[tuple[str, str]]:
    """Return list of (node_id, en_name) for each author in field."""
    result = []
    for token in field.split("|"):
        node_id, en = parse_name_token(token)
        if node_id:
            result.append((node_id, en))
    return result


def build_paper_author_lookup(path: str) -> dict[str, list[tuple[str, str]]]:
    """paper_id -> [(node_id, en_name), ...]"""
    lookup: dict[str, list[tuple[str, str]]] = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            paper_id = row.get("文獻ID", "").strip()
            authors  = parse_authors(row.get("作者", ""))
            if paper_id and authors:
                lookup[paper_id] = authors
    return lookup


def build_author_graph(
    ref_pair_path: str,
    articles_path: str,
    paper_author_lookup: dict[str, list[tuple[str, str]]],
) -> nx.Graph:
    edge_weights: dict[tuple[str, str], int] = defaultdict(int)
    node_info: dict[str, set[str]] = {}  # node_id -> set of en_names seen

    def register(node_id: str, en: str) -> None:
        if node_id not in node_info:
            node_info[node_id] = set()
        if en:
            node_info[node_id].add(en)

    def add_edge(a: str, b: str) -> None:
        if a == b:
            return
        key = (min(a, b), max(a, b))
        edge_weights[key] += 1

    # --- citation edges from ref_pair.csv ---
    with open(ref_pair_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            citing = parse_authors(row.get("作者", ""))
            cited_id = row.get("經濟學門文獻ID", "").strip()
            cited = paper_author_lookup.get(cited_id, [])

            if not citing or not cited:
                continue

            for ca_id, ca_en in citing:
                register(ca_id, ca_en)
            for cd_id, cd_en in cited:
                register(cd_id, cd_en)

            for ca_id, _ in citing:
                for cd_id, _ in cited:
                    add_edge(ca_id, cd_id)

    # --- co-authorship edges from articles_keywords.csv ---
    with open(articles_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            authors = parse_authors(row.get("作者", ""))
            if len(authors) < 2:
                continue
            for a_id, a_en in authors:
                register(a_id, a_en)
            for i, (a_id, _) in enumerate(authors):
                for b_id, _ in authors[i + 1:]:
                    add_edge(a_id, b_id)

    G = nx.Graph()
    for (a, b), weight in edge_weights.items():
        G.add_edge(a, b, weight=weight)

    for node_id, en_set in node_info.items():
        if G.has_node(node_id):
            G.nodes[node_id]["en_name"] = " / ".join(sorted(en_set))

    return G


def print_stats(G: nx.Graph) -> None:
    print(f"Nodes (authors) : {G.number_of_nodes()}")
    print(f"Edges           : {G.number_of_edges()}")

    weights = [d["weight"] for _, _, d in G.edges(data=True)]
    print(f"Max edge weight : {max(weights)}")
    print(f"Avg edge weight : {sum(weights)/len(weights):.2f}")

    print("\nTop 10 by degree:")
    for node, deg in sorted(G.degree(), key=lambda x: -x[1])[:10]:
        strength = sum(d["weight"] for _, _, d in G.edges(node, data=True))
        en = G.nodes[node].get("en_name", "")
        print(f"  {node:15s}  {en:25s}  degree={deg}  strength={strength}")


def main() -> None:
    print(f"Reading {ARTICLES_FILE} ...")
    paper_author_lookup = build_paper_author_lookup(ARTICLES_FILE)
    print(f"  Papers with authors: {len(paper_author_lookup)}")

    print(f"Reading {REF_PAIR_FILE} + co-authorship from {ARTICLES_FILE} ...")
    G = build_author_graph(REF_PAIR_FILE, ARTICLES_FILE, paper_author_lookup)
    print_stats(G)

    nx.write_graphml(G, OUTPUT_FILE)
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
