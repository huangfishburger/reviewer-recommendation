#!/usr/bin/env python3
"""
Build an undirected citation graph from ref_pair.csv.

Nodes : paper IDs (any prefix — F, E, T, G, S, A, ...)
Edges : undirected edge between citing paper and cited paper
        (one row in ref_pair = one edge; duplicate pairs are deduplicated)
Output: citation_graph.graphml
"""

import csv
import re
import sys
from collections import Counter

import networkx as nx

sys.stdout.reconfigure(encoding="utf-8")

INPUT_FILE  = "ref_pair.csv"
OUTPUT_FILE = "citation_graph.graphml"

CITING_COL = "經濟學門_被引來源文獻ID"
CITED_COL  = "經濟學門文獻ID"


def id_prefix(paper_id: str) -> str:
    m = re.match(r"^[A-Za-z]+", paper_id)
    return m.group() if m else "?"


def build_graph(path: str) -> nx.Graph:
    G = nx.Graph()

    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            citing = row[CITING_COL].strip()
            cited  = row[CITED_COL].strip()
            if not citing or not cited:
                continue

            for pid in (citing, cited):
                if not G.has_node(pid):
                    G.add_node(pid, prefix=id_prefix(pid))

            G.add_edge(citing, cited)  # duplicate pairs become one edge

    return G


def print_stats(G: nx.Graph) -> None:
    prefix_counts = Counter(d["prefix"] for _, d in G.nodes(data=True))
    print(f"Nodes : {G.number_of_nodes()}")
    for prefix, count in sorted(prefix_counts.items(), key=lambda x: -x[1]):
        print(f"  {prefix:4s} {count}")
    print(f"Edges : {G.number_of_edges()}")

    print("\nTop 10 by degree:")
    for node, deg in sorted(G.degree(), key=lambda x: -x[1])[:10]:
        print(f"  {node:15s}  degree={deg}  prefix={G.nodes[node]['prefix']}")


def main() -> None:
    print(f"Reading {INPUT_FILE} ...")
    G = build_graph(INPUT_FILE)
    print_stats(G)

    nx.write_graphml(G, OUTPUT_FILE)
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
