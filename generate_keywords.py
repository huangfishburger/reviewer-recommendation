#!/usr/bin/env python3
"""
Generate keywords for articles missing them in articles.csv.
Uses OpenAI API with few-shot prompting from example.txt.
Saves checkpoint so it can resume if interrupted.
"""

import csv
import os
import time
import json
import sys

try:
    from openai import OpenAI
except ImportError:
    print("Installing openai...")
    os.system(f"{sys.executable} -m pip install openai")
    from openai import OpenAI

INPUT_FILE = "articles.csv"
OUTPUT_FILE = "articles_updated.csv"
CHECKPOINT_FILE = "keywords_checkpoint.json"
MODEL = "gpt-4o-mini"
BATCH_SIZE = 5       # articles per API call
SAVE_EVERY = 50      # checkpoint frequency
SLEEP_BETWEEN = 0.3  # seconds between API calls


def load_env(filename=".env"):
    if not os.path.exists(filename):
        return
    with open(filename, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


load_env()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

SYSTEM_PROMPT = (
    "You are a research librarian. Given an academic paper's title and abstract, "
    "generate 4-6 keywords that capture the main topics, methods, and contributions. "
    "Output Chinese keywords first, then their English translations, all joined by | with no spaces around |. "
    "Format: 中文詞1|中文詞2|中文詞3|English term1|English term2|English term3 — nothing else."
)

FEW_SHOT_BLOCK = """\
example 1:
title: RecMind: Large Language Model Powered Agent For Recommendation
abstract: While the recommendation system (RS) has advanced significantly through deep learning, current RS approaches usually train and fine-tune models on task-specific datasets, limiting their generalizability to new recommendation tasks and their ability to leverage external knowledge due to model scale and data size constraints. Thus, we designed an LLM-powered autonomous recommender agent, RecMind, which is capable of leveraging external knowledge, utilizing tools with careful planning to provide zero-shot personalized recommendations. We propose a Self-Inspiring algorithm to improve the planning ability. At each intermediate step, the LLM "self-inspires" to consider all previously explored states to plan for the next step. This mechanism greatly improves the model's ability to comprehend and utilize historical information in planning for recommendation. We evaluate RecMind's performance in various recommendation scenarios. Our experiment shows that RecMind outperforms existing zero/few-shot LLM-based recommendation baseline methods in various tasks and achieves comparable performance to a fully trained recommendation model P5.
keywords: 大語言模型推薦|工具利用|推薦策略規劃|個性化推薦|LLM-based recommendation|Tool utilization|Recommendation Strategy Planning|Personalized Recommendation

example 2:
title: RATT: A Thought Structure for Coherent and Correct LLM Reasoning
abstract: Large Language Models (LLMs) gain substantial reasoning and decision-making capabilities from thought structures. However, existing methods such as Tree of Thought and Retrieval Augmented Thoughts often fall short in complex tasks due to the limitations of insufficient local retrieval of factual knowledge and inadequate global selection of strategies. These limitations make it challenging for these methods to balance factual accuracy and comprehensive logical optimization effectively. To address these limitations, we introduce the Retrieval Augmented Thought Tree (RATT), a novel thought structure that considers both overall logical soundness and factual correctness at each step of the thinking process. Specifically, at every point of a thought branch, RATT performs planning and lookahead to explore and evaluate multiple potential reasoning steps, and integrate the fact-checking ability of Retrieval-Augmented Generation (RAG) with LLMs' ability to assess overall strategy. Through this combination of factual knowledge and strategic feasibility, the RATT adjusts and integrates the thought tree structure to search for the most promising branches within the search space. This thought structure significantly enhances the model's coherence in logical inference and efficiency in decision-making, and thus increases the limit of the capacity of LLMs to generate reliable inferences and decisions based on thought structures. A broad range of experiments on different types of tasks showcases that the RATT structure significantly outperforms existing methods in factual correctness and logical coherence.
keywords: 思維樹|檢索增強思維|思維結構|大語言模型推理|Tree of Thought|Retrieval Augmented Thoughts|Thought Structure|LLM reasoning

example 3:
title: Re2LLM: Reflective Reinforcement Large Language Model for Session-based Recommendation
abstract: Large Language Models (LLMs) are emerging as promising approaches to enhance session-based recommendation (SBR), where both prompt-based and fine-tuning-based methods have been widely investigated to align LLMs with SBR. However, the former methods struggle with optimal prompts to elicit the correct reasoning of LLMs due to the lack of task-specific feedback, leading to unsatisfactory recommendations. Although the latter methods attempt to fine-tune LLMs with domain-specific knowledge, they face limitations such as high computational costs and reliance on opensource backbones. To address such issues, we propose a Reflective Reinforcement Large Language Model (Re2LLM) for SBR, guiding LLMs to focus on specialized knowledge essential for more accurate recommendations effectively and efficiently. In particular, we first design the Reflective Exploration Module to effectively extract knowledge that is readily understandable and digestible by LLMs. To be specific, we direct LLMs to examine recommendation errors through self-reflection and construct a knowledge base (KB) comprising hints capable of rectifying these errors. To efficiently elicit the correct reasoning of LLMs, we further devise the Reinforcement Utilization Module to train a lightweight retrieval agent. It learns to select hints from the constructed KB based on the task specific feedback, where the hints can serve as guidance to help correct LLMs reasoning for better recommendations. Extensive experiments on multiple real-world datasets demonstrate that our method consistently outperforms state-of-the-art methods.
keywords: 基於會話的推薦|反思性探索|強化利用|知識庫構建|session-based recommendation|Reflective Exploration|Reinforcement Utilization|Knowledge Base Construction

example 4:
title: Process-Supervised LLM Recommenders via Flow-guided Tuning
abstract: While large language models (LLMs) are increasingly adapted for recommendation systems via supervised fine-tuning (SFT), this approach amplifies popularity bias due to its likelihood maximization objective, compromising recommendation diversity and fairness. To address this, we present Flow-guided fine-tuning recommender (Flower), which replaces SFT with a Generative Flow Network (GFlowNet) framework that enacts process supervision through token-level reward propagation. Flower's key innovation lies in decomposing item-level rewards into constituent token rewards, enabling direct alignment between token generation probabilities and their reward signals. This mechanism achieves three critical advancements: (1) popularity bias mitigation and fairness enhancement through empirical distribution matching, (2) preservation of diversity through GFlowNet's proportional sampling, and (3) flexible integration of personalized preferences via adaptable token rewards. Experiments demonstrate Flower's superior distribution-fitting capability and its significant advantages over traditional SFT in terms of accuracy, fairness, and diversity, highlighting its potential to improve LLM-based recommendation systems.
keywords: 流引導微調|過程監督|推薦多樣性|標記級對齊|Flow-guided Fine-tuning|Process Supervision|Recommendation Diversity|Token-level Alignment"""


def build_batch_prompt(articles: list[tuple[str, str]]) -> str:
    """articles: list of (title, abstract)"""
    lines = [FEW_SHOT_BLOCK, "\nNow generate keywords for each article below.",
             "Output exactly one line per article in the format:  N: keywords",
             "where N is the article number and keywords uses the 中文|English| format.\n"]
    for i, (title, abstract) in enumerate(articles, 1):
        lines.append(f"article {i}:")
        lines.append(f"title: {title}")
        lines.append(f"abstract: {abstract}")
        lines.append("")
    return "\n".join(lines)


def parse_batch_response(text: str, expected: int) -> list[str]:
    """Parse numbered lines like '1: keyword|keyword|...' into a list."""
    results = [""] * expected
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            prefix, _, rest = line.partition(":")
            try:
                idx = int(prefix.strip()) - 1
                if 0 <= idx < expected:
                    results[idx] = rest.strip()
            except ValueError:
                pass
    return results


def call_api_batch(articles: list[tuple[str, str]], max_retries: int = 3) -> list[str]:
    prompt = build_batch_prompt(articles)
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=200 * len(articles),
                temperature=0.3,
            )
            raw = response.choices[0].message.content.strip()
            return parse_batch_response(raw, len(articles))
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt * 2
                print(f"  Error: {e}. Retrying in {wait}s...", flush=True)
                time.sleep(wait)
            else:
                print(f"  Failed after {max_retries} attempts: {e}", flush=True)
                return [""] * len(articles)
    return [""] * len(articles)


def load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_checkpoint(checkpoint: dict):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)


def main():
    print(f"Reading {INPUT_FILE}...", flush=True)
    rows = []
    fieldnames = []
    with open(INPUT_FILE, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    checkpoint = load_checkpoint()
    print(f"Loaded checkpoint with {len(checkpoint)} cached entries.", flush=True)

    # Identify rows that need keywords
    todo_indices = [
        i for i, row in enumerate(rows)
        if not row.get("關鍵字", "").strip()
    ]
    print(f"Total rows: {len(rows)}, need keywords: {len(todo_indices)}", flush=True)

    processed = 0
    skipped_from_cache = 0

    # Apply cached results first
    pending_indices = []
    for idx in todo_indices:
        key = str(idx)
        if key in checkpoint:
            rows[idx]["關鍵字"] = checkpoint[key]
            skipped_from_cache += 1
        else:
            pending_indices.append(idx)

    print(f"From cache: {skipped_from_cache}, to call API: {len(pending_indices)}", flush=True)

    # Process in batches
    for batch_start in range(0, len(pending_indices), BATCH_SIZE):
        batch_indices = pending_indices[batch_start: batch_start + BATCH_SIZE]

        articles = []
        for idx in batch_indices:
            row = rows[idx]
            title = (row.get("英文篇名") or row.get("主要篇名") or "").strip()
            abstract = (row.get("英文摘要") or row.get("主要摘要") or "").strip()
            articles.append((title[:300], abstract[:1000]))

        keywords_list = call_api_batch(articles)

        for idx, keywords in zip(batch_indices, keywords_list):
            rows[idx]["關鍵字"] = keywords
            checkpoint[str(idx)] = keywords
            title_preview = articles[batch_indices.index(idx)][0][:55]
            print(f"[{idx}] {title_preview} → {keywords[:55]}", flush=True)

        processed += len(batch_indices)

        if processed % SAVE_EVERY < BATCH_SIZE:
            save_checkpoint(checkpoint)
            print(f"  Checkpoint saved ({processed} processed so far).", flush=True)

        time.sleep(SLEEP_BETWEEN)

    # Final checkpoint save
    save_checkpoint(checkpoint)

    # Write output CSV
    print(f"\nWriting {OUTPUT_FILE}...", flush=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Processed {processed} new, {skipped_from_cache} from cache.", flush=True)
    print(f"Output: {OUTPUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
