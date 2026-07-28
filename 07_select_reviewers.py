#!/usr/bin/env python3
import argparse
import sys
import pandas as pd
import json
import os
import time
from openai import OpenAI
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

INPUT_FILE = "reviewer_candidates_reranked.csv"
OUTPUT_FILE = "reviewer_recommendations.json"
MODEL = "gpt-4o"


def build_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is missing. Put it in .env or export it before running 07_select_reviewers.py.")
    return OpenAI(api_key=api_key)


def graph_summary(cand):
    nearest_author = cand.get("nearest_paper_author", "")
    graph_distance = cand.get("author_graph_distance", "")
    graph_weight = cand.get("author_graph_weight", "")
    graph_note = str(cand.get("graph_relation_note", "") or "")

    if pd.isna(nearest_author) or not str(nearest_author).strip():
        return "作者圖未找到明確關係。"

    nearest_author = str(nearest_author).strip()
    if graph_note.startswith("direct_weight="):
        return f"與投稿作者「{nearest_author}」在作者圖中直接相連，關係權重 {graph_weight}。"
    if str(graph_distance) in {"2", "2.0"}:
        return f"與投稿作者「{nearest_author}」在作者圖中距離 2 hops。"
    if str(graph_distance) in {"3", "3.0"}:
        return f"與投稿作者「{nearest_author}」在作者圖中距離 3 hops。"
    if graph_note == "same_author":
        return f"與投稿作者「{nearest_author}」為同一作者，需排除或特別注意利益衝突。"
    return f"與投稿作者「{nearest_author}」的作者圖關係：{graph_note}。"


def build_prompt(paper, candidates):
    prompt = f"""請扮演資深學術評審。以下有一篇投稿論文，以及 {len(candidates)} 位候選審稿人的代表著作（編號 #1 到 #{len(candidates)}）。

【投稿論文】
標題：{paper['paper_title']}
作者：{paper['paper_authors']}
摘要：{paper['paper_abstract']}

【評分標準】：請根據以下兩項標準評估每位候選人的審稿適合度：
1. 研究相關性：候選人的著作主題與本論文研究方向的契合程度
2. 專業深度：候選人在相關領域是否具有持續且深入的研究積累

【候選審稿人】
"""
    for i, (_, cand) in enumerate(candidates.iterrows(), 1):
        prompt += f"\n#{i}：{cand['author']}（{cand['english_name']}）\n代表著作：\n"
        for j in range(1, 4):
            title = cand[f'abstract_{j}_title']
            abstract = cand[f'abstract_{j}_abstract']
            if pd.notna(title) and pd.notna(abstract):
                prompt += f"  - {title}：{str(abstract)[:250]}\n"
        if "combined_score" in cand.index:
            prompt += f"  圖譜資訊：{graph_summary(cand)}\n"

    prompt += f"""
【輸出格式】
請選出最適合的 3 位審稿人，依適合程度由高到低排列。
每位的理由請寫 3-4 句話（約 50 字），須具體指出該審稿人的哪些研究成果與本論文的哪些核心議題或方法直接相關，並點名其代表著作中哪一篇與本論文最為契合。
禁止使用「適合審稿」、「適合擔任審稿人」、「因此推薦」等空洞的結尾套語。

請嚴格以以下 JSON 格式回應，name 必須使用候選人的中文姓名：
{{
  "reviewers": [
    {{"name": "中文姓名", "reason": "推薦理由"}},
    {{"name": "中文姓名", "reason": "推薦理由"}},
    {{"name": "中文姓名", "reason": "推薦理由"}}
  ]
}}"""
    return prompt


def select_reviewers(client, model, paper, candidates):
    prompt = build_prompt(paper, candidates)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.3,
    )

    result = json.loads(response.choices[0].message.content)
    return result.get("reviewers", [])


def main():
    parser = argparse.ArgumentParser(
        description="Use an OpenAI model to select final reviewers from reranked candidates."
    )
    parser.add_argument("input_file", nargs="?", default=INPUT_FILE)
    parser.add_argument("output_file", nargs="?", default=OUTPUT_FILE)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    client = build_client()
    df = pd.read_csv(args.input_file)
    score_column = "combined_score" if "combined_score" in df.columns else "final_score"

    output = {"papers": []}

    papers = df.groupby("source_id", sort=False)
    total_papers = len(papers)

    for source_id, paper_group in papers:
        paper_info = paper_group.iloc[0]

        # 取 graph rerank 後的前 10 名送給 LLM；沒有 combined_score 時退回 final_score
        top10 = paper_group.nlargest(10, score_column)

        print(f"[{len(output['papers']) + 1}/{total_papers}] 處理：{paper_info['paper_title'][:30]}...")

        try:
            reviewers_raw = select_reviewers(client, args.model, paper_info, top10)
        except Exception as e:
            print(f"  !! API 錯誤：{e}")
            reviewers_raw = []

        # 對應回原始資料取得 english_name 與 final_score
        reviewer_details = []
        for rev in reviewers_raw:
            name = rev.get("name", "")
            match = top10[top10["author"] == name]
            if not match.empty:
                row = match.iloc[0]
                reviewer_details.append({
                    "name": row["author"],
                    "english_name": row["english_name"],
                    "final_score": round(float(row[score_column]), 4),
                    "reason": rev.get("reason", ""),
                })
            else:
                # LLM 偶爾會回傳英文名或略有出入，做模糊比對
                fuzzy = top10[top10["english_name"].str.contains(name, case=False, na=False)]
                if not fuzzy.empty:
                    row = fuzzy.iloc[0]
                    reviewer_details.append({
                        "name": row["author"],
                        "english_name": row["english_name"],
                        "final_score": round(float(row[score_column]), 4),
                        "reason": rev.get("reason", ""),
                    })
                else:
                    reviewer_details.append({
                        "name": name,
                        "english_name": "",
                        "final_score": None,
                        "reason": rev.get("reason", ""),
                    })

        paper_record = {
            "source_id": paper_info["source_id"],
            "paper_rank": int(paper_info["paper_rank"]),
            "paper_title": paper_info["paper_title"],
            "paper_authors": paper_info["paper_authors"],
            "paper_abstract": paper_info["paper_abstract"],
            "paper_keywords": paper_info["paper_keywords"],
            "recommended_reviewers": reviewer_details,
        }

        output["papers"].append(paper_record)
        print(f"  -> 選出：{[r['name'] for r in reviewer_details]}")

        time.sleep(0.5)  # 避免打太快

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n完成！結果已儲存至 {args.output_file}")


if __name__ == "__main__":
    main()
