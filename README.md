# Reviewer Recommendation

用論文關鍵字、摘要、引用關係和作者關係圖，推薦適合的審稿人。

## 安裝

```bash
python3 -m pip install -r requirement.txt
```

如果要用 repo 裡已經存在的 virtualenv：

```bash
source myvenv/bin/activate
```

會呼叫 OpenAI API 的腳本需要 `.env`：

```bash
OPENAI_API_KEY=你的 API key
```

第一次執行 `05_recommend_authors.py` 會下載 `BAAI/bge-m3` embedding 模型。

## Pipeline 順序

照這個順序跑：

```bash
python3 01_generate_keywords.py
python3 02_aggregate_author_keywords.py
python3 03_build_citation_graph.py
python3 04_build_author_citation_graph.py
python3 05_recommend_authors.py
python3 06_rerank_with_author_graph.py
python3 07_select_reviewers.py
```

每一步做的事：

| 步驟 | 腳本 | 做什麼 | 主要輸出 |
|------|------|--------|----------|
| 01 | `01_generate_keywords.py` | 幫缺關鍵字的論文用 OpenAI 補關鍵字 | `articles_keywords.csv` |
| 02 | `02_aggregate_author_keywords.py` | 把每位作者文章的中英文關鍵字彙整起來 | `author_keywords.csv` |
| 03 | `03_build_citation_graph.py` | 用引用資料建立論文引用圖 | `citation_graph.graphml` |
| 04 | `04_build_author_citation_graph.py` | 用引用與共著關係建立作者圖 | `author_citation_graph.graphml` |
| 05 | `05_recommend_authors.py` | 用 keyword/abstract embedding 推薦前 20 位候選作者 | `reviewer_candidates.csv` |
| 06 | `06_rerank_with_author_graph.py` | 用作者圖重新排序候選人，並保留前 10 位 | `reviewer_candidates_reranked.csv` |
| 07 | `07_select_reviewers.py` | 讓 LLM 從前 10 名候選人選出 3 位並寫理由 | `reviewer_recommendations.json` |

如果 `articles_keywords.csv` 已經有關鍵字，可以跳過 `01`，從 `02` 開始。

## 05 推薦邏輯

`05_recommend_authors.py` 主要用投稿論文關鍵字和作者累積關鍵字的 embedding similarity 推薦候選審稿人：

- 投稿論文預設從 `文獻ID` 以 `F03` 開頭的文章中隨機抽 40 篇。
- 每篇投稿論文的英文關鍵字會先做 embedding；若英文關鍵字為空，才退回使用中文關鍵字。
- 每個投稿關鍵字會去比對所有作者累積關鍵字，預設先保留最相近的 30 個作者關鍵字，且 similarity 必須大於等於 `0.55`。
- 作者如果擁有被命中的相近關鍵字，就累加該 similarity；預設 `soft_and` 會再給「命中多個投稿關鍵字」的作者共現加分。
- `keyword_score` 是 05 的主要排序分數；目前 `final_score` 等於 `keyword_score`。
- 05 預設先保留每篇前 20 位候選作者，之後交給 06 用作者圖重排。

05 也會替每位候選作者找代表著作證據：把投稿論文摘要和該候選作者過去文章的摘要做 embedding similarity，取最相近的前三篇寫進 `abstract_1_*`、`abstract_2_*`、`abstract_3_*` 欄位。這些 `abstract_*` 欄位目前不參與 05 排名，主要是提供給 `07_select_reviewers.py`，讓 LLM 可以根據候選人的具體著作選出最後 3 位並寫理由。

## 三種推薦方式

### 1. 隨機抽 40 篇 F03 論文

這是 `05_recommend_authors.py` 的預設模式：抽 40 篇 `文獻ID` 以 `F03` 開頭的投稿論文，每篇先推薦 20 位候選人；`06_rerank_with_author_graph.py` 會重排後保留前 10 位。

注意：`F03` 只限制「被抽出來要推薦審稿人的投稿論文」。候選審稿人的作者資料仍來自完整的 `author_keywords.csv`，不限制在 `F03`。

```bash
python3 05_recommend_authors.py
python3 06_rerank_with_author_graph.py
python3 07_select_reviewers.py
```

先檢查會抽到哪些文章，不產生 embedding：

```bash
python3 05_recommend_authors.py --dry-run --seed 42
```

### 2. 選某位作者的文章

```bash
python3 05_recommend_authors.py --author 林明仁  --output-file lin_candidates.csv
python3 06_rerank_with_author_graph.py lin_candidates.csv lin_candidates_reranked.csv
python3 07_select_reviewers.py lin_candidates_reranked.csv lin_reviewers.json
```

`--author` 可用中文名或英文名；這個模式不套用預設的 `F03` 隨機抽樣限制。

### 3. 直接指定某篇文章

用文獻 ID：

```bash
python3 05_recommend_authors.py --source-id F030156 --output-file one_candidates.csv
python3 06_rerank_with_author_graph.py one_candidates.csv one_candidates_reranked.csv
python3 07_select_reviewers.py one_candidates_reranked.csv one_reviewers.json
```

或用完整篇名：

```bash
python3 05_recommend_authors.py --paper-title 台灣金融情勢與經濟預測
```

## 常用參數

| 參數 | 預設 | 說明 |
|------|------|------|
| `--num-papers` | `40` | 隨機抽幾篇論文 |
| `--top-k` | `20` | `05_recommend_authors.py` 每篇論文先保留幾位候選審稿人 |
| `--random-id-prefix` | `F03` | 只影響預設隨機抽樣；用 `--random-id-prefix ""` 可從全部論文隨機抽 |
| `--author` | 無 | 只從指定作者的文章中抽樣 |
| `--source-id` | 無 | 直接指定單篇文獻 ID |
| `--paper-title` | 無 | 直接指定單篇完整篇名 |
| `--seed` | 無 | 固定隨機抽樣結果 |
| `--dry-run` | 關閉 | 只印出會處理哪些論文，不產生結果 |

`06_rerank_with_author_graph.py` 也有 `--top-k`，預設為 `10`，代表作者圖重排後每篇論文只輸出前 10 位候選審稿人。

## 資料檔案

| 檔案 | 說明 |
|------|------|
| `articles_keywords.csv` | 論文資料，含標題、作者、摘要、關鍵字、`文獻ID` |
| `author_keywords.csv` | 每位作者累積的中英文關鍵字，由 `02` 產生 |
| `ref_pair.csv` | 引用對，一列代表一組引用關係 |
| `source_paper.csv` | 原始來源論文清單，目前 pipeline 不直接讀這個檔 |
