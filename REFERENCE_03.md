# 05 關鍵字推薦作者 — 詳細說明

## 資料來源

程式會讀兩個 CSV：

- `articles_keywords.csv`：論文資料，且摘要、可 embedding 的關鍵字都不能是空值。
- `author_keywords.csv`：每位作者累積的關鍵字，用來建立作者推薦索引。

符合條件的 paper 必須同時滿足：

1. `articles_keywords.csv` 裡有 `文獻ID` 和 `主要篇名`。
2. `英文摘要` 不為空；若沒有英文摘要，程式會退回使用 `主要摘要`。
3. `英文關鍵字` 不為空；若沒有英文關鍵字，程式會退回使用 `關鍵字`。

預設隨機抽樣會從 `文獻ID` 以 `F03` 開頭的 paper 抽 40 篇。這個限制只影響被推薦審稿人的投稿論文，不限制候選審稿人的作者資料範圍。

## 基本使用

隨機抽 40 篇符合條件的 F03 paper，每篇推薦 10 位作者，輸出 CSV：

```bash
python3 05_recommend_authors.py
```

先檢查會抽到哪些 paper，但不產生 embedding：

```bash
python3 05_recommend_authors.py --dry-run --seed 42 --num-papers 40
```

指定某一篇 paper：

```bash
python3 05_recommend_authors.py --source-id F03N000029
```

或用 `主要篇名` 指定：

```bash
python3 05_recommend_authors.py --paper-title 台灣金融情勢與經濟預測
```

調整最後推薦幾位作者：

```bash
python3 05_recommend_authors.py --source-id F03N000029 --top-k 10
```

調整一次隨機跑幾篇 paper：

```bash
python3 05_recommend_authors.py --num-papers 40 --top-k 10
```

## Embedding Cache

作者關鍵字很多，所以程式會先把 `author_keywords.csv` 裡所有不重複的作者關鍵字各自做 embedding，並快取成兩個檔案：

- `author_keyword_embeddings_cache.json`：記錄模型名稱與 keyword 順序。
- `author_keyword_embeddings_cache.npy`：儲存 keyword embedding 矩陣。

只要模型和作者 keyword 清單沒有改變，之後執行會直接讀 cache，不會重算全部作者 keyword embedding。

paper 的 keyword 數量很少，所以每次執行時即時計算。
摘要不會用來推薦或篩選作者；摘要只會在作者推薦完成後，針對已推薦作者的文章即時計算相似度 evidence。

## 算分機制

採用 keyword 粒度的推薦：

1. paper 的每個 keyword 各自做 embedding。
2. `author_keywords.csv` 裡每個不重複的作者 keyword 也各自做 embedding。
3. 對每個 paper keyword，去所有作者 keyword 裡找最相近的前 `--keyword-top-k` 個。
4. 相似度低於 `--similarity-threshold` 的 author keyword 會被丟掉。
5. 命中的 author keyword 會回推到擁有該 keyword 的作者。
6. 作者分數由所有命中的 keyword similarity 加總，再依 `--score-mode` 聚合。

預設分數模式是 `soft_and`：

```text
score = similarity 加總 + cooccur_bonus * (命中的 paper keyword 數 - 1)
```

例如某作者被 3 個不同 paper keyword 命中，similarity 加總是 `2.28`，且 `--cooccur-bonus` 是 `0.2`：

```text
score = 2.28 + 0.2 * (3 - 1)
score = 2.68
```

這樣會鼓勵同時和多個 paper keyword 相關的作者，而不是只靠單一 keyword 很相似就排很前面。

最後排序依序看：

1. `score` 高者優先。
2. `source_keyword_hits` 多者優先，也就是命中的 paper keyword 數。
3. `exact_hits` 多者優先。
4. `similar_hits` 多者優先。
5. `article_count` 多者優先。
6. 作者名稱排序。

輸出 CSV 裡每位推薦作者會包含：

- `score`：keyword 階段的推薦分數。
- `keyword_score`：同 `score`，保留語意更清楚的欄位。
- `abstract_score`：推薦後摘要 evidence 的平均相似度，不參與作者推薦排序。
- `source_keyword_hits`：命中的 paper keyword 數。
- `exact_hits`：paper keyword 和 author keyword 完全相同的次數。
- `similar_hits`：語意相似但不完全相同的次數。
- `matched_keywords`：實際命中的 keyword pair 和 similarity。
- `author_keywords`：該作者在 `author_keywords.csv` 裡的全部關鍵字。

## 摘要 Evidence

作者推薦完成後，程式會再針對每位推薦作者找文章摘要 evidence：

1. 從 `articles_keywords.csv` 建立「作者 → 文章」索引。
2. 對每一篇目標 paper，先用 keyword 推薦出作者。
3. 只收集這些已推薦作者的文章摘要，不會預先計算全部 7728 篇摘要。
4. 計算目標 paper 摘要和該作者每篇文章摘要的 cosine similarity。
5. 每位作者最多保留 `--abstract-top-k` 篇文章，預設是 3。
6. similarity 低於 `--abstract-threshold` 的文章不保留，預設是 0.45。

摘要 evidence 不會改變推薦作者名單，也不會改變作者排序。它只會輸出在 CSV 的 `abstract_1_*`、`abstract_2_*`、`abstract_3_*` 欄位，方便你看這位作者有哪些文章和目標 paper 摘要最接近。

## 完整參數

| 參數 | 預設 | 說明 |
|------|------|------|
| `--top-k` | 10 | 每篇論文推薦幾位作者 |
| `--num-papers` | 40 | 隨機抽幾篇論文 |
| `--random-id-prefix` | F03 | 只影響預設隨機抽樣；設成空字串可從全部論文抽 |
| `--keyword-top-k` | 30 | 每個 paper keyword 保留幾個最相似的 author keyword |
| `--similarity-threshold` | 0.55 | keyword 相似度門檻 |
| `--score-mode` | soft_and | 分數聚合方式（soft_and / sum / avg / min） |
| `--cooccur-bonus` | 0.2 | soft_and 模式下命中多個 keyword 的加分 |
| `--abstract-top-k` | 3 | 每位作者保留幾篇摘要 evidence |
| `--abstract-threshold` | 0.45 | 摘要 evidence 相似度門檻 |
| `--abstract-batch-size` | 8 | 摘要 embedding 的 batch size |
| `--max-seq-length` | 512 | embedding model 最大 token 長度 |
| `--seed` | - | 固定隨機抽樣結果 |
| `--include-paper-authors` | - | 允許推薦論文原作者 |
| `--articles-file` | articles_keywords.csv | 論文資料檔案 |
| `--author-file` | author_keywords.csv | 作者關鍵字檔案 |
| `--output-file` | reviewer_candidates.csv | 輸出 CSV 路徑 |
| `--json-output-file` | - | 額外輸出 JSON（選填） |
| `--model` | BAAI/bge-m3 | sentence-transformers 模型 |

## 輸出欄位

CSV 每列代表一個 paper-author 推薦結果，主要欄位：

| 欄位 | 說明 |
|------|------|
| `paper_rank` | 論文編號 |
| `source_id` | 論文 ID |
| `paper_title` | 論文標題 |
| `paper_authors` | 論文作者 |
| `paper_keywords` | 論文關鍵字 |
| `recommended_rank` | 推薦排名 |
| `author` | 推薦作者中文名 |
| `english_name` | 推薦作者英文名 |
| `keyword_score` | keyword 階段分數 |
| `abstract_score` | 摘要 evidence 平均相似度 |
| `final_score` | 最終分數 |
| `source_keyword_hits` | 命中的 paper keyword 數 |
| `exact_hits` | 完全相同的 keyword 命中數 |
| `similar_hits` | 語意相似的 keyword 命中數 |
| `matched_keywords` | 命中的 keyword pair 和 similarity |
| `author_keywords` | 該作者的全部關鍵字 |
| `abstract_1_*` ~ `abstract_3_*` | 摘要 evidence 欄位 |


若還想另外輸出 JSON，可以加：

```bash
python3 05_recommend_authors.py --json-output-file result.json
```
