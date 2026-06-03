# Reviewer Recommendation

這個專案用 paper 的關鍵字推薦可能適合的審稿作者。主要程式是
`keyword_filter_auther.py`。

## 資料來源

程式會讀三個 CSV：

- `source_paper.csv`：用來篩選來源文獻 ID，以 `F03` 開頭的 paper 才會進入候選。
- `articles_updated.csv`：用 `主要篇名` 和 `source_paper.csv` 比對，且 `主要摘要`、`關鍵字` 都不能是空值。
- `author_keywords.csv`：每位作者累積的關鍵字，用來建立作者推薦索引。

符合條件的 paper 必須同時滿足：

1. `source_paper.csv` 的 `來源文獻ID` 以 `F03` 開頭。
2. `source_paper.csv` 的 `主要篇名` 能在 `articles_updated.csv` 找到同名文章。
3. `articles_updated.csv` 裡該文章的 `主要摘要` 不為空。
4. `articles_updated.csv` 裡該文章的 `關鍵字` 不為空。

## 安裝

```bash
python3 -m pip install -r requirement.txt
```

第一次正式執行時會下載 `all-mpnet-base-v2` 模型。

## 基本使用

隨機抽一篇符合條件的 F03 paper，推薦作者：

```bash
python3 keyword_filter_auther.py
```

先檢查會抽到哪一篇，但不產生 embedding：

```bash
python3 keyword_filter_auther.py --dry-run --seed 42
```

指定某一篇 paper：

```bash
python3 keyword_filter_auther.py --source-id F03N000029
```

或用 `主要篇名` 指定：

```bash
python3 keyword_filter_auther.py --paper-title 台灣金融情勢與經濟預測
```

調整最後推薦幾位作者：

```bash
python3 keyword_filter_auther.py --source-id F03N000029 --top-k 10
```

## Embedding Cache

作者關鍵字很多，所以程式會先把 `author_keywords.csv` 裡所有不重複的作者關鍵字各自做 embedding，並快取成兩個檔案：

- `author_keyword_embeddings_cache.json`：記錄模型名稱與 keyword 順序。
- `author_keyword_embeddings_cache.npy`：儲存 keyword embedding 矩陣。

只要模型和作者 keyword 清單沒有改變，之後執行會直接讀 cache，不會重算全部作者 keyword embedding。

paper 的 keyword 數量很少，所以每次執行時即時計算。

## 算分機制

目前不是把作者所有 keyword 串成一段文字後做一個 embedding。現在採用 keyword 粒度的推薦：

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

輸出 JSON 裡每位推薦作者會包含：

- `score`：最後推薦分數。
- `source_keyword_hits`：命中的 paper keyword 數。
- `exact_hits`：paper keyword 和 author keyword 完全相同的次數。
- `similar_hits`：語意相似但不完全相同的次數。
- `matched_keywords`：實際命中的 keyword pair 和 similarity。
- `author_keywords`：該作者在 `author_keywords.csv` 裡的全部關鍵字。

## 關鍵可調參數

`--top-k`

最後輸出幾位推薦作者。預設是 `20`。

```bash
python3 keyword_filter_auther.py --top-k 30
```

`--keyword-top-k`

每個 paper keyword 先保留幾個最相似的作者 keyword。預設是 `30`。

數字越大，候選作者越多，但可能比較雜，也會多花一點時間。

```bash
python3 keyword_filter_auther.py --keyword-top-k 50
```

`--similarity-threshold`

paper keyword 和 author keyword 的 cosine similarity 門檻。預設是 `0.55`。

數字越高，推薦越嚴格；數字越低，推薦越寬鬆。

```bash
python3 keyword_filter_auther.py --similarity-threshold 0.65
```

`--score-mode`

控制多個 paper keyword 的分數怎麼合併。可選：

- `soft_and`：預設。similarity 加總，並鼓勵命中多個 paper keyword 的作者。
- `sum`：只做 similarity 加總。
- `avg`：similarity 平均。
- `min`：取各 paper keyword 分數中的最小值，比較嚴格。

```bash
python3 keyword_filter_auther.py --score-mode sum
```

`--cooccur-bonus`

只在 `--score-mode soft_and` 時使用。預設是 `0.2`。

數字越大，越偏好同時命中多個 paper keyword 的作者。

```bash
python3 keyword_filter_auther.py --cooccur-bonus 0.3
```

`--include-paper-authors`

預設會排除該篇 paper 自己的作者，避免把原作者推薦成審稿者。若想允許推薦原作者，可以加這個參數：

```bash
python3 keyword_filter_auther.py --include-paper-authors
```

`--seed`

在沒有指定 `--source-id` 或 `--paper-title` 時，固定隨機抽樣結果，方便測試。

```bash
python3 keyword_filter_auther.py --seed 42
```

`--model`

指定 sentence-transformers embedding model。預設是 `all-mpnet-base-v2`。

```bash
python3 keyword_filter_auther.py --model all-mpnet-base-v2
```

如果換模型，作者 keyword embedding cache 會自動重建。

## 輸出

預設輸出：

```text
sample_author_recommendations.json
```

可以用 `--output-file` 改路徑：

```bash
python3 keyword_filter_auther.py --output-file result.json
```
