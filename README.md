# Reviewer Recommendation

這個專案從論文關鍵字、摘要、引用關係推薦適合的審稿人。

## Pipeline 概覽

```
01_generate_keywords.py          產生論文關鍵字
02_aggregate_author_keywords.py  彙整作者關鍵字
03_keyword_filter_auther.py      關鍵字 + 摘要相似度推薦作者候選人
04_select_reviewers.py           LLM 從候選人中選出最終審稿人
05_build_citation_graph.py       建立論文引用圖 (citation_graph.graphml)
06_build_author_citation_graph.py 建立作者引用圖 (author_citation_graph.graphml)
```

## 資料檔案

| 檔案 | 說明 |
|------|------|
| `source_paper.csv` | 來源論文，`來源文獻ID` 以 `F03` 開頭者為候選投稿論文 |
| `articles_keywords.csv` | 論文資料（標題、作者、摘要、關鍵字、`文獻ID`） |
| `author_keywords.csv` | 每位作者累積的關鍵字（由 `02` 產生） |
| `ref_pair.csv` | 引用對，每列代表一篇論文引用一篇經濟學門論文 |

## 安裝

```bash
python3 -m pip install -r requirement.txt
```

第一次執行 `03` 時會下載 `BAAI/bge-m3` 模型。

---

## 01 產生關鍵字

```bash
python3 01_generate_keywords.py
```

輸出更新後的 `articles_keywords.csv`。

---

## 02 彙整作者關鍵字

```bash
python3 02_aggregate_author_keywords.py
```

輸出 `author_keywords.csv`，每位作者對應其所有論文的關鍵字聯集。

---

## 03 關鍵字推薦作者

隨機抽 40 篇 F03 論文，每篇推薦 10 位作者：

```bash
python3 03_keyword_filter_auther.py
```

指定特定論文：

```bash
python3 03_keyword_filter_auther.py --source-id F03N000029
python3 03_keyword_filter_auther.py --paper-title 台灣金融情勢與經濟預測
```

預設輸出：`40X10_author_recommendations.csv`

算分邏輯、完整參數、輸出欄位說明請見 [REFERENCE_03.md](REFERENCE_03.md)。

---

## 04 LLM 選出最終審稿人

從 `03` 的輸出取前 10 名候選人，送給 LLM 選出最適合的 3 位：

```bash
python3 04_select_reviewers.py
```

輸出：`reviewer_recommendations.json`

---

## 05 建立論文引用圖

從 `ref_pair.csv` 建立無向引用圖：

```bash
python3 05_build_citation_graph.py
```

輸出：`citation_graph.graphml`

- **節點**：論文 ID，帶 `prefix` 屬性（字母 + 前兩位數字，例如 `F03`、`E06`、`T28`）
- **邊**：引用關係（一列 ref_pair = 一條邊）

查詢範例：

```python
import networkx as nx
G = nx.read_graphml("citation_graph.graphml")

# 找 5 hops 以內的相關論文
reachable = nx.single_source_shortest_path_length(G, "F030053", cutoff=5)
```

---

## 06 建立作者引用圖

從 `ref_pair.csv` + `articles_keywords.csv` 建立無向加權作者圖：

```bash
python3 06_build_author_citation_graph.py
```

輸出：`author_citation_graph.graphml`

- **節點**：作者（中文名優先；純英文名保留英文），帶 `en_name` 屬性
- **邊權重**：引用次數（A 引用 B + B 引用 A）+ 共著論文數

查詢範例：

```python
import networkx as nx
G = nx.read_graphml("author_citation_graph.graphml")

# 找與某作者關係最重的前 10 位
neighbors = sorted(G["林明仁"].items(), key=lambda x: -x[1]["weight"])
for node, data in neighbors[:10]:
    print(node, data["weight"])
```

節點 ID 對應 `author_keywords.csv` 的 `作者` 欄，可直接 join。
