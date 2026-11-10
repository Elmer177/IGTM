# HNSW グラフのマージ実験

2つの HNSW グラフをマージする手法を、全データから作ったグラフと比べる実験です。今は全体構築と SIGM（Simple Insertion Graph Merge）を実装しています。NGM・IGTM・CGTM は今後追加します。SIGM は、大きい方のグラフにもう一方の点を1点ずつ挿入する方法です。

参考: Ponomarenko, [Three Algorithms for Merging Hierarchical Navigable Small World Graphs](https://arxiv.org/abs/2505.16064) (2025)

## 構成

| パス | 内容 |
|---|---|
| `src/hnsw_core.hpp` | HNSW 本体、データ読み込み、評価、CSV とトレースの書き出し |
| `src/build_full.cpp` | 全体構築：part1 ∪ part2 を空のグラフから挿入する |
| `src/build_sigm.cpp` | SIGM |
| `src/hnsw.py`, `src/hnsw.cpp` | 教育用の最小実装（両者の出力は完全に一致） |
| `visual/trace.html` | 挿入・検索の様子を1ステップずつ再生するビジュアライザ |
| `results/` | 実験結果の CSV |

## データ

SIFT1M（128 次元）と GIST1M（960 次元）を、それぞれ2通りに 50 万本ずつ分けたもの（`random`：一様ランダム、`gauss`：分布に偏りあり）を使います。データは再配布しないのでリポジトリには入っていません。`kenpro 2/<random|gauss>/<SIFT|GIST>/` に置いてください。

## 使い方

リポジトリ直下で実行します。

```sh
clang++ -std=c++17 -O3 -o src/build_full src/build_full.cpp
clang++ -std=c++17 -O3 -o src/build_sigm src/build_sigm.cpp

src/build_full --dataset SIFT --split random --limit 1000
src/build_sigm --dataset SIFT --split random --limit 1000
```

`--limit N` で各 part の先頭 N 本だけを使います。結果は `results/sub_*.csv` に、トレースは `visual/data/trace_*.js` に出力されます。`visual/trace.html` をブラウザで開くと再生できます。全オプションは `--help` で表示されます。

## 実験条件

既定値は論文に合わせています。

- M=16（Layer 0 は 32）
- efConstruction=32
- recall@5、L = 32, 40, 50, 64, 72

クエリはデータから取り除いた点を使い、正解は全件探索で求めます。比較の指標は、マージ中の距離計算の回数と、マージ後のグラフの recall です。
