// build_full.cpp — 全体構築：part1 ∪ part2 の全点を、空のグラフから1点ずつ挿入する
//
// マージ手法と比べるための基準。部分データ（各 part の先頭 --limit 本）で動かし、
// 挿入1回ごとの挙動（どの層をどう探索し、どの近傍を選び、誰の辺を刈り込んだか）を
// トレースに書く。トレースは visual/trace.html で再生できる。
//
// ビルド: g++ -std=c++17 -O3 -o src/build_full src/build_full.cpp   （clang++ でも可）
// 実行例: src/build_full --dataset SIFT --split random --limit 1000  （miyazaki フォルダで実行）
// 出力:   results/sub_<dataset>_<split>_full.csv、visual/data/trace_full.js
//
// 同じ --limit・--queries・--seed で hnsw_merge --method full を動かすと、同じグラフになる。

#include "hnsw_core.hpp"

int main(int argc, char** argv) {
    Options o = parse_args(argc, argv, "full");
    Experiment ex = setup(o);

    std::printf("[full] 全 %zu 点を挿入 (M=%d, efC=%d)\n", ex.order.size(), o.M, o.ef_construction);
    std::fflush(stdout);
    HNSW h(ex.data, o.M, o.ef_construction, o.seed + 1);  // hnsw_merge と同じ種
    Tracer tr;
    tr.search_detail = o.search_detail;
    if (o.write_trace) h.tracer = &tr;

    double t = now_sec();
    insert_all(h, ex.order, "full");
    double sec = now_sec() - t;
    std::printf("  構築 %.2f 秒, 距離計算 %llu 回\n", sec, (unsigned long long)h.n_dist);

    std::vector<Phase> phases = {{"全体構築", 0, ex.order.size(), o.ef_construction, h.n_dist, sec}};
    h.tracer = nullptr;
    GraphStats gs = graph_stats(h, ex.order.size());
    auto rows = evaluate(h, ex);
    write_csv(ex, phases, rows, gs);
    if (o.write_trace) write_trace(ex, tr, phases, rows, gs, trace_queries(h, ex));
    return 0;
}
