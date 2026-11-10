// build_sigm.cpp — SIGM（Simple Insertion Graph Merge）
//
// part1 と part2 のうち大きい方で HNSW を作り（土台）、そこへもう一方の点を1点ずつ
// 挿入する（Ponomarenko, arXiv:2505.16064 の Section 4.1）。挿入される点の層は新しく
// 抽選する。小さい方のグラフの構造は使わないので作らない。
//
// 部分データ（各 part の先頭 --limit 本）で動かし、土台づくりと後からの挿入の両方について、
// 挿入1回ごとの挙動をトレースに書く。トレースは visual/trace.html で再生できる。
//
// ビルド: g++ -std=c++17 -O3 -o src/build_sigm src/build_sigm.cpp   （clang++ でも可）
// 実行例: src/build_sigm --dataset SIFT --split random --limit 1000  （miyazaki フォルダで実行）
// 出力:   results/sub_<dataset>_<split>_sigm.csv、visual/data/trace_sigm.js
//
// 同じ --limit・--queries・--seed で hnsw_merge --method sigm を動かすと、同じグラフになる。

#include "hnsw_core.hpp"

int main(int argc, char** argv) {
    Options o = parse_args(argc, argv, "sigm");
    Experiment ex = setup(o);

    std::vector<int> a, b;
    for (int id : ex.order) ((size_t)id < ex.n1 ? a : b).push_back(id);
    const std::vector<int>& big = a.size() >= b.size() ? a : b;
    const std::vector<int>& small = a.size() >= b.size() ? b : a;
    const char* big_name = &big == &a ? "part1" : "part2";
    const char* small_name = &big == &a ? "part2" : "part1";
    std::printf("[sigm] 土台 %s %zu 点 (efC=%d) に %s %zu 点を挿入 (efC=%d)\n",
                big_name, big.size(), o.ef_construction, small_name, small.size(), o.sigm_ef);
    std::fflush(stdout);

    HNSW h(ex.data, o.M, o.ef_construction, o.seed + 2);  // hnsw_merge と同じ種
    Tracer tr;
    tr.search_detail = o.search_detail;
    if (o.write_trace) h.tracer = &tr;

    // 1. 土台：大きい方を空のグラフから作る
    tr.phase = 0;
    double t = now_sec();
    insert_all(h, big, "sigm:土台");
    double base_sec = now_sec() - t;
    uint64_t base_dist = h.n_dist;

    // 2. マージ：小さい方の点を、出来上がった土台に1点ずつ挿入する
    tr.phase = 1;
    h.ef_construction = o.sigm_ef;
    t = now_sec();
    insert_all(h, small, "sigm:挿入");
    double merge_sec = now_sec() - t;
    uint64_t merge_dist = h.n_dist - base_dist;
    std::printf("  土台 %.2f 秒 / 距離計算 %llu 回, マージ %.2f 秒 / 距離計算 %llu 回\n",
                base_sec, (unsigned long long)base_dist, merge_sec, (unsigned long long)merge_dist);

    std::vector<Phase> phases = {
        {std::string("土台（") + big_name + "）", 0, big.size(), o.ef_construction, base_dist, base_sec},
        {std::string("挿入（") + small_name + "）", big.size(), big.size() + small.size(), o.sigm_ef, merge_dist, merge_sec},
    };
    h.tracer = nullptr;
    GraphStats gs = graph_stats(h, ex.order.size());
    auto rows = evaluate(h, ex);
    write_csv(ex, phases, rows, gs);
    if (o.write_trace) write_trace(ex, tr, phases, rows, gs, trace_queries(h, ex));
    return 0;
}
