// build_ngm.cpp — NGM（Naive Graph Merge）
//
// part1 と part2 のそれぞれで HNSW を作り（土台 A・B）、両方の辺を使ってマージする
// （Ponomarenko, arXiv:2505.16064）。マージ後のグラフ H の点と層は A・B のものを引き継ぎ、
// 各層 l の各点 v について次のように近傍を作り直す。
//   1. 相手のグラフを最上層から l 層まで検索し、v に近い点を最大 mmax(l) 件集める
//   2. 候補 =（自分のグラフでの v の元の近傍）∪（1 で集めた点）
//   3. 候補からヒューリスティック（select_neighbors）で mmax(l) 本を選び、H での v の近傍にする
// A を処理してから B を処理し、それを Layer 0 から上の層へ順に行う。
// 検索するのは元の A・B なので、H を書き換えても次の点の検索結果は変わらない。
//
// A は SIGM の土台と同じく大きい方の part で、乱数の種も同じ（seed+2）。同じ設定なら
// SIGM と NGM はまったく同じグラフ A からマージを始める。
// 挿入をしないので、トレース（visual/trace.html）は書かない。
//
// ビルド: clang++ -std=c++17 -O3 -o src/build_ngm src/build_ngm.cpp
// 実行例: src/build_ngm --dataset SIFT --split random --limit 1000  （miyazaki フォルダで実行）
// 出力:   results/sub_<dataset>_<split>_ngm.csv

#include "hnsw_core.hpp"

struct NgmCost { uint64_t search = 0, rebuild = 0; };  // 相手のグラフの検索 / 近傍の作り直し

// A・B をマージしたグラフを返す。a_ids・b_ids は A・B に挿入した順の点。
static HNSW ngm_merge(const Dataset& data, const HNSW& A, const HNSW& B, const std::vector<int>& a_ids,
                      const std::vector<int>& b_ids, int ef, NgmCost& cost) {
    // マージ後のグラフの骨組み：点と層の高さは元のグラフのものを引き継ぐ
    HNSW H(data, A.M, A.ef_construction, 0);
    for (const auto* ids : {&a_ids, &b_ids})
        for (int v : *ids) {
            int lv = (ids == &a_ids ? A : B).levels[v];
            H.levels[v] = lv;
            H.graph[v].resize(lv + 1);
        }
    H.max_level = std::max(A.max_level, B.max_level);
    H.entry_point = A.max_level >= B.max_level ? A.entry_point : B.entry_point;

    uint64_t search_before = A.n_dist + B.n_dist;
    std::vector<DistNode> cand;
    for (int l = 0; l <= H.max_level; l++) {
        const int mmax = l == 0 ? 2 * H.M : H.M;
        for (int side = 0; side < 2; side++) {
            const HNSW& S = side == 0 ? A : B;  // v が元々いるグラフ
            const HNSW& O = side == 0 ? B : A;  // 相手のグラフ
            for (int v : side == 0 ? a_ids : b_ids) {
                if (S.levels[v] < l) continue;
                // 1. 相手のグラフで v に近い点を探す（上の層でも ef を使って探す）
                std::vector<int> found = search_to_layer(O, data[v], mmax, ef, l, ef);
                // 2. 候補 = 元の近傍 ∪ 見つけた点（重複と v 自身は除く）
                std::vector<int> ids = S.graph[v][l];
                ids.insert(ids.end(), found.begin(), found.end());
                std::sort(ids.begin(), ids.end());
                ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
                cand.clear();
                for (int u : ids)
                    if (u != v) cand.push_back({dist2(data[v], data[u], data.dim), u});
                H.n_dist += cand.size();  // 候補との距離も数える
                // 3. ヒューリスティックで選び直す
                H.graph[v][l] = H.select_neighbors(cand, mmax);
            }
        }
    }
    cost.search = A.n_dist + B.n_dist - search_before;
    cost.rebuild = H.n_dist;
    return H;
}

int main(int argc, char** argv) {
    Options o = parse_args(argc, argv, "ngm");
    Experiment ex = setup(o);

    std::vector<int> p1, p2;
    for (int id : ex.order) ((size_t)id < ex.n1 ? p1 : p2).push_back(id);
    const bool p1_big = p1.size() >= p2.size();
    const std::vector<int>& a_ids = p1_big ? p1 : p2;
    const std::vector<int>& b_ids = p1_big ? p2 : p1;
    const char* a_name = p1_big ? "part1" : "part2";
    const char* b_name = p1_big ? "part2" : "part1";
    std::printf("[ngm] 土台 A=%s %zu 点・B=%s %zu 点 (efC=%d) をマージ (探索 ef=%d)\n",
                a_name, a_ids.size(), b_name, b_ids.size(), o.ef_construction, o.ngm_ef);
    std::fflush(stdout);

    // 1. 土台：A と B をそれぞれ空のグラフから作る
    HNSW A(ex.data, o.M, o.ef_construction, o.seed + 2);  // SIGM の土台と同じ種
    HNSW B(ex.data, o.M, o.ef_construction, o.seed + 3);
    double t = now_sec();
    insert_all(A, a_ids, "ngm:土台A");
    double a_sec = now_sec() - t;
    t = now_sec();
    insert_all(B, b_ids, "ngm:土台B");
    double b_sec = now_sec() - t;
    uint64_t a_dist = A.n_dist, b_dist = B.n_dist;

    // 2. マージ：全点の近傍を、相手のグラフを検索して作り直す
    NgmCost cost;
    t = now_sec();
    HNSW H = ngm_merge(ex.data, A, B, a_ids, b_ids, o.ngm_ef, cost);
    double merge_sec = now_sec() - t;
    uint64_t merge_dist = cost.search + cost.rebuild;
    std::printf("  土台 A %.2f 秒 / 距離計算 %llu 回, 土台 B %.2f 秒 / 距離計算 %llu 回\n", a_sec,
                (unsigned long long)a_dist, b_sec, (unsigned long long)b_dist);
    std::printf("  マージ %.2f 秒 / 距離計算 %llu 回（相手のグラフの検索 %llu ＋ 近傍の作り直し %llu）\n",
                merge_sec, (unsigned long long)merge_dist, (unsigned long long)cost.search,
                (unsigned long long)cost.rebuild);

    size_t n = a_ids.size() + b_ids.size();
    std::vector<Phase> phases = {
        {std::string("土台A（") + a_name + "）", 0, a_ids.size(), o.ef_construction, a_dist, a_sec},
        {std::string("土台B（") + b_name + "）", a_ids.size(), n, o.ef_construction, b_dist, b_sec},
        {"マージ", 0, n, o.ngm_ef, merge_dist, merge_sec},
    };
    GraphStats gs = graph_stats(H, ex.order.size());
    auto rows = evaluate(H, ex);
    write_csv(ex, phases, rows, gs);
    return 0;
}
