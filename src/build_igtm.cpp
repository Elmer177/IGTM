// build_igtm.cpp — IGTM（Iterative Graph Traversal Merge）
//
// 論文2（Ponomarenko, arXiv:2505.16064）の IGTM を、hnsw_core.hpp の上に実装したもの。
// 2つの HNSW グラフ（part1 / part2 で個別に構築）を1つにマージする。
//
// 考え方（論文の4ステップの繰り返しを、層ごとに実行する）:
//   ① 処理する頂点の選択: 同じ側のグラフの中で「前の頂点の近傍」から次の頂点を選ぶ（鎖）
//   ② 候補集め: 相手グラフでの探索。前の頂点の候補集合を使い回し、LocalSearch
//      （1層だけの探索）で済ませる。鎖の開始時と候補が空のときだけ、入口から
//      たどる HNSW-Search（フル探索）を行う
//   ③ 近傍の構築: 候補＋現在の近傍から select_neighbors（RNG ヒューリスティクス）
//      で選び直す
//   ④ 情報の伝播: 選ばれた相手には逆向きの辺も足し、次数オーバーは刈り込む。
//      次の頂点へは候補集合を持ち越す（= 使い回しそのもの）
//
// NGM との違いは ② で、NGM は毎回フル探索するが、IGTM は鎖の開始時だけにする。
//
// ビルド: g++ -std=c++17 -O3 -o src/build_igtm src/build_igtm.cpp   （clang++ でも可）
// 実行例: src/build_igtm --dataset SIFT --split random --limit 1000  （miyazaki フォルダで実行）
// 追加オプション:
//   --igtm-ef N   マージ時の探索 ef（候補集めの幅。既定 20。本多さんの NGM と同じ値）
// 出力: results/sub_<dataset>_<split>_igtm.csv、visual/data/trace_igtm.js
//
// 実装上の割り切り（論文の記述が曖昧な点）:
//   - 「行き詰まり」の判定は単純化し、LocalSearch の結果が空のときだけフル探索で
//     再スタートする。鎖は「未処理の近傍が尽きたら」終わる。
//   - LocalSearch は、前の候補集合を結果の初期値として引き継ぐ形にした
//     （論文の「候補集合の使い回し」をそのまま実装）。
//   - 距離計算の数え方: 土台2本の構築はマージに含めない（全体構築・SIGM と同じ流儀）。
//     マージ中の距離計算は、3つの HNSW オブジェクトのカウンタ増分＋このファイル内で
//     直接計算した分（manual_dist）を合計する。

#include "hnsw_core.hpp"

namespace {

// このファイル内で直接計算した距離の回数（HNSW::dist を通らない分）
uint64_t manual_dist = 0;

Scalar dist_counted(const Dataset& data, int a, int b) {
    manual_dist++;
    return dist2(data[a], data[b], data.dim);
}

int max_degree_of(int M, int layer) { return layer == 0 ? 2 * M : M; }

// 相手グラフ G' の層 l だけを対象にした HNSW-Search（入口から層を降りてくる）
std::vector<DistNode> hnsw_search_at_layer(const HNSW& g, const Scalar* q, int layer, int ef) {
    if (g.entry_point < 0 || g.max_level < layer) return {};
    int ep = g.entry_point;
    for (int l = g.max_level; l > layer; l--)
        ep = g.search_layer(q, ep, 1, l)[0].second;
    return g.search_layer(q, ep, ef, layer);
}

// LocalSearch: 前の頂点の候補集合を「結果の初期値」として引き継ぐ1層だけの探索。
// 前の頂点と近ければ、種の候補の多くがそのまま残り、少しの展開で終わる（= IGTM の節約の本体）。
// 距離は新しい頂点 v から測り直す。vis/token は訪問済みマーク（呼び出し側で使い回す）。
std::vector<DistNode> local_search_seeded(const HNSW& g, const Dataset& data, int v, int layer, int ef,
                                          const std::vector<DistNode>& seeds,
                                          std::vector<uint32_t>& vis, uint32_t& token) {
    if (++token == 0) { std::fill(vis.begin(), vis.end(), 0); token = 1; }
    std::priority_queue<DistNode, std::vector<DistNode>, std::greater<DistNode>> C;  // 近い順
    std::priority_queue<DistNode> W;                                                // 遠い順（結果 ef 件）
    for (const auto& [d0, c] : seeds) {
        (void)d0;  // 距離は前の頂点からのものなので使わない
        if (vis[c] == token) continue;
        vis[c] = token;
        Scalar d = dist_counted(data, v, c);
        C.push({d, c});
        W.push({d, c});
    }
    while (!C.empty()) {
        auto [d, u] = C.top();
        C.pop();
        if ((int)W.size() == ef && d > W.top().first) break;
        for (int w : g.graph[u][layer]) {
            if (vis[w] == token) continue;
            vis[w] = token;
            Scalar dw = dist_counted(data, v, w);
            if ((int)W.size() < ef || dw < W.top().first) {
                C.push({dw, w});
                W.push({dw, w});
                if ((int)W.size() > ef) W.pop();
            }
        }
    }
    std::vector<DistNode> out;
    while (!W.empty()) { out.push_back(W.top()); W.pop(); }
    std::sort(out.begin(), out.end());
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    // --igtm-ef だけ先に抜き取り、残りは共通の parse_args に渡す
    int igtm_ef = 20;
    std::vector<char*> args;
    args.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--igtm-ef") {
            if (i + 1 >= argc) { std::fprintf(stderr, "--igtm-ef に値がありません\n"); return 2; }
            igtm_ef = std::atoi(argv[++i]);
            if (igtm_ef <= 0) { std::fprintf(stderr, "--igtm-ef の値が不正です\n"); return 2; }
        } else {
            args.push_back(argv[i]);
        }
    }
    Options o = parse_args((int)args.size(), args.data(), "igtm");
    Experiment ex = setup(o);
    const Dataset& data = ex.data;

    // 挿入対象を part ごとに分ける（ノード番号 0..n1-1 が part1、残りが part2）
    std::vector<int> ids1, ids2;
    for (int id : ex.order) ((size_t)id < ex.n1 ? ids1 : ids2).push_back(id);
    const size_t total = ex.order.size();

    // 土台 A・B の決め方は SIGM・NGM と同じにする:
    //   A = 大きい方の part（種 seed+2、SIGM の土台・NGM の A と同じグラフになる）
    //   B = もう一方の part（種 seed+3、NGM の B と同じグラフになる）
    // こうすると、全手法がまったく同じ土台からマージを始める。
    const bool p1_big = ids1.size() >= ids2.size();
    const std::vector<int>& a_ids = p1_big ? ids1 : ids2;
    const std::vector<int>& b_ids = p1_big ? ids2 : ids1;
    const char* a_name = p1_big ? "part1" : "part2";
    const char* b_name = p1_big ? "part2" : "part1";
    std::vector<char> in_a(data.n, 0);  // その点が A 側か（B 側なら 0）
    for (int id : a_ids) in_a[id] = 1;

    // ---- 土台: 2本のグラフを個別に構築する（この分の距離計算はマージに含めない）----
    Tracer tr;
    tr.search_detail = o.search_detail;

    HNSW hA(data, o.M, o.ef_construction, o.seed + 2);
    HNSW hB(data, o.M, o.ef_construction, o.seed + 3);
    double t = now_sec();
    if (o.write_trace) hA.tracer = &tr;
    tr.phase = 0;
    insert_all(hA, a_ids, "igtm:土台A");
    if (o.write_trace) hB.tracer = &tr;
    tr.phase = 1;
    insert_all(hB, b_ids, "igtm:土台B");
    hA.tracer = hB.tracer = nullptr;
    double build_sec = now_sec() - t;
    uint64_t build_dist = hA.n_dist + hB.n_dist;
    std::printf("[igtm] 土台 A=%s %zu 点（距離計算 %llu 回）/ B=%s %zu 点（距離計算 %llu 回）を個別に構築（%.2f 秒）\n",
                a_name, a_ids.size(), (unsigned long long)hA.n_dist, b_name, b_ids.size(),
                (unsigned long long)hB.n_dist, build_sec);

    // ---- マージ後のグラフの器: 各点の層と自側の辺をコピーしたもの ----
    HNSW hm(data, o.M, o.ef_construction, o.seed + 5);
    for (int id : ex.order) {
        const HNSW& own = in_a[id] ? hA : hB;
        hm.levels[id] = own.levels[id];
        hm.graph[id] = own.graph[id];
    }
    if (hA.max_level >= hB.max_level) {
        hm.entry_point = hA.entry_point;
        hm.max_level = hA.max_level;
    } else {
        hm.entry_point = hB.entry_point;
        hm.max_level = hB.max_level;
    }

    // ---- マージ本体（層ごとに、上から下へ）----
    const uint64_t snapA = hA.n_dist, snapB = hB.n_dist, snapM = hm.n_dist;
    manual_dist = 0;
    size_t n_full = 0, n_local = 0, n_chains = 0;
    t = now_sec();

    std::vector<char> in_layer(data.n, 0), done(data.n, 0);
    std::vector<uint32_t> vis(data.n, 0);
    uint32_t vis_token = 0;
    for (int layer = hm.max_level; layer >= 0; layer--) {
        const int cap = max_degree_of(o.M, layer);
        size_t remaining = 0;
        for (int id : ex.order) {
            in_layer[id] = hm.levels[id] >= layer;
            done[id] = 0;
            if (in_layer[id]) remaining++;
        }

        int prev = -1;                    // 鎖の前の頂点（-1 = 鎖なし）
        int next_v = -1;                  // 鎖で次に処理する頂点
        std::vector<DistNode> prev_cand;  // 前の頂点の候補集合（使い回し用）

        while (remaining > 0) {
            int v;
            std::vector<DistNode> cand;
            if (prev < 0) {
                // 鎖の開始: 未処理の最初の頂点を選び、フル探索する
                v = -1;
                for (int id : ex.order)
                    if (in_layer[id] && !done[id]) { v = id; break; }
                const HNSW& other = in_a[v] ? hB : hA;
                cand = hnsw_search_at_layer(other, data[v], layer, igtm_ef);
                n_full++;
                n_chains++;
            } else {
                // 鎖の継続: 前の候補を使い回して LocalSearch する
                v = next_v;
                const HNSW& other = in_a[v] ? hB : hA;
                if (!prev_cand.empty() && other.max_level >= layer) {
                    cand = local_search_seeded(other, data, v, layer, igtm_ef, prev_cand, vis, vis_token);
                    n_local++;
                }
                if (cand.empty()) {  // 行き詰まり: フル探索で再スタート
                    cand = hnsw_search_at_layer(other, data[v], layer, igtm_ef);
                    n_full++;
                }
            }

            // ③ 近傍の構築: 相手側の候補＋現在の近傍から選び直す
            // cand 由来の距離は探索で計算済みなので使い回し、足りない分だけ計算する
            std::vector<DistNode> pool2;
            auto in_pool = [&](int u) {
                for (const auto& [d, w] : pool2)
                    if (w == u) return true;
                return false;
            };
            for (const auto& [d, u] : cand)
                if (u != v && !in_pool(u)) pool2.push_back({d, u});
            for (int u : hm.graph[v][layer])
                if (u != v && !in_pool(u)) pool2.push_back({dist_counted(data, v, u), u});
            std::vector<int> nbrs = hm.select_neighbors(pool2, cap);
            hm.graph[v][layer] = nbrs;

            // ④ 情報の伝播: 逆向きの辺を足し、次数オーバーは刈り込む
            for (int u : nbrs) {
                auto& adj = hm.graph[u][layer];
                if (std::find(adj.begin(), adj.end(), v) == adj.end()) {
                    adj.push_back(v);
                    if ((int)adj.size() > cap) {
                        std::vector<DistNode> c2;
                        for (int w : adj) c2.push_back({dist_counted(data, u, w), w});
                        adj = hm.select_neighbors(std::move(c2), cap);
                    }
                }
            }
            done[v] = 1;
            remaining--;

            // ① 次の頂点: 自側グラフの元の近傍のうち、未処理で最も近い点
            const HNSW& own = in_a[v] ? hA : hB;
            int nxt = -1;
            Scalar best = 0;
            for (int u : own.graph[v][layer]) {
                if (!in_layer[u] || done[u]) continue;
                Scalar d = dist_counted(data, v, u);
                if (nxt < 0 || d < best) { nxt = u; best = d; }
            }
            if (nxt < 0) {
                prev = -1;
                prev_cand.clear();
            } else {
                prev = v;
                prev_cand = cand;
                next_v = nxt;
            }
        }
    }

    double merge_sec = now_sec() - t;
    uint64_t merge_dist = (hA.n_dist - snapA) + (hB.n_dist - snapB) + (hm.n_dist - snapM) + manual_dist;
    std::printf("[igtm] マージ %.2f 秒, 距離計算 %llu 回（フル探索 %zu 回 / LocalSearch %zu 回 / 鎖 %zu 本, ef=%d）\n",
                merge_sec, (unsigned long long)merge_dist, n_full, n_local, n_chains, igtm_ef);

    // ---- 評価と出力（全体構築・SIGM と同じ形式）----
    std::vector<Phase> phases = {
        {"個別構築", 0, total, o.ef_construction, build_dist, build_sec},
        {"マージ", total, total, igtm_ef, merge_dist, merge_sec},
    };
    GraphStats gs = graph_stats(hm, total);
    auto rows = evaluate(hm, ex);
    write_csv(ex, phases, rows, gs);
    if (o.write_trace) write_trace(ex, tr, phases, rows, gs, trace_queries(hm, ex));
    return 0;
}
