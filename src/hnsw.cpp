// HNSW の教育用・最小実装（C++17、外部ライブラリなし）
// hnsw.py をそのまま C++ に移したもの。アルゴリズム・乱数列・出力まで同一。
//
// 論文: Malkov & Yashunin, "Efficient and robust approximate nearest
// neighbor search using Hierarchical Navigable Small World graphs" (2020)
//
// 読み方:
//   - グラフ本体は graph[node][layer] = {隣のノード番号の集合}
//   - 中核は search_layer() の1関数。探索も構築もこれを層ごとに呼ぶだけ。
//
// ビルド: clang++ -std=c++17 -O2 -o hnsw hnsw.cpp
//   （-ffast-math は付けないこと。浮動小数の計算順が変わり Python と一致しなくなる）
// 実行:   ./hnsw          … recall を表示（Python 版と同じ出力）
//         ./hnsw --dump   … 層・グラフ・検索結果をすべて出力（Python 版との突き合わせ用）

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <queue>
#include <set>
#include <utility>
#include <vector>

using Vec = std::vector<double>;
using DistNode = std::pair<double, int>;  // Python の (距離, ノード) タプル

// ---------------------------------------------------------------
// Python の random.Random と全く同じ乱数列を出すクラス
//   - MT19937 を init_by_array で初期化（Python の Random(seed) と同じ）
//   - random()  : 53bit 精度の [0,1) 一様乱数（genrand_res53）
//   - gauss()   : Box-Muller 法、2個目の値を次回用にキャッシュ（Python と同じ）
// std::mt19937 は初期化方法が違うので自前実装している。
// ---------------------------------------------------------------
class PyRandom {
public:
    explicit PyRandom(uint32_t seed) {
        uint32_t key[1] = {seed};  // 32bit に収まる非負の seed は key=[seed]
        init_by_array(key, 1);
    }

    double random() {
        uint32_t a = genrand() >> 5, b = genrand() >> 6;
        return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
    }

    double gauss(double mu, double sigma) {
        double z;
        if (has_gauss_next_) {
            z = gauss_next_;
            has_gauss_next_ = false;
        } else {
            const double TWOPI = 2.0 * M_PI;
            double x2pi = random() * TWOPI;
            double g2rad = std::sqrt(-2.0 * std::log(1.0 - random()));
            z = std::cos(x2pi) * g2rad;
            gauss_next_ = std::sin(x2pi) * g2rad;
            has_gauss_next_ = true;
        }
        return mu + z * sigma;
    }

private:
    static constexpr int N = 624, M = 397;
    uint32_t mt_[N];
    int mti_ = N + 1;
    double gauss_next_ = 0.0;
    bool has_gauss_next_ = false;

    void init_genrand(uint32_t s) {
        mt_[0] = s;
        for (mti_ = 1; mti_ < N; mti_++)
            mt_[mti_] = 1812433253U * (mt_[mti_ - 1] ^ (mt_[mti_ - 1] >> 30)) + mti_;
    }

    void init_by_array(const uint32_t* key, int key_length) {
        init_genrand(19650218U);
        int i = 1, j = 0;
        for (int k = (N > key_length ? N : key_length); k; k--) {
            mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1664525U)) + key[j] + j;
            i++; j++;
            if (i >= N) { mt_[0] = mt_[N - 1]; i = 1; }
            if (j >= key_length) j = 0;
        }
        for (int k = N - 1; k; k--) {
            mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1566083941U)) - i;
            i++;
            if (i >= N) { mt_[0] = mt_[N - 1]; i = 1; }
        }
        mt_[0] = 0x80000000U;
    }

    uint32_t genrand() {
        static const uint32_t mag01[2] = {0x0U, 0x9908b0dfU};
        uint32_t y;
        if (mti_ >= N) {
            int kk;
            for (kk = 0; kk < N - M; kk++) {
                y = (mt_[kk] & 0x80000000U) | (mt_[kk + 1] & 0x7fffffffU);
                mt_[kk] = mt_[kk + M] ^ (y >> 1) ^ mag01[y & 1U];
            }
            for (; kk < N - 1; kk++) {
                y = (mt_[kk] & 0x80000000U) | (mt_[kk + 1] & 0x7fffffffU);
                mt_[kk] = mt_[kk + (M - N)] ^ (y >> 1) ^ mag01[y & 1U];
            }
            y = (mt_[N - 1] & 0x80000000U) | (mt_[0] & 0x7fffffffU);
            mt_[N - 1] = mt_[M - 1] ^ (y >> 1) ^ mag01[y & 1U];
            mti_ = 0;
        }
        y = mt_[mti_++];
        y ^= (y >> 11);
        y ^= (y << 7) & 0x9d2c5680U;
        y ^= (y << 15) & 0xefc60000U;
        y ^= (y >> 18);
        return y;
    }
};

// 2点間の距離の2乗（大小比較だけなら sqrt は不要なので省略）。
// Python の sum() と同じく先頭から順に足す。
double dist2(const Vec& a, const Vec& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); i++) {
        double d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

// 最小ヒープ（Python の heapq と同じく、タプルの小さい順に出てくる）
using MinHeap = std::priority_queue<DistNode, std::vector<DistNode>, std::greater<DistNode>>;

class HNSW {
public:
    int M;                    // 1ノードの最大次数（Layer 0 は 2*M）
    int ef_construction;
    double mL;                // レベル抽選の係数
    PyRandom rng;

    std::vector<Vec> vectors;                     // ノード番号 -> ベクトル
    std::vector<int> levels;                      // ノード番号 -> そのノードの最上層
    std::vector<std::vector<std::set<int>>> graph; // ノード番号 -> [層ごとの隣接集合]
    int entry_point = -1;     // 探索の入口（最上層にいるノード）。-1 は None の代わり
    int max_level = -1;

    explicit HNSW(int M_ = 16, int ef_construction_ = 100, uint32_t seed = 0)
        : M(M_), ef_construction(ef_construction_), mL(1.0 / std::log((double)M_)), rng(seed) {}

    // ---------------------------------------------------------------
    // 中核: 1つの層の中を探索する
    //   C = 次に調べる候補（近い順に出てくる最小ヒープ）
    //   W = いま保持している上位 ef 件（遠いものが先頭の最大ヒープを
    //       「距離をマイナスにして」最小ヒープで代用。Python 版と同じ）
    // ---------------------------------------------------------------
    std::vector<DistNode> search_layer(const Vec& q, int entry, int ef, int layer) const {
        std::set<int> visited = {entry};
        double d0 = dist2(q, vectors[entry]);
        MinHeap C;  // 候補キュー
        MinHeap W;  // 結果（距離の符号反転で最大ヒープ化）
        C.push({d0, entry});
        W.push({-d0, entry});

        while (!C.empty()) {
            auto [d, v] = C.top();
            C.pop();
            if (d > -W.top().first && (int)W.size() == ef)
                break;  // 候補の最良でも結果の最悪より遠い
            for (int u : graph[v][layer]) {
                if (visited.count(u)) continue;
                visited.insert(u);
                double du = dist2(q, vectors[u]);
                if ((int)W.size() < ef || du < -W.top().first) {
                    C.push({du, u});
                    W.push({-du, u});
                    if ((int)W.size() > ef)
                        W.pop();  // 一番遠いものを捨てて ef 件に保つ
                }
            }
        }
        // 結果を (距離, ノード) の近い順で返す
        std::vector<DistNode> result;
        result.reserve(W.size());
        while (!W.empty()) {
            result.push_back({-W.top().first, W.top().second});
            W.pop();
        }
        std::sort(result.begin(), result.end());
        return result;
    }

    // ---------------------------------------------------------------
    // 隣接点の選び方（ヒューリスティック）:
    //   近い順に見て「既に選んだどの点よりも q に近い」点だけ残す。
    //   → 辺が同じ方向に固まらず、方角がばらける。
    //   candidates は (距離, ノード) のリスト。
    // ---------------------------------------------------------------
    std::vector<int> select_neighbors(const Vec& q, std::vector<DistNode> candidates,
                                      int limit) const {
        std::vector<int> selected;
        std::sort(candidates.begin(), candidates.end());
        for (const auto& [_, c] : candidates) {
            bool ok = true;
            for (int r : selected) {
                if (!(dist2(q, vectors[c]) < dist2(vectors[c], vectors[r]))) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                selected.push_back(c);
                if ((int)selected.size() == limit) break;
            }
        }
        return selected;
    }

    // ---------------------------------------------------------------
    // 構築: 点を1つずつ入れる
    // ---------------------------------------------------------------
    int insert(const Vec& vec) {
        int node = (int)vectors.size();
        int level = random_level();
        vectors.push_back(vec);
        levels.push_back(level);
        graph.emplace_back(level + 1);

        if (entry_point == -1) {
            entry_point = node;
            max_level = level;
            return node;
        }

        int ep = entry_point;
        // 自分がいない上の層では、入口を自分の近くまで寄せるだけ（ef=1 の貪欲）
        for (int layer = max_level; layer > level; layer--)
            ep = search_layer(vec, ep, 1, layer)[0].second;

        // 自分がいる層では近傍を探し、双方向に辺を張る
        for (int layer = std::min(level, max_level); layer >= 0; layer--) {
            auto W = search_layer(vec, ep, ef_construction, layer);
            auto neighbors = select_neighbors(vec, W, M);
            graph[node][layer].insert(neighbors.begin(), neighbors.end());
            for (int nb : neighbors) {
                graph[nb][layer].insert(node);
                int max_deg = max_degree(layer);
                if ((int)graph[nb][layer].size() > max_deg) {
                    // 次数オーバーした相手側は、相手から見た候補で選び直す
                    std::vector<DistNode> cand;
                    for (int u : graph[nb][layer])
                        cand.push_back({dist2(vectors[nb], vectors[u]), u});
                    auto kept = select_neighbors(vectors[nb], cand, max_deg);
                    graph[nb][layer] = std::set<int>(kept.begin(), kept.end());
                }
            }
            ep = W[0].second;
        }

        if (level > max_level) {
            entry_point = node;
            max_level = level;
        }
        return node;
    }

    // ---------------------------------------------------------------
    // 探索: 上の層で大きく跳び、下の層（Layer 0）で候補 ef 件を持って詰める
    // ---------------------------------------------------------------
    std::vector<int> search(const Vec& q, int k = 10, int ef = 50) const {
        assert(entry_point != -1 && "まず insert してください");
        assert(ef >= k && "ef は k 以上にしてください");
        int ep = entry_point;
        for (int layer = max_level; layer > 0; layer--)
            ep = search_layer(q, ep, 1, layer)[0].second;
        auto W = search_layer(q, ep, ef, 0);
        std::vector<int> out;
        for (int i = 0; i < k && i < (int)W.size(); i++) out.push_back(W[i].second);
        return out;
    }

private:
    int random_level() {
        // ほとんどのノードは低い層までしか出ず、稀に高い層まで出る
        return (int)(-std::log(rng.random() + 1e-12) * mL);
    }

    int max_degree(int layer) const { return layer == 0 ? 2 * M : M; }
};

// ---------------------------------------------------------------
// 動作確認: 全探索（正確な答え）と比べて recall@k を測る
// ---------------------------------------------------------------
int main(int argc, char** argv) {
    bool dump = argc > 1 && std::strcmp(argv[1], "--dump") == 0;

    PyRandom rng(42);
    const int N = 500, DIM = 16, K = 10, QUERIES = 20;

    std::vector<Vec> data(N, Vec(DIM));
    for (auto& v : data)
        for (auto& x : v) x = rng.gauss(0, 1);

    HNSW index(16, 100, 1);
    for (const auto& v : data) index.insert(v);

    auto brute_force = [&](const Vec& q) {
        std::vector<DistNode> all;
        for (int i = 0; i < N; i++) all.push_back({dist2(q, data[i]), i});
        std::sort(all.begin(), all.end());
        std::vector<int> out;
        for (int i = 0; i < K; i++) out.push_back(all[i].second);
        return out;
    };

    if (dump) {
        std::printf("entry_point %d max_level %d\n", index.entry_point, index.max_level);
        for (int i = 0; i < N; i++) {
            std::printf("node %d level %d", i, index.levels[i]);
            for (auto& s : index.graph[i]) {
                std::printf(" |");
                for (int u : s) std::printf(" %d", u);
            }
            std::printf("\n");
        }
    }

    double total_recall = 0.0;
    for (int t = 0; t < QUERIES; t++) {
        Vec q(DIM);
        for (auto& x : q) x = rng.gauss(0, 1);
        auto approx_v = index.search(q, K, 50);
        auto exact_v = brute_force(q);
        if (dump) {
            std::printf("query %d:", t);
            for (int u : approx_v) std::printf(" %d", u);
            std::printf("\n");
        }
        std::set<int> approx(approx_v.begin(), approx_v.end());
        int hit = 0;
        for (int e : exact_v) hit += approx.count(e);
        total_recall += (double)hit / K;
    }
    std::printf("recall@%d = %.3f  (N=%d, 次元=%d)\n", K, total_recall / QUERIES, N, DIM);
    return 0;
}
