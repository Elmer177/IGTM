// hnsw_core.hpp — build_full.cpp と build_sigm.cpp が共有する部分
//
//   - 乱数（Python の random.Random と同じ列）、データ読み込み、距離
//   - HNSW 本体（hnsw_merge.cpp の class HNSW と同じアルゴリズム）
//   - 挿入・検索の「挙動」を記録するトレース（visual/trace.html で再生する）
//   - 部分データの準備（--limit）、クエリと正解、評価、CSV とトレースの書き出し
//
// HNSW の辺の決まり方は hnsw_merge.cpp と同じ。同じ --limit・--queries・--seed なら
// グラフと距離計算回数まで一致する。トレースの記録は距離計算をしないので、回数は変わらない。
// アルゴリズムを変えるときは hnsw_merge.cpp と visual/index.html も合わせること。

#pragma once

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <functional>
#include <numeric>
#include <queue>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifndef SCALAR
#define SCALAR float
#endif
using Scalar = SCALAR;
using DistNode = std::pair<Scalar, int>;  // (距離, ノード)
using MinHeap = std::priority_queue<DistNode, std::vector<DistNode>, std::greater<DistNode>>;

[[noreturn]] static void die(const std::string& msg) {
    std::fprintf(stderr, "エラー: %s\n", msg.c_str());
    std::exit(1);
}

static double now_sec() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------
// Python の random.Random と同じ乱数列
// ---------------------------------------------------------------
class PyRandom {
public:
    explicit PyRandom(uint32_t seed) {
        uint32_t key[1] = {seed};
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

// ---------------------------------------------------------------
// データ: n 本の dim 次元ベクトルを1本の配列に並べて持つ
// ---------------------------------------------------------------
struct Dataset {
    size_t n = 0, dim = 0;
    std::vector<Scalar> x;
    const Scalar* operator[](size_t i) const { return x.data() + i * dim; }
};

// 2点間の距離の2乗。16 レーンに分けて足すと、-ffast-math なしでも SIMD 化される。
static inline Scalar dist2(const Scalar* a, const Scalar* b, size_t n) {
    constexpr size_t L = 16;
    Scalar lane[L] = {};
    size_t i = 0;
    for (; i + L <= n; i += L)
        for (size_t j = 0; j < L; j++) {
            Scalar d = a[i + j] - b[i + j];
            lane[j] += d * d;
        }
    Scalar s = 0;
    for (size_t j = 0; j < L; j++) s += lane[j];
    for (; i < n; i++) {
        Scalar d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

// 「[n] [d]\n ベクトル...」形式のテキストを読み、ds の末尾に追加する。読んだ本数を返す。
// 形式が違う（本数・次元の不一致、数値以外の文字など）ときは黙って進めず止める。
static size_t load_text(const std::string& path, size_t limit, Dataset& ds) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) die("開けません: " + path);
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size == 0) die("読めません: " + path);
    size_t size = (size_t)st.st_size;
    void* map = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) die("mmap に失敗: " + path);
    madvise(map, size, MADV_SEQUENTIAL);
    const char* p = static_cast<const char*>(map);
    const char* end = p + size;

    static const double POW10[] = {1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9,
                                   1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18};
    size_t row = 0;  // エラーメッセージ用（0 はヘッダ行）
    auto fail = [&](const char* what) {
        die(path + " の " + std::to_string(row + 1) + " 行目: " + what);
    };
    auto skip_blank = [&] { while (p < end && (*p == ' ' || *p == '\t' || *p == '\r')) p++; };
    auto expect_eol = [&] {
        skip_blank();
        if (p < end) {
            if (*p != '\n') fail("行末に余分な値があります");
            p++;
        }
    };
    auto parse_number = [&]() -> double {
        skip_blank();
        uint64_t mant = 0;
        int digits = 0, frac = 0;
        while (p < end && *p >= '0' && *p <= '9') { mant = mant * 10 + (uint64_t)(*p++ - '0'); digits++; }
        if (p < end && *p == '.') {
            p++;
            while (p < end && *p >= '0' && *p <= '9') { mant = mant * 10 + (uint64_t)(*p++ - '0'); digits++; frac++; }
        }
        if (digits == 0) fail("数値が足りないか、数値以外の文字があります");
        if (digits > 18) fail("桁数が多すぎます");
        return (double)mant / POW10[frac];
    };

    size_t n = (size_t)parse_number(), dim = (size_t)parse_number();
    expect_eol();
    if (n == 0 || dim == 0) fail("ヘッダが不正です");
    if (ds.dim != 0 && ds.dim != dim) fail("もう一方のファイルと次元が違います");
    ds.dim = dim;
    size_t take = std::min(n, limit);
    ds.x.resize((ds.n + take) * dim);
    Scalar* out = ds.x.data() + ds.n * dim;
    for (row = 1; row <= take; row++) {
        for (size_t j = 0; j < dim; j++) *out++ = (Scalar)parse_number();
        expect_eol();
    }
    if (take == n) {  // 全部読んだときはファイルが丁度終わることも確認する
        row = n;
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) p++;
        if (p != end) fail("ヘッダの本数より多くのデータがあります");
    }
    ds.n += take;
    munmap(map, size);
    close(fd);
    return take;
}

// ---------------------------------------------------------------
// トレース: 1つの層での探索と、その層で決まった辺の記録
// ---------------------------------------------------------------
struct LayerTrace {
    int layer = 0;
    bool greedy = false;                 // true: ef=1 で降りるだけの層（辺は張らない）
    std::vector<int> expanded;           // 候補から取り出して隣を調べたノード（順番どおり）
    std::vector<int> evaluated;          // 距離を計算したノード。(どこから, どれを) の組を平らに並べる。入口は from=-1
    std::vector<int> cand;               // 探索結果 W（近い順）。辺を張る層だけ
    std::vector<int> nbrs;               // ヒューリスティックで選んだ近傍。辺を張る層だけ
    std::vector<std::pair<int, std::vector<int>>> pruned;  // 次数オーバーで選び直した相手と、選び直した後の隣接リスト
};

struct InsertTrace {
    int node = -1, level = 0, phase = 0;
    uint64_t dist = 0;                   // この挿入での距離計算回数
    bool new_entry = false;              // この点が新しい入口になったか
    std::vector<LayerTrace> layers;      // 上の層から順に
};

struct QueryTrace {
    int query = -1;                      // クエリ番号（Experiment::query_ids の添字）
    uint64_t dist = 0;
    std::vector<int> result;
    std::vector<LayerTrace> layers;
};

struct Tracer {
    bool search_detail = true;           // false なら expanded / evaluated を記録しない（ファイルを小さくする）
    int phase = 0;
    std::vector<InsertTrace> inserts;
};

// ---------------------------------------------------------------
// HNSW 本体（hnsw_merge.cpp の class HNSW と同じ処理に、記録用のフックを足したもの）
// ---------------------------------------------------------------
class HNSW {
public:
    const Dataset& data;
    int M;                    // 1ノードの最大次数（Layer 0 は 2*M）
    int ef_construction;
    double mL;                // レベル抽選の係数
    PyRandom rng;

    std::vector<int> levels;                        // ノード番号 -> 最上層（未挿入は -1）
    std::vector<std::vector<std::vector<int>>> graph;  // ノード番号 -> [層ごとの隣接リスト]
    int entry_point = -1;     // 探索の入口（最上層にいるノード）
    int max_level = -1;
    mutable uint64_t n_dist = 0;  // 距離計算の回数（論文の評価指標）
    Tracer* tracer = nullptr;     // 設定すると insert の挙動を記録する

    HNSW(const Dataset& data_, int M_, int ef_construction_, uint32_t seed)
        : data(data_), M(M_), ef_construction(ef_construction_),
          mL(1.0 / std::log((double)M_)), rng(seed),
          levels(data_.n, -1), graph(data_.n), visited_(data_.n, 0) {}

    // 中核: 1つの層の中を探索する（C = 候補の最小ヒープ、W = 距離の符号反転で最大ヒープ化）
    // lt を渡すと、取り出したノードと距離を計算したノードを記録する
    std::vector<DistNode> search_layer(const Scalar* q, int entry, int ef, int layer,
                                       LayerTrace* lt = nullptr) const {
        uint32_t mark = next_mark();
        visited_[entry] = mark;
        Scalar d0 = dist(q, data[entry]);
        if (lt) { lt->evaluated.push_back(-1); lt->evaluated.push_back(entry); }
        MinHeap C, W;
        C.push({d0, entry});
        W.push({-d0, entry});

        while (!C.empty()) {
            auto [d, v] = C.top();
            C.pop();
            if (d > -W.top().first && (int)W.size() == ef)
                break;  // 候補の最良でも結果の最悪より遠い
            if (lt) lt->expanded.push_back(v);
            for (int u : graph[v][layer]) {
                if (visited_[u] == mark) continue;
                visited_[u] = mark;
                Scalar du = dist(q, data[u]);
                if (lt) { lt->evaluated.push_back(v); lt->evaluated.push_back(u); }
                if ((int)W.size() < ef || du < -W.top().first) {
                    C.push({du, u});
                    W.push({-du, u});
                    if ((int)W.size() > ef)
                        W.pop();  // 一番遠いものを捨てて ef 件に保つ
                }
            }
        }
        std::vector<DistNode> result;
        result.reserve(W.size());
        while (!W.empty()) {
            result.push_back({-W.top().first, W.top().second});
            W.pop();
        }
        std::sort(result.begin(), result.end());
        return result;
    }

    // 隣接点の選び方（ヒューリスティック）: 近い順に見て「既に選んだどの点よりも
    // 基準点に近い」点だけ残す。candidates は (基準点からの距離, ノード)。
    std::vector<int> select_neighbors(std::vector<DistNode> candidates, int limit) const {
        std::vector<int> selected;
        std::sort(candidates.begin(), candidates.end());
        for (const auto& [dc, c] : candidates) {
            bool ok = true;
            for (int r : selected) {
                if (!(dc < dist(data[c], data[r]))) {
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

    // 構築: 点を1つ入れる
    void insert(int node) {
        InsertTrace* it = nullptr;
        uint64_t dist_before = n_dist;
        if (tracer) {
            tracer->inserts.emplace_back();
            it = &tracer->inserts.back();
            it->node = node;
            it->phase = tracer->phase;
        }
        auto new_layer = [&](int layer, bool greedy) -> LayerTrace* {
            if (!it) return nullptr;
            it->layers.emplace_back();
            LayerTrace* lt = &it->layers.back();
            lt->layer = layer;
            lt->greedy = greedy;
            return tracer->search_detail ? lt : nullptr;
        };
        auto layer_rec = [&]() -> LayerTrace* { return it ? &it->layers.back() : nullptr; };

        const Scalar* vec = data[node];
        int level = random_level();
        levels[node] = level;
        graph[node].resize(level + 1);
        if (it) it->level = level;

        if (entry_point == -1) {
            entry_point = node;
            max_level = level;
            if (it) it->new_entry = true;
            return;
        }

        int ep = entry_point;
        // 自分がいない上の層では、入口を自分の近くまで寄せるだけ（ef=1 の貪欲）
        for (int layer = max_level; layer > level; layer--) {
            LayerTrace* lt = new_layer(layer, true);
            ep = search_layer(vec, ep, 1, layer, lt)[0].second;
            if (it) layer_rec()->cand = {ep};
        }

        // 自分がいる層では近傍を探し、双方向に辺を張る
        for (int layer = std::min(level, max_level); layer >= 0; layer--) {
            LayerTrace* lt = new_layer(layer, false);
            auto W = search_layer(vec, ep, ef_construction, layer, lt);
            auto neighbors = select_neighbors(W, M);
            graph[node][layer] = neighbors;
            LayerTrace* rec = layer_rec();
            if (rec) {
                for (const auto& w : W) rec->cand.push_back(w.second);
                rec->nbrs = neighbors;
            }
            int max_deg = max_degree(layer);
            for (int nb : neighbors) {
                auto& adj = graph[nb][layer];
                adj.push_back(node);
                if ((int)adj.size() > max_deg) {
                    // 次数オーバーした相手側は、相手から見た候補で選び直す
                    std::vector<DistNode> cand;
                    cand.reserve(adj.size());
                    for (int u : adj) cand.push_back({dist(data[nb], data[u]), u});
                    adj = select_neighbors(std::move(cand), max_deg);
                    if (rec) rec->pruned.push_back({nb, adj});
                }
            }
            ep = W[0].second;
        }

        if (level > max_level) {
            entry_point = node;
            max_level = level;
            if (it) it->new_entry = true;
        }
        if (it) it->dist = n_dist - dist_before;
    }

    // 探索: 上の層で大きく跳び、Layer 0 で候補 ef 件を持って詰める
    // qt を渡すと層ごとの探索を記録する
    std::vector<int> search(const Scalar* q, int k, int ef, QueryTrace* qt = nullptr) const {
        uint64_t before = n_dist;
        auto rec = [&](int layer, bool greedy) -> LayerTrace* {
            if (!qt) return nullptr;
            qt->layers.emplace_back();
            qt->layers.back().layer = layer;
            qt->layers.back().greedy = greedy;
            return &qt->layers.back();
        };
        int ep = entry_point;
        for (int layer = max_level; layer > 0; layer--) {
            LayerTrace* lt = rec(layer, true);
            ep = search_layer(q, ep, 1, layer, lt)[0].second;
            if (lt) lt->cand = {ep};
        }
        LayerTrace* lt = rec(0, false);
        auto W = search_layer(q, ep, ef, 0, lt);
        std::vector<int> out;
        out.reserve(k);
        for (int i = 0; i < k && i < (int)W.size(); i++) out.push_back(W[i].second);
        if (qt) {
            for (const auto& w : W) lt->cand.push_back(w.second);
            qt->result = out;
            qt->dist = n_dist - before;
        }
        return out;
    }

private:
    mutable std::vector<uint32_t> visited_;  // ノード -> 最後に訪問した探索の世代番号
    mutable uint32_t mark_ = 0;

    uint32_t next_mark() const {
        if (++mark_ == 0) {  // 一周したら配列を消してやり直す
            std::fill(visited_.begin(), visited_.end(), 0);
            mark_ = 1;
        }
        return mark_;
    }

    Scalar dist(const Scalar* a, const Scalar* b) const {
        n_dist++;
        return dist2(a, b, data.dim);
    }

    int random_level() {
        // ほとんどのノードは低い層までしか出ず、稀に高い層まで出る
        return (int)(-std::log(rng.random() + 1e-12) * mL);
    }

    int max_degree(int layer) const { return layer == 0 ? 2 * M : M; }
};

// ---------------------------------------------------------------
// 実験の設定（build_full と build_sigm で共通）
// ---------------------------------------------------------------
struct Options {
    std::string method;  // "full" または "sigm"（プログラムが決める）
    std::string data_dir = "kenpro 2", dataset = "SIFT", split = "random", out, trace;
    std::string runs = "results/sub_runs.csv";  // 実行ごとに追記する一覧
    int M = 16, ef_construction = 32, sigm_ef = 32, k = 5, queries = 100, query_traces = 20;
    uint32_t seed = 1;
    size_t limit = 1000;          // 各ファイルの先頭から読む本数
    bool search_detail = true;
    bool write_trace = true;
    std::vector<int> Ls = {32, 40, 50, 64, 72};
};

static void usage(const std::string& method) {
    std::fprintf(stderr,
        "使い方: src/build_%s [オプション]   （miyazaki フォルダで実行）\n"
        "  --dataset SIFT|GIST      データセット（既定 SIFT）\n"
        "  --split random|gauss     分け方（既定 random）\n"
        "  --limit N                part1・part2 それぞれ先頭 N 本を使う（既定 1000）\n"
        "  --queries N              データから取り除いてクエリにする本数（既定 100）\n"
        "  --M N                    最大次数（既定 16、Layer 0 は 2M）\n"
        "  --efc N                  構築時の ef（既定 32）\n"
        "%s"
        "  --k N                    recall@k の k（既定 5）\n"
        "  --L a,b,c                検索時の ef（既定 32,40,50,64,72）\n"
        "  --seed N                 乱数の種（既定 1。hnsw_merge と同じ値なら同じグラフになる）\n"
        "  --data DIR               データのフォルダ（既定 \"kenpro 2\"）\n"
        "  --out PATH               この実行の CSV（既定 results/sub_<dataset>_<split>_%s.csv、毎回上書き）\n"
        "  --runs PATH              全実行の一覧 CSV（既定 results/sub_runs.csv、毎回1行ずつ追記）\n"
        "  --trace PATH             トレースの出力先（既定 visual/data/trace_%s.js）\n"
        "  --query-traces N         検索の様子を記録するクエリの数（既定 20）\n"
        "  --no-search-detail       挿入時の探索経路を記録しない（トレースが小さくなる）\n"
        "  --no-trace               トレースを書かない\n",
        method.c_str(),
        method == "sigm" ? "  --sigm-efc N             SIGM で後から挿入するときの ef（既定 32）\n" : "",
        method.c_str(), method.c_str());
    std::exit(2);
}

static Options parse_args(int argc, char** argv, const std::string& method) {
    Options o;
    o.method = method;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s に値がありません\n", a.c_str()); usage(method); }
            return argv[++i];
        };
        auto ival = [&]() -> long long {
            std::string s = val();
            char* e = nullptr;
            long long v = std::strtoll(s.c_str(), &e, 10);
            if (s.empty() || *e || v <= 0) { std::fprintf(stderr, "%s の値が不正です: %s\n", a.c_str(), s.c_str()); usage(method); }
            return v;
        };
        if (a == "--dataset") o.dataset = val();
        else if (a == "--split") o.split = val();
        else if (a == "--data") o.data_dir = val();
        else if (a == "--M") o.M = (int)ival();
        else if (a == "--efc") o.ef_construction = (int)ival();
        else if (a == "--sigm-efc" && method == "sigm") o.sigm_ef = (int)ival();
        else if (a == "--k") o.k = (int)ival();
        else if (a == "--queries") o.queries = (int)ival();
        else if (a == "--query-traces") o.query_traces = (int)ival();
        else if (a == "--seed") o.seed = (uint32_t)ival();
        else if (a == "--limit") o.limit = (size_t)ival();
        else if (a == "--out") o.out = val();
        else if (a == "--runs") o.runs = val();
        else if (a == "--trace") o.trace = val();
        else if (a == "--no-search-detail") o.search_detail = false;
        else if (a == "--no-trace") o.write_trace = false;
        else if (a == "--L") {
            o.Ls.clear();
            std::string s = val();
            size_t pos = 0;
            while (pos <= s.size()) {
                size_t c = s.find(',', pos);
                if (c == std::string::npos) c = s.size();
                int v = std::atoi(s.substr(pos, c - pos).c_str());
                if (v <= 0) { std::fprintf(stderr, "--L の値が不正です: %s\n", s.c_str()); usage(method); }
                o.Ls.push_back(v);
                pos = c + 1;
            }
        } else usage(method);
    }
    if ((o.dataset != "SIFT" && o.dataset != "GIST") || (o.split != "random" && o.split != "gauss")) usage(method);
    if (o.M < 2) die("--M は 2 以上にしてください");  // log(1)=0 で mL が発散する
    for (int L : o.Ls)
        if (L < o.k) die("L は k 以上にしてください（L=" + std::to_string(L) + ", k=" + std::to_string(o.k) + "）");
    if (o.limit > 20000 && o.write_trace && o.search_detail)
        std::fprintf(stderr, "注意: --limit %zu だとトレースが大きくなります。--no-search-detail か --no-trace を検討してください\n", o.limit);
    if (o.out.empty()) o.out = "results/sub_" + o.dataset + "_" + o.split + "_" + method + ".csv";
    if (o.trace.empty()) o.trace = "visual/data/trace_" + method + ".js";
    return o;
}

// ---------------------------------------------------------------
// 部分データの準備（hnsw_merge.cpp の main と同じ手順・同じ乱数）
//   1. part1・part2 のファイルを、それぞれ先頭から --limit 本だけ読む（load_text の limit）。
//      ランダムに選ぶのではなく「ファイルの先頭 N 行」。ノード番号は part1 が 0..n1-1、part2 が n1..
//   2. 読んだ全点を --seed の乱数でシャッフルする（これが挿入順になる）
//   3. シャッフル後の先頭 --queries 本をクエリにして、挿入する点からは取り除く
//   4. 残りの点に対して、各クエリの正解 k 件を全件探索で求める
// ---------------------------------------------------------------
struct Experiment {
    Options o;
    Dataset data;
    size_t n1 = 0, n2 = 0;               // ノード番号 0..n1-1 が part1、n1.. が part2
    std::vector<int> order;              // 挿入する点（シャッフル済み、クエリは除く）
    std::vector<int> query_ids;          // クエリにした点のノード番号
    Dataset queries;
    std::vector<std::vector<int>> gt;    // 各クエリの正解 k 件
};

static void ground_truth(Experiment& ex) {
    const Dataset& data = ex.data;
    ex.gt.assign(ex.queries.n, {});
    unsigned nt = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < nt; t++)
        pool.emplace_back([&, t] {
            std::vector<DistNode> all(ex.order.size());
            for (size_t q = t; q < ex.queries.n; q += nt) {
                for (size_t i = 0; i < ex.order.size(); i++)
                    all[i] = {dist2(ex.queries[q], data[ex.order[i]], data.dim), ex.order[i]};
                std::partial_sort(all.begin(), all.begin() + ex.o.k, all.end());
                for (int i = 0; i < ex.o.k; i++) ex.gt[q].push_back(all[i].second);
            }
        });
    for (auto& th : pool) th.join();
}

static Experiment setup(const Options& o) {
    Experiment ex;
    ex.o = o;
    std::string lower = o.dataset == "SIFT" ? "sift" : "gist";
    std::string base = o.data_dir + "/" + o.split + "/" + o.dataset + "/" + lower + "1m_part";
    double t0 = now_sec();
    ex.n1 = load_text(base + "1.txt", o.limit, ex.data);
    ex.n2 = load_text(base + "2.txt", o.limit, ex.data);
    std::printf("%s/%s: part1 %zu 本 + part2 %zu 本, %zu 次元（読み込み %.1f 秒）\n",
                o.dataset.c_str(), o.split.c_str(), ex.n1, ex.n2, ex.data.dim, now_sec() - t0);

    ex.order.resize(ex.data.n);
    std::iota(ex.order.begin(), ex.order.end(), 0);
    PyRandom shuffle_rng(o.seed);
    for (size_t i = 0; i + 1 < ex.order.size(); i++) {
        size_t j = i + (size_t)(shuffle_rng.random() * (ex.order.size() - i));
        std::swap(ex.order[i], ex.order[j]);
    }
    if ((size_t)o.queries >= ex.data.n) die("--queries がデータ数以上です");
    ex.queries.dim = ex.data.dim;
    ex.queries.n = o.queries;
    ex.queries.x.resize(ex.queries.n * ex.queries.dim);
    for (size_t i = 0; i < ex.queries.n; i++) {
        ex.query_ids.push_back(ex.order[i]);
        std::memcpy(ex.queries.x.data() + i * ex.queries.dim, ex.data[ex.order[i]], sizeof(Scalar) * ex.data.dim);
    }
    ex.order.erase(ex.order.begin(), ex.order.begin() + o.queries);
    if (ex.order.size() < (size_t)o.k) die("データ数が k より少なくなっています");
    ground_truth(ex);
    std::printf("クエリ %zu 本、挿入対象 %zu 点\n", ex.queries.n, ex.order.size());
    std::fflush(stdout);
    return ex;
}

static void insert_all(HNSW& h, const std::vector<int>& ids, const char* label) {
    double t0 = now_sec();
    size_t step = std::max<size_t>(1, ids.size() / 5);
    for (size_t i = 0; i < ids.size(); i++) {
        h.insert(ids[i]);
        if ((i + 1) % step == 0)
            std::fprintf(stderr, "  [%s] %zu/%zu 点 %.1f 秒\n", label, i + 1, ids.size(), now_sec() - t0);
    }
}

// ---------------------------------------------------------------
// 評価とグラフの健全性
// ---------------------------------------------------------------
struct EvalRow { int L; double recall, qps, dpq; };
struct GraphStats { int layers = 0; double avg_deg0 = 0; size_t reached = 0, n = 0; std::vector<size_t> per_layer; };

static GraphStats graph_stats(const HNSW& h, size_t n_indexed) {
    GraphStats s;
    s.layers = h.max_level + 1;
    s.n = n_indexed;
    s.per_layer.assign(s.layers, 0);
    uint64_t edges = 0;
    for (size_t v = 0; v < h.graph.size(); v++) {
        const auto& g = h.graph[v];
        if (g.empty()) continue;
        edges += g[0].size();
        for (int l = 0; l < (int)g.size() && l < s.layers; l++) s.per_layer[l]++;
    }
    std::vector<char> seen(h.graph.size(), 0);
    std::vector<int> stack = {h.entry_point};
    seen[h.entry_point] = 1;
    while (!stack.empty()) {
        int v = stack.back();
        stack.pop_back();
        s.reached++;
        for (int u : h.graph[v][0])
            if (!seen[u]) { seen[u] = 1; stack.push_back(u); }
    }
    s.avg_deg0 = (double)edges / n_indexed;
    std::printf("  層数 %d, Layer 0 の平均次数 %.2f, 入口から到達可能 %zu/%zu 点\n",
                s.layers, s.avg_deg0, s.reached, n_indexed);
    return s;
}

static std::vector<EvalRow> evaluate(const HNSW& h, const Experiment& ex) {
    std::vector<EvalRow> rows;
    for (int L : ex.o.Ls) {
        uint64_t before = h.n_dist;
        size_t hits = 0;
        double t = now_sec();
        std::vector<std::vector<int>> found(ex.queries.n);
        for (size_t q = 0; q < ex.queries.n; q++) found[q] = h.search(ex.queries[q], ex.o.k, L);
        double sec = std::max(1e-9, now_sec() - t);
        for (size_t q = 0; q < ex.queries.n; q++)
            for (int id : found[q])
                hits += std::count(ex.gt[q].begin(), ex.gt[q].end(), id);
        EvalRow r{L, (double)hits / (ex.queries.n * ex.o.k), ex.queries.n / sec,
                  (double)(h.n_dist - before) / ex.queries.n};
        std::printf("  L=%-4d recall@%d=%.4f  QPS=%.0f  距離計算/クエリ=%.0f\n", L, ex.o.k, r.recall, r.qps, r.dpq);
        rows.push_back(r);
    }
    std::fflush(stdout);
    return rows;
}

// 最終グラフでの検索の様子を、先頭のいくつかのクエリについて記録する（L は最大のもの）
static std::vector<QueryTrace> trace_queries(const HNSW& h, const Experiment& ex) {
    std::vector<QueryTrace> out;
    int L = *std::max_element(ex.o.Ls.begin(), ex.o.Ls.end());
    size_t nq = std::min<size_t>(ex.queries.n, (size_t)ex.o.query_traces);
    uint64_t saved = h.n_dist;  // 評価の回数に混ざらないよう戻す
    for (size_t q = 0; q < nq; q++) {
        QueryTrace qt;
        qt.query = (int)q;
        h.search(ex.queries[q], ex.o.k, L, &qt);
        out.push_back(std::move(qt));
    }
    h.n_dist = saved;
    return out;
}

// ---------------------------------------------------------------
// 書き出し: CSV（1行 = 1つの L）とトレース（visual/trace.html が読む）
// ---------------------------------------------------------------
struct Phase { std::string name; size_t start = 0, end = 0; int efc = 0; uint64_t dist = 0; double sec = 0; };

// 1行 = 1つの L。パラメータ・グラフの状態・実行時間・評価指標をすべて並べる。
// --out はこの実行だけ（上書き）、--runs は全実行の一覧（追記）に同じ行を書く。
static const char* CSV_HEADER =
    "time,dataset,split,method,limit,n,dim,n_query,M,efc,sigm_efc,seed,"
    "layers,avg_deg0,reached,base_points,inserted_points,base_dist,base_sec,merge_dist,merge_sec,"
    "total_dist,total_sec,k,L,recall,qps,dist_per_query\n";

static void write_csv(const Experiment& ex, const std::vector<Phase>& phases, const std::vector<EvalRow>& rows,
                      const GraphStats& gs) {
    const Options& o = ex.o;
    const Phase& b = phases.front();
    const Phase none{};
    const Phase& m = phases.size() > 1 ? phases.back() : none;  // SIGM の「挿入」段階
    char stamp[32];
    std::time_t now = std::time(nullptr);
    std::strftime(stamp, sizeof stamp, "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    auto emit = [&](FILE* f) {
        for (const auto& r : rows)
            std::fprintf(f, "%s,%s,%s,%s,%zu,%zu,%zu,%zu,%d,%d,%d,%u,%d,%.4f,%zu,%zu,%zu,%llu,%.4f,%llu,%.4f,%llu,%.4f,%d,%d,%.5f,%.1f,%.1f\n",
                         stamp, o.dataset.c_str(), o.split.c_str(), o.method.c_str(), o.limit, ex.order.size(),
                         ex.data.dim, ex.queries.n, o.M, o.ef_construction, o.method == "sigm" ? o.sigm_ef : 0, o.seed,
                         gs.layers, gs.avg_deg0, gs.reached, b.end - b.start, m.end - m.start,
                         (unsigned long long)b.dist, b.sec, (unsigned long long)m.dist, m.sec,
                         (unsigned long long)(b.dist + m.dist), b.sec + m.sec, o.k, r.L, r.recall, r.qps, r.dpq);
    };
    auto open_for = [](const std::string& path, const char* mode) {
        std::filesystem::path p(path);
        if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
        FILE* f = std::fopen(path.c_str(), mode);
        if (!f) die("書き込めません: " + path);
        return f;
    };
    FILE* f = open_for(o.out, "w");
    std::fputs(CSV_HEADER, f);
    emit(f);
    std::fclose(f);

    // 一覧は追記。新しいファイルか、列が古い形式のときだけ見出しを書く
    bool need_header = true;
    if (FILE* r = std::fopen(o.runs.c_str(), "r")) {
        char line[512] = {};
        if (std::fgets(line, sizeof line, r)) need_header = std::string(line) != CSV_HEADER;
        std::fclose(r);
        if (need_header) die(o.runs + " の列が今の形式と違います。別名に移すか --runs で別のファイルを指定してください");
    }
    f = open_for(o.runs, "a");
    if (need_header) std::fputs(CSV_HEADER, f);
    emit(f);
    std::fclose(f);

    // 画面にもまとめを出す
    std::printf("\n== まとめ（%s, %s/%s, 各 part 先頭 %zu 本, M=%d, efC=%d%s）==\n", o.method.c_str(),
                o.dataset.c_str(), o.split.c_str(), o.limit, o.M, o.ef_construction,
                o.method == "sigm" ? (", 挿入 efC=" + std::to_string(o.sigm_ef)).c_str() : "");
    for (const auto& p : phases)
        std::printf("  %-16s %6zu 点  %8.3f 秒  距離計算 %llu 回\n", p.name.c_str(), p.end - p.start, p.sec,
                    (unsigned long long)p.dist);
    std::printf("  %-6s %-10s %-10s %s\n", "L", "recall", "QPS", "距離計算/クエリ");
    for (const auto& r : rows) std::printf("  %-6d %-10.4f %-10.0f %.0f\n", r.L, r.recall, r.qps, r.dpq);
    std::printf("CSV: %s（今回）, %s（一覧に追記）\n", o.out.c_str(), o.runs.c_str());
}

// 主成分分析で2次元に写す（べき乗法。表示用なので距離計算の回数には数えない）
static std::vector<float> pca2(const Dataset& d) {
    size_t n = d.n, dim = d.dim;
    std::vector<double> mean(dim, 0.0);
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < dim; j++) mean[j] += d[i][j];
    for (auto& m : mean) m /= (double)n;
    auto cov_mul = [&](const std::vector<double>& v) {  // (X^T X) v を X を作らずに計算
        std::vector<double> out(dim, 0.0);
        for (size_t i = 0; i < n; i++) {
            const Scalar* x = d[i];
            double s = 0;
            for (size_t j = 0; j < dim; j++) s += (x[j] - mean[j]) * v[j];
            for (size_t j = 0; j < dim; j++) out[j] += s * (x[j] - mean[j]);
        }
        return out;
    };
    auto normalize = [](std::vector<double>& v) {
        double s = 0;
        for (double x : v) s += x * x;
        s = std::sqrt(s);
        if (s > 0) for (double& x : v) x /= s;
    };
    std::vector<std::vector<double>> pcs;
    for (int k = 0; k < 2; k++) {
        std::vector<double> v(dim);
        for (size_t j = 0; j < dim; j++) v[j] = 1.0 + 0.01 * (double)((j * 7 + k * 13) % 17);
        for (int it = 0; it < 200; it++) {
            for (const auto& p : pcs) {  // 既に求めた主成分の向きを取り除く
                double dp = 0;
                for (size_t j = 0; j < dim; j++) dp += v[j] * p[j];
                for (size_t j = 0; j < dim; j++) v[j] -= dp * p[j];
            }
            normalize(v);
            v = cov_mul(v);
        }
        for (const auto& p : pcs) {
            double dp = 0;
            for (size_t j = 0; j < dim; j++) dp += v[j] * p[j];
            for (size_t j = 0; j < dim; j++) v[j] -= dp * p[j];
        }
        normalize(v);
        pcs.push_back(v);
    }
    std::vector<double> xy(2 * n);
    double lo[2] = {1e300, 1e300}, hi[2] = {-1e300, -1e300};
    for (size_t i = 0; i < n; i++)
        for (int k = 0; k < 2; k++) {
            double s = 0;
            for (size_t j = 0; j < dim; j++) s += (d[i][j] - mean[j]) * pcs[k][j];
            xy[2 * i + k] = s;
            lo[k] = std::min(lo[k], s);
            hi[k] = std::max(hi[k], s);
        }
    // 縦横を同じ縮尺で [0,1] に収める
    double span = std::max(hi[0] - lo[0], hi[1] - lo[1]);
    if (span <= 0) span = 1;
    std::vector<float> out(2 * n);
    for (size_t i = 0; i < n; i++)
        for (int k = 0; k < 2; k++) {
            double c = (lo[k] + hi[k]) / 2;
            out[2 * i + k] = (float)(0.5 + (xy[2 * i + k] - c) / span);
        }
    return out;
}

class JsonOut {
public:
    explicit JsonOut(FILE* f) : f_(f) {}
    void raw(const char* s) { std::fputs(s, f_); }
    void ints(const std::vector<int>& v) {
        std::fputc('[', f_);
        for (size_t i = 0; i < v.size(); i++) std::fprintf(f_, i ? ",%d" : "%d", v[i]);
        std::fputc(']', f_);
    }
    void layer(const LayerTrace& lt, bool detail) {
        std::fprintf(f_, "{\"l\":%d,\"g\":%d", lt.layer, lt.greedy ? 1 : 0);
        if (detail) { raw(",\"ex\":"); ints(lt.expanded); raw(",\"ev\":"); ints(lt.evaluated); }
        raw(",\"c\":"); ints(lt.cand);
        if (!lt.greedy) {
            raw(",\"nb\":"); ints(lt.nbrs);
            raw(",\"pr\":[");
            for (size_t i = 0; i < lt.pruned.size(); i++) {
                std::fprintf(f_, i ? ",[%d," : "[%d,", lt.pruned[i].first);
                ints(lt.pruned[i].second);
                raw("]");
            }
            raw("]");
        }
        raw("}");
    }
private:
    FILE* f_;
};

static void write_trace(const Experiment& ex, const Tracer& tr, const std::vector<Phase>& phases,
                        const std::vector<EvalRow>& rows, const GraphStats& gs,
                        const std::vector<QueryTrace>& qts) {
    const Options& o = ex.o;
    std::filesystem::path p(o.trace);
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
    FILE* f = std::fopen(o.trace.c_str(), "w");
    if (!f) die("書き込めません: " + o.trace);
    JsonOut j(f);
    std::vector<float> xy = pca2(ex.data);

    // file:// で開いた HTML からも読めるよう、JSON を代入する .js として書く
    std::fprintf(f, "window.HNSW_TRACES=window.HNSW_TRACES||{};window.HNSW_TRACES[\"%s\"]=", o.method.c_str());
    std::fprintf(f, "{\"version\":1,\"method\":\"%s\",\"dataset\":\"%s\",\"split\":\"%s\",\"limit\":%zu,"
                    "\"M\":%d,\"efc\":%d,\"sigm_efc\":%d,\"seed\":%u,\"k\":%d,\"n1\":%zu,\"n2\":%zu,\"dim\":%zu,"
                    "\"search_detail\":%d,",
                 o.method.c_str(), o.dataset.c_str(), o.split.c_str(), o.limit, o.M, o.ef_construction,
                 o.method == "sigm" ? o.sigm_ef : 0, o.seed, o.k, ex.n1, ex.n2, ex.data.dim, o.search_detail ? 1 : 0);
    j.raw("\"proj\":[");
    for (size_t i = 0; i < xy.size(); i++) std::fprintf(f, i ? ",%.4f" : "%.4f", xy[i]);
    j.raw("],\"query_ids\":");
    j.ints(ex.query_ids);
    j.raw(",\"gt\":[");
    for (size_t q = 0; q < ex.gt.size(); q++) { if (q) j.raw(","); j.ints(ex.gt[q]); }
    j.raw("],\"phases\":[");
    for (size_t i = 0; i < phases.size(); i++)
        std::fprintf(f, "%s{\"name\":\"%s\",\"start\":%zu,\"end\":%zu,\"efc\":%d,\"dist\":%llu,\"sec\":%.3f}",
                     i ? "," : "", phases[i].name.c_str(), phases[i].start, phases[i].end, phases[i].efc,
                     (unsigned long long)phases[i].dist, phases[i].sec);
    j.raw("],\"inserts\":[");
    for (size_t i = 0; i < tr.inserts.size(); i++) {
        const auto& it = tr.inserts[i];
        std::fprintf(f, "%s{\"v\":%d,\"lv\":%d,\"ph\":%d,\"d\":%llu,\"ne\":%d,\"L\":[", i ? "," : "",
                     it.node, it.level, it.phase, (unsigned long long)it.dist, it.new_entry ? 1 : 0);
        for (size_t k = 0; k < it.layers.size(); k++) { if (k) j.raw(","); j.layer(it.layers[k], o.search_detail); }
        j.raw("]}");
    }
    j.raw("],\"eval\":[");
    for (size_t i = 0; i < rows.size(); i++)
        std::fprintf(f, "%s{\"L\":%d,\"recall\":%.5f,\"qps\":%.1f,\"dpq\":%.2f}", i ? "," : "",
                     rows[i].L, rows[i].recall, rows[i].qps, rows[i].dpq);
    std::fprintf(f, "],\"stats\":{\"layers\":%d,\"avg_deg0\":%.4f,\"reached\":%zu,\"n\":%zu,\"per_layer\":[",
                 gs.layers, gs.avg_deg0, gs.reached, gs.n);
    for (size_t i = 0; i < gs.per_layer.size(); i++) std::fprintf(f, i ? ",%zu" : "%zu", gs.per_layer[i]);
    j.raw("]},\"qtraces\":[");
    for (size_t i = 0; i < qts.size(); i++) {
        const auto& qt = qts[i];
        std::fprintf(f, "%s{\"q\":%d,\"d\":%llu,\"res\":", i ? "," : "", qt.query, (unsigned long long)qt.dist);
        j.ints(qt.result);
        j.raw(",\"L\":[");
        for (size_t k = 0; k < qt.layers.size(); k++) { if (k) j.raw(","); j.layer(qt.layers[k], true); }
        j.raw("]}");
    }
    std::fprintf(f, "],\"qtrace_L\":%d};\n", *std::max_element(o.Ls.begin(), o.Ls.end()));
    long size = std::ftell(f);
    std::fclose(f);
    std::printf("トレース: %s（%.1f MB）\n", o.trace.c_str(), size / 1048576.0);
}
