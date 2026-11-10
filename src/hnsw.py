"""
HNSW の教育用・最小実装（純 Python、外部ライブラリなし）

論文: Malkov & Yashunin, "Efficient and robust approximate nearest
neighbor search using Hierarchical Navigable Small World graphs" (2020)

読み方:
  - グラフ本体は self.graph[node][layer] = {隣のノード番号の集合}
  - 中核は search_layer() の1関数。探索も構築もこれを層ごとに呼ぶだけ。
高速化（numpy、近似距離、並列構築など）は一切していない。
"""

import heapq
import math
import random


def dist2(a, b):
    """2点間の距離の2乗（大小比較だけなら sqrt は不要なので省略）。"""
    return sum((x - y) ** 2 for x, y in zip(a, b))


class HNSW:
    def __init__(self, M=16, ef_construction=100, seed=0):
        self.M = M                      # 1ノードの最大次数（Layer 0 は 2*M）
        self.ef_construction = ef_construction
        self.mL = 1.0 / math.log(M)     # レベル抽選の係数
        self.rng = random.Random(seed)

        self.vectors = []               # ノード番号 -> ベクトル
        self.levels = []                # ノード番号 -> そのノードの最上層
        self.graph = []                 # ノード番号 -> [層ごとの隣接集合]
        self.entry_point = None         # 探索の入口（最上層にいるノード）
        self.max_level = -1

    # ---------------------------------------------------------------
    # 中核: 1つの層の中を探索する
    #   C = 次に調べる候補（近い順に出てくる最小ヒープ）
    #   W = いま保持している上位 ef 件（遠いものが先頭の最大ヒープを
    #       「距離をマイナスにして」最小ヒープで代用）
    # ---------------------------------------------------------------
    def search_layer(self, q, entry, ef, layer):
        visited = {entry}
        d0 = dist2(q, self.vectors[entry])
        C = [(d0, entry)]               # 候補キュー
        W = [(-d0, entry)]              # 結果（距離の符号反転で最大ヒープ化）
        heapq.heapify(C)
        heapq.heapify(W)

        while C:
            d, v = heapq.heappop(C)
            if d > -W[0][0] and len(W) == ef:
                break                    # 候補の最良でも結果の最悪より遠い
            for u in self.graph[v][layer]:
                if u in visited:
                    continue
                visited.add(u)
                du = dist2(q, self.vectors[u])
                if len(W) < ef or du < -W[0][0]:
                    heapq.heappush(C, (du, u))
                    heapq.heappush(W, (-du, u))
                    if len(W) > ef:
                        heapq.heappop(W)  # 一番遠いものを捨てて ef 件に保つ
        # 結果を (距離, ノード) の近い順で返す
        return sorted((-negd, node) for negd, node in W)

    # ---------------------------------------------------------------
    # 隣接点の選び方（ヒューリスティック）:
    #   近い順に見て「既に選んだどの点よりも q に近い」点だけ残す。
    #   → 辺が同じ方向に固まらず、方角がばらける。
    #   candidates は (距離, ノード) のリスト。
    # ---------------------------------------------------------------
    def select_neighbors(self, q, candidates, limit):
        selected = []
        for _, c in sorted(candidates):
            if all(dist2(q, self.vectors[c]) < dist2(self.vectors[c], self.vectors[r])
                   for r in selected):
                selected.append(c)
                if len(selected) == limit:
                    break
        return selected

    def _random_level(self):
        # ほとんどのノードは低い層までしか出ず、稀に高い層まで出る
        return int(-math.log(self.rng.random() + 1e-12) * self.mL)

    def _max_degree(self, layer):
        return 2 * self.M if layer == 0 else self.M

    # ---------------------------------------------------------------
    # 構築: 点を1つずつ入れる
    # ---------------------------------------------------------------
    def insert(self, vec):
        node = len(self.vectors)
        level = self._random_level()
        self.vectors.append(vec)
        self.levels.append(level)
        self.graph.append([set() for _ in range(level + 1)])

        if self.entry_point is None:
            self.entry_point, self.max_level = node, level
            return node

        ep = self.entry_point
        # 自分がいない上の層では、入口を自分の近くまで寄せるだけ（ef=1 の貪欲）
        for layer in range(self.max_level, level, -1):
            ep = self.search_layer(vec, ep, ef=1, layer=layer)[0][1]

        # 自分がいる層では近傍を探し、双方向に辺を張る
        for layer in range(min(level, self.max_level), -1, -1):
            W = self.search_layer(vec, ep, ef=self.ef_construction, layer=layer)
            neighbors = self.select_neighbors(vec, W, self.M)
            self.graph[node][layer].update(neighbors)
            for nb in neighbors:
                self.graph[nb][layer].add(node)
                max_deg = self._max_degree(layer)
                if len(self.graph[nb][layer]) > max_deg:
                    # 次数オーバーした相手側は、相手から見た候補で選び直す
                    cand = [(dist2(self.vectors[nb], self.vectors[u]), u)
                            for u in self.graph[nb][layer]]
                    kept = self.select_neighbors(self.vectors[nb], cand, max_deg)
                    self.graph[nb][layer] = set(kept)
            ep = W[0][1]

        if level > self.max_level:
            self.entry_point, self.max_level = node, level
        return node

    # ---------------------------------------------------------------
    # 探索: 上の層で大きく跳び、下の層（Layer 0）で候補 ef 件を持って詰める
    # ---------------------------------------------------------------
    def search(self, q, k=10, ef=50):
        assert self.entry_point is not None, "まず insert してください"
        assert ef >= k, "ef は k 以上にしてください"
        ep = self.entry_point
        for layer in range(self.max_level, 0, -1):
            ep = self.search_layer(q, ep, ef=1, layer=layer)[0][1]
        W = self.search_layer(q, ep, ef=ef, layer=0)
        return [node for _, node in W[:k]]


# ---------------------------------------------------------------
# 動作確認: 全探索（正確な答え）と比べて recall@k を測る
# ---------------------------------------------------------------
if __name__ == "__main__":
    rng = random.Random(42)
    N, DIM, K, QUERIES = 500, 16, 10, 20

    data = [[rng.gauss(0, 1) for _ in range(DIM)] for _ in range(N)]
    index = HNSW(M=16, ef_construction=100, seed=1)
    for v in data:
        index.insert(v)

    def brute_force(q):
        return [i for _, i in sorted((dist2(q, v), i) for i, v in enumerate(data))[:K]]

    total_recall = 0.0
    for _ in range(QUERIES):
        q = [rng.gauss(0, 1) for _ in range(DIM)]
        approx = set(index.search(q, k=K, ef=50))
        exact = set(brute_force(q))
        total_recall += len(approx & exact) / K
    print(f"recall@{K} = {total_recall / QUERIES:.3f}  (N={N}, 次元={DIM})")
