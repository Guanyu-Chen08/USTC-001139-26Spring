# Copyright 2026, Yumeng Liu @ USTC

"""
地铁网络算法模块 —— 数据加载、图构建、Dijkstra 求解
"""

import csv
import heapq
from pathlib import Path

import numpy as np


# ============================================================
# Graph 数据结构
# ============================================================

class Graph:
    """
    简单的无向加权图。

    需要实现的接口
    -------------
    - add_node(node_id, **attrs) : 添加节点
    - add_edge(u, v, weight)     : 添加无向边
    - neighbors(node_id)         : 返回邻居字典 {neighbor_id: weight}
    - number_of_nodes()          : 返回节点数
    - number_of_edges()          : 返回边数
    - edges()                    : 返回所有边列表 [(u, v, weight), ...]

    属性
    ----
    nodes : dict[int, dict]
        节点字典，{node_id: {"name": str, ...}}。
        GUI 会读取此属性来获取节点信息，请确保 add_node 时正确填充。

    提示
    ----
    你可以自由选择底层数据结构（邻接表、邻接矩阵、边列表等）。
    """

    def __init__(self):
        self.nodes = {}
        # TODO: 初始化你的数据结构
        self.adj = {}

    def add_node(self, node_id, **attrs):
        """
        添加节点。

        Parameters
        ----------
        node_id : int
            节点编号。
        **attrs
            节点属性，例如 name="StationA"。
        """
        # TODO: 将节点及其属性存入 self.nodes，并初始化邻接结构
        # pass
        self.nodes[node_id] = attrs
        if node_id not in self.adj:
            self.adj[node_id] = {}

    def add_edge(self, u, v, weight=1.0):
        """
        添加无向边 (u, v)，权重为 weight。
        """
        # TODO: 在邻接结构中记录无向边及权重
        # pass
        if u not in self.adj:
            self.adj[u] = {}
        
        if v not in self.adj:
            self.adj[v] = {}

        self.adj[u][v] = weight
        self.adj[v][u] = weight

    def neighbors(self, node_id):
        """
        返回 node_id 的邻居字典 {neighbor_id: weight}。

        若节点不存在或无邻居，返回空字典。
        """
        # TODO: 返回邻居及对应权重
        return self.adj.get(node_id, {})

    def number_of_nodes(self):
        """返回图中节点数量。"""
        # TODO
        return len(self.nodes)

    def number_of_edges(self):
        """返回图中边的数量（每条无向边只计一次）。"""
        # TODO
        num = 0

        for node in self.adj:
            num += len(self.adj[node])

        return num // 2

    def edges(self):
        """
        返回所有边的列表 [(u, v, weight), ...]，每条边只出现一次。

        GUI 的绘图函数会调用此方法来绘制网络边。
        """
        # TODO
        edges = []
        for u in self.adj:
            for v in self.adj[u]:
                if u < v:
                    edges.append((u, v, self.adj[u][v]))
        return edges


# ============================================================
# 数据加载
# ============================================================

def load_station_map(tsv_path: str) -> dict[int, str]:
    """读取 station-id-map.tsv，返回 {id: name} 映射。"""
    stations: dict[int, str] = {}
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            stations[int(row["id"])] = row["name"]
    return stations


def load_adjacency_matrix(csv_path: str) -> np.ndarray:
    """读取 adjacency-distance.csv，返回 N×N numpy 矩阵。"""
    return np.loadtxt(csv_path, delimiter=",")


def load_lines(lines_path: str) -> dict[str, set]:
    station_lines = {}

    with open(lines_path, encoding="utf-8") as file:
        information = csv.DictReader(file, delimiter="\t")

        for row in information:
            name = row["station"]
            lines = set(row["lines"].split(","))
            station_lines[name] = lines
    
    return station_lines

def build_graph(stations: dict[int, str], adj: np.ndarray) -> Graph:
    """
    根据站点映射和邻接矩阵构建加权图。

    Parameters
    ----------
    stations : dict[int, str]
        站点 id → 名称映射（id 从 1 开始）。
    adj : np.ndarray
        N×N 邻接距离矩阵，adj[i,j] > 0 表示站点 i+1 与 j+1 之间有边。

    Returns
    -------
    Graph
        带权无向图，节点属性 name 为站名，边权 weight 为距离。

    提示
    ----
    - 使用 Graph.add_node(node_id, name=...) 添加节点
    - 使用 Graph.add_edge(u, v, weight=...) 添加边
    - 矩阵下标从 0 开始，站点 id 从 1 开始
    """
    # TODO: 构建加权图
    G = Graph()

    for node_id, name in stations.items():
        G.add_node(node_id, name=name)

    n = adj.shape[0]
    
    for i in range(n):
        for j in range(i + 1, n):
            weight = adj[i][j]
            if weight > 0:
                G.add_edge(i + 1, j + 1, weight)

    return G


# ============================================================
# Dijkstra 最短路径
# ============================================================

def dijkstra(G: Graph, src: int, dst: int) -> tuple[float, list[int]]:
    """
    实现 Dijkstra 求 src → dst 最短路径。

    Parameters
    ----------
    G : Graph
        带权图。
    src : int
        起点站点 id。
    dst : int
        终点站点 id。

    Returns
    -------
    (cost, path) : (float, list[int])
        cost 为最短距离，path 为站点 id 序列（含起终点）。
        若不可达，返回 (float("inf"), [])。

    提示
    ----
    - 使用 G.neighbors(u) 获取邻居字典 {neighbor_id: weight}
    - 使用 heapq 实现最小堆
    - 使用前驱字典 prev 回溯路径
    """
    # TODO: 实现 Dijkstra 算法
    pq = [(0.0, src)]
    dist = {src: 0.0}
    prev = {src: None}

    while pq:
        d, u = heapq.heappop(pq)

        if d > dist.get(u, float('inf')):
            continue
        if u == dst:
            break

        for v, weight in G.neighbors(u).items():
            dist_curr = d + weight
            if dist_curr < dist.get(v, float('inf')):
                dist[v] = dist_curr
                prev[v] = u
                heapq.heappush(pq, (dist_curr, v))

    if dst not in dist:
        return float("inf"), []
    
    path = []
    curr = dst
    while curr is not None:
        path.append(curr)
        curr = prev[curr]

    path.reverse()
    
    return dist[dst], path


def dijkstra_considering_transfer(G: Graph, src: int, dst: int, station_lines: dict[int, set], transfer_time: float) -> tuple[float, list[int]]:
    pq = []
    dist = {}
    prev = {}

    src_lines = station_lines.get(src, set())

    for line in src_lines:
        pq.append((0.0, src, line))
        dist[(src, line)] = 0.0
        prev[(src, line)] = None
    heapq.heapify(pq)

    dst_state = None

    while pq:
        d, u, line_u = heapq.heappop(pq)
        state_u = (u, line_u)

        if d > dist.get(state_u, float('inf')):
            continue
            
        if u == dst:
            dst_state = state_u
            break

        u_lines = station_lines.get(u, set())

        for v, weight in G.neighbors(u).items():
            v_lines = station_lines.get(v, set())
            common = u_lines & v_lines

            if line_u in common:
                new_line = line_u
                extra = 0.0
            elif common:
                new_line = next(iter(common))
                extra = transfer_time
            else:
                continue

            dist_curr = d + weight + extra
            state_v = (v, new_line)

            if dist_curr < dist.get(state_v, float('inf')):
                dist[state_v] = dist_curr
                prev[state_v] = state_u
                heapq.heappush(pq, (dist_curr, v, new_line))

    if dst_state not in dist:
        return float("inf"), []
    
    path = []
    curr = dst_state
    while curr is not None:
        node, _ = curr
        path.append(node)
        curr = prev[curr]

    path.reverse()
    
    return dist[dst_state], path


# ============================================================
# MetroSystem 高层封装
# ============================================================

class MetroSystem:
    """封装单个城市的地铁系统：加载数据、构建图、求解路径。"""

    def __init__(self, data_dir: str | Path):
        data_dir = Path(data_dir)
        self.city = data_dir.name

        tsv = next(data_dir.glob("*station-id-map.tsv"))
        csv_f = next(data_dir.glob("*adjacency-distance.csv"))

        self.stations = load_station_map(str(tsv))
        adj = load_adjacency_matrix(str(csv_f))
        self.graph = build_graph(self.stations, adj)

        self.name_to_id: dict[str, int] = {
            name: sid for sid, name in self.stations.items()
        }

        self.station_lines = {}
        self.has_line_info = False

        
        if self.city == 'Beijing':
            lines_info = data_dir / "Beijing-2010-station-lines.txt"
            self.has_line_info = True

            info = load_lines(str(lines_info))
            for name, lines in info.items():
                id = self.name_to_id.get(name)
                if id is not None:
                    self.station_lines[id] = lines

        self.transfer_time = 3.0

    def sorted_station_names(self) -> list[str]:
        """返回按字母排序的站名列表。"""
        return sorted(self.stations.values())

    def shortest_path(self, src_name: str, dst_name: str) -> tuple[float, list[int]]:
        """
        求两站之间的最短路径。

        Parameters
        ----------
        src_name : str
            起点站名。
        dst_name : str
            终点站名。

        Returns
        -------
        (cost, path) : (float, list[int])
            cost 为最短距离 (km)，path 为站点 id 序列。

        提示
        ----
        - 使用 self.name_to_id 将站名转为 id
        - 调用 dijkstra(self.graph, src_id, dst_id)
        """
        # TODO: 将站名转为 id，调用 dijkstra 函数求解
        # return float("inf"), []
        src_id = self.name_to_id[src_name]
        dst_id = self.name_to_id[dst_name]

        if self.has_line_info:
            return dijkstra_considering_transfer(self.graph, src_id, dst_id, self.station_lines, self.transfer_time)

        return dijkstra(self.graph, src_id, dst_id)



def detect_cities(data_root: str | Path) -> list[str]:
    """扫描 data_root 下所有包含数据文件的城市子目录。"""
    data_root = Path(data_root)
    cities: list[str] = []
    for d in sorted(data_root.iterdir()):
        if d.is_dir() and list(d.glob("*adjacency-distance.csv")):
            cities.append(d.name)
    return cities
