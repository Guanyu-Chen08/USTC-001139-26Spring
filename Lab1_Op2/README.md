# 选项2：城市地铁出行路径规划

## 目录

- [引言](#引言)
- [实验要求](#实验要求)
- [Python 框架](#python-框架)
  - [环境配置](#环境配置)
  - [文件结构](#文件结构)
  - [运行方式](#运行方式)
  - [GUI 说明](#gui-说明)
  - [需要完成的代码](#需要完成的代码)
- [C++ 框架](#c-框架)
  - [文件结构](#文件结构)
  - [需要完成的代码](#需要完成的代码-1)
  - [编译](#编译)
  - [运行方式](#运行方式-1)
  - [GUI 说明](#gui-说明-1)

## 引言

现代城市交通系统中，地铁网络通常由多个车站和线路构成。乘客出行时，希望找到从起点到终点的最优路径，以减少出行时间成本。

<p align="center">
  <img src="./data/Berlin/Berlin-2010-network.svg" width=600>
</p>
<p style="text-align: center; font-size: 12px; color: #666;">Berlin 2010 年地铁网络</p>

地铁系统可以抽象为一个 **图（Graph）** 模型，其中：
- 每个车站作为图的一个节点（Node）。
- 每条地铁线路作为图的一条边（Edge），边的权重表示通过该线路的时间成本。

这样，我们可以将地铁出行路径规划问题建模为一个 **最短路径问题（Shortest Path Problem）**，目标是找到从起点到终点的最短路径。

## 实验要求

请将城市地铁网络建模为一个带权图，并完成以下分析：
- 构建地铁网络图模型（节点，边，权重）
- 实现 Dijkstra 算法求解最优出行路径
  - 给定任意起点站 $s$ 和终点站 $t$，求解从 $s$ 到 $t$ 的最短路径。
  - 输出最短路径的时间成本和具体的地铁站序列
  - 请对求解结果进行可视化展示，高亮最优路径
- [Optional] 考虑换乘时间
  - 如果乘客在某个车站进行换乘，需要额外增加换乘时间（假设换乘时间恒定），请在路径计算中考虑该因素
  - 目前仅 Beijing 提供了线路信息（`Beijing-2010-station-lines.txt`），该问题只需对北京求解

实验中需要的数据见 [data](./data) 文件夹；数据格式说明请见 [data/README.md](./data/README.md)。 **最短路径问题选择 3 座城市进行求解即可；Optional 的换乘问题仅需对 Beijing 求解。**

> 数据来源：[Data for the 15 largest metro systems worldwide (2010)](https://explore.openaire.eu/search/dataset?pid=10.5281%2Fzenodo.17635286)

## Python 框架

我们提供了一个交互式可视化框架，包含完整的 GUI 界面。你只需要在算法模块中补全核心函数即可。模板仅供参考，欢迎实现更 fancy 的版本。

⚠️ 注意框架中没有提供optional problem的接口，需自己修改GUI实现换乘问题。

下面是一个演示视频（也可查看路径'./demos/python框架运行demo_compressed.mp4'）：


https://github.com/user-attachments/assets/e889bedb-990f-4f6a-b455-75eef6b51c2f



### 环境配置
推荐使用[Miniforge](https://conda-forge.org/download/)管理python环境。

使用 conda 创建并激活环境：

```bash
conda create -n mm26 python=3.12
conda activate mm26
pip install numpy matplotlib
```




### 文件结构

```
code_template/
├── main.py              # 程序入口，直接运行即可启动 GUI
├── gui.py               # GUI 模块（无需修改）
└── metro_algorithm.py   # 算法模块（需要补全）
```

### 运行方式

```bash
conda activate mm26
cd code_template
python main.py
```

### GUI 说明

运行 `python main.py` 启动交互式界面：

- **右侧面板**：选择城市 → 选择起点站 / 终点站 → 点击 "Find Shortest Path" 求解
- **左侧画布**：展示地铁网络图，选中站点会高亮标记，求解后最短路径以红色高亮显示
- **Route Output**：在右下角文本窗口输出路径详情（距离、站数、完整站名序列）

> 在算法未完成时，GUI 仍然可以正常启动和交互，但不会有实际的网络图和求解结果。

### 需要完成的代码

请打开 `code_template/metro_algorithm.py`，补全以下标有 `TODO` 的部分：

| 部分 | 说明 |
|------|------|
| `Graph` 类 | 实现无向加权图的数据结构（`add_node`、`add_edge`、`neighbors`、`edges` 等） |
| `build_graph(stations, adj)` | 使用 `Graph` 类，根据站点映射和邻接矩阵构建加权图 |
| `dijkstra(G, src, dst)` | 手写 Dijkstra 算法，返回最短距离和路径 |
| `MetroSystem.shortest_path(src_name, dst_name)` | 将站名转为 id，调用 `dijkstra` 求解 |

完成后，重新运行 `python main.py`，选择城市和站点即可验证结果。

## C++ 框架

我们同样提供了 C++ 实现框架，依赖 OpenCV 和标准库。仅供参考。

### 文件结构

```
code_template_cpp/
├── main.cpp              # 程序入口
├── metro_algorithm.h     # 算法头文件（相关类和函数的声明）
├── metro_algorithm.cpp   # 算法实现（需补全 TODO）
├── metro_gui.h           # GUI 头文件
├── metro_gui.cpp         # GUI 实现（已提供）
└── CMakeLists.txt
```

### 需要完成的代码

打开 `code_template_cpp/metro_algorithm.cpp`，实现以下三处 TODO：

| TODO | 函数 / 方法 | 说明 |
|------|------------|------|
| 1/3 | `Graph::addEdge(u, v, w)` | 向邻接表添加一条无向加权边|
| 2/3 | `Graph::numEdges()` | 返回图中无向边的总数 |
| 3/3 | `dijkstra(G, src, dst)` | 实现 Dijkstra算法，返回最短路径 `Path{total_cost, node_ids}` |

已提供的部分：
- 完整 OpenCV GUI（图形绘制、鼠标交互、信息面板）

### 编译

`code_template_cpp/` 目录下包含独立的 `CMakeLists.txt`，可直接在该目录下编译。

环境配置（安装 CMake、编译器、OpenCV）请参考 [BUILD_CPP.md](../../BUILD_CPP.md)。

```bash
# Linux / macOS（在 code_template_cpp/ 目录下执行）
mkdir build && cd build
cmake ..
make
```

```powershell
# Windows（在 code_template_cpp\ 目录下执行）
cmake -B build -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake
cmake --build build --config Release
```

**推荐使用 VS Code + CMake Tools 插件**：用 VS Code 打开 `code_template_cpp/` 文件夹，点击 CMake Tools 底部状态栏的 **Build** 即可。

### 运行方式

可执行文件 `op2_template` 位于 `build/` 目录下。城市通过**命令行参数**指定。

```bash
# 从 build/ 运行（自动查找 ../../data，默认第一个城市）
./op2_template

# 指定城市（名称，不区分大小写）
./op2_template Barcelona

# 指定数据目录和城市
./op2_template ../../data Barcelona

# 或按城市索引 0, 1, 2, ...
./op2_template ../../data 0          
```

> **Windows：**
> ```powershell
> .\op2_template.exe
> .\op2_template.exe Barcelona
> .\op2_template.exe C:\path\to\data Beijing
> ```

### GUI 说明

程序启动后弹出地铁网络可视化窗口，左侧为图形区，右侧为信息面板。

| 操作 | 说明 |
|------|------|
| **第 1 次左键单击** | 选中起始站（标记为绿色 S） |
| **第 2 次左键单击** | 选中终点站（标记为蓝色 D） |
| **第 3 次左键单击** | 重置选择 |
| 按 **Space / Enter** | 执行 Dijkstra，在图上高亮最短路径（蓝色），右侧显示距离与站名 |
| 按 **r** | 清除当前选择和路径 |
| 按 **q / Esc** | 退出程序 |


