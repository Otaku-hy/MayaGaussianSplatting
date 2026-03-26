# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目简介

这是一个 Maya 插件（C++/DX11），用于在 Autodesk Maya 的 Viewport 2.0 中可视化 3D Gaussian Splatting 点云。目前是 alpha/调试原型阶段——着色器是调试用的（点扩展为四边形 + 圆形遮罩），并非完整的生产级 Gaussian Splatting 渲染器。

## 构建

**工具链：** Visual Studio 2022，MSBuild，仅支持 x64。无自动化测试或 lint 命令。

**构建前必须设置的环境变量：**
```
MAYA_LOCATION    - Maya 安装目录（用于头文件和库的搜索路径）
DEVKIT_LOCATION  - 编译后 .mll 插件文件的目标目录
```

通过以下解决方案文件构建：
```
GaussianSplatting/GaussianSplatting.sln
```

构建后步骤会将 `.mll` 文件复制到 `$(DEVKIT_LOCATION)\plug-ins\plug-ins\`。

**Maya 设置：** 需将渲染引擎切换为 DirectX 11：
`Windows → Settings/Preferences → Preferences → Display → Viewport 2.0 → Rendering Engine: DirectX 11`

## 架构

### 数据流
```
PLY 文件 → PLYReader → GaussianData → GaussianNode (MPxLocatorNode)
                                              ↓
                                    GaussianDrawOverride (DX11)
                                              ↓
                                gaussianDebug.fx 着色器 (VS → GS → PS)
```

### 核心模块

- **`plugin_main.cpp`** — 插件入口/出口；向 Maya 注册 `gaussianSplat` 节点类型及其绘制覆盖。
- **`GaussianNode`** — Maya DAG 节点（`MPxLocatorNode`）。持有 `filePath` 和 `pointSize` 属性。属性变更时调用 PLYReader 并存储解析结果。
- **`GaussianData.h`** — POD 结构体：`GaussianSplat`（单个 splat 的原始数据）和 `GaussianData`（展平后的 GPU 就绪数组）。
- **`PLYReader`** — 解析 `binary_little_endian` 和 ASCII 格式的 PLY 文件；提取位置、SH DC 系数（f_dc_0–2）、不透明度、缩放和旋转四元数。
- **`GaussianDrawOverride`** — `MPxDrawOverride` 实现；管理 DX11 设备、顶点/常量缓冲区和着色器管线。这是**当前的活跃渲染路径**。
- **`GaussianGeometryOverride`** — `MPxGeometryOverride`（较旧的 VP2 API）；保留但已被 `GaussianDrawOverride` 取代。
- **`gaussianDebug.fx`** — HLSL 着色器：VS 变换顶点位置，GS 将每个点扩展为屏幕对齐的四边形，PS 应用带软边淡出的圆形遮罩。

### Maya 插件模式

Maya 采用节点与覆盖分离的设计：`GaussianNode` 负责存储数据和属性，`GaussianDrawOverride` 负责所有渲染逻辑。两者通过 `plugin_main.cpp` 中的节点类型注册关联。新增可渲染属性需同时修改两个类，并处理覆盖中的 `prepareForDraw` / `draw` 周期。

### 生产级渲染管线（计算着色器，进行中）

位于 `GaussianSplatting/shaders/`，所有计算着色器均需要 **SM 6.x**（DXC 编译器，`#pragma use_dxc`）和 HLSL Wave Intrinsics。

#### `WaveCommon.hlsl` — 共享头文件
被所有计算着色器 `#include`，提供：
- **块级 Reduce/Scan**：`BlockReduce`、`BlockScanExclusive`、`BlockScanInclusive`——两级（warp → block）实现，通过 `groupshared` 内存汇聚各 wave 结果
- **Decoupled Lookback 状态位打包**：`PackBlockState` / `UnpackBlockFlag` / `UnpackBlockSum`——高 2 位存 flag（`01`=块和已就绪，`10`=前缀和已就绪），低 30 位存 sum
- **WaveMatch 模拟**：`WaveMatch32_8bits`——在不支持 SM 6.5 `WaveMatch` 时对 8-bit key 逐 bit 做 ballot 模拟
- **直方图工具**：`BuildHistogram`（wave 内 scatter-and-count）、`ScanHistogram`（wave 间扫描）、`BuildHistogramMultiPlace`（一次遍历同时统计 4 个 digit 位）
- **常量**：`GROUP_SIZE=256`，`MAX_WAVES=8`，`RADIX=8`，`BUCKETS=256`，`SORT_COUNT=4`

#### `Preprocessing.compute` — 逐 splat 预处理（`PreprocessKernel`，256 线程/组）
每帧将原始 PLY 数据转换为 GPU 渲染所需的屏幕空间数据。

| 输入缓冲 | 说明 |
|---|---|
| `gPositionWS` | 世界空间位置 float3 |
| `gScale` | 3 轴缩放 float3 |
| `gRotation` | 旋转四元数 float4 |
| `gOpacity` | 原始不透明度（logit 空间）float |
| `gSHsCoeff` | 球谐系数，每个 splat 16 组 float3（0–3 阶）|

| 输出缓冲 | 说明 |
|---|---|
| `gPositionSS` | 屏幕像素坐标 float2 |
| `gDepth` | NDC 深度（z/w）float |
| `gRadius` | 像素半径 = `ceil(3√λmax)` float |
| `gColor` | SH 评估后的 RGB float3 |
| `gCov2D_opacity` | 逆 2D 协方差 (a,b,c) + opacity，打包为 float4 |

关键函数：
- `Get3DCovariance(scale, quat)` → `R · S² · Rᵀ`（世界空间 3×3 协方差）
- `Get2DCovariance(cov3D, viewMat, mean, tanHalfFov)` → EWA Jacobian 投影 → 2D 协方差 + 0.3 正则化
- `ComputeSphericalHarmonics(idx, pos, camPos)` → 沿视线方向评估 0–3 阶 SH（16 系数）

**当前代码已知问题：**
- `Get2DCovariance` 声明返回 `float2`，实际返回 `float3`——类型不匹配
- `ComputeSphericalHarmonics` 末尾 `max(shColor, 1.0f)` 将最小值钳制为白色，应为 `0.0f`
- `#pragma kernel PreprocessKernel` 与函数名 `CSMain` 不一致——调度时会找不到入口

#### `WaveRadixSort.compute` — GPU 基数排序（用于深度排序）
对深度 key（uint32）做 8-bit 基数、4 趟的 32-bit 完整排序。

| Kernel | 说明 |
|---|---|
| `GlobalHistogramKernel` | 同时统计所有 4 个 digit 位的全局直方图，原子累加到 `gHistogram[BUCKETS×4]` |
| `GlobalHistogramScanKernel` | 对全局直方图做 exclusive prefix scan → `gHistogramOffset` |
| `RadixSortNew`（当前使用） | 每线程处理 16 个元素（ITEMS\_PER\_THREAD），decoupled lookback 计算块间偏移，scatter 到 `gOutput` |
| `RadixSortKernel`（已废弃） | 每线程 1 元素的旧版本，被 `RadixSortNew` 取代 |

Decoupled Lookback 流程：当前块发布块和（flag=1）→ 扫描前序块 → 获得完整前缀和后升级为（flag=2），后续块可提前终止回溯。

#### `WaveScan.compute` — 前缀扫描工具
排序管线的辅助 kernel，也可独立使用。

| Kernel | 说明 |
|---|---|
| `BlockReduceKernel` | 每块求和写入 `gblockSum[]` |
| `GlobalScanKernel` | 对块和做 exclusive scan（单组） |
| `BlockScanKernel` | 每块 inclusive scan + 加入组偏移 → `gOutput` |
| `ChainedScanKernel` | 单趟扫描，内置 decoupled lookback，无需单独的全局扫描分发 |
