# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目简介

这是一个 Maya 插件（C++/DX11），用于在 Autodesk Maya 2026 的 Viewport 2.0 中可视化 3D Gaussian Splatting 点云。当前 master 分支已实现完整的生产级渲染管线：**Compute 预处理 → GPU Radix 排序 → 实例化椭圆 Splat 渲染**，同时保留了简单圆点的 debug 路径。插件同时支持 Maya 几何体深度遮挡和完整 Maya 变换（移动/旋转/缩放）。

## 构建

**工具链：** Visual Studio 2022（v143），MSBuild，仅 x64。无自动化测试或 lint。

**构建前必须设置的环境变量：**
```
MAYA_LOCATION    — Maya 安装目录（头文件/库搜索路径），例：C:\Program Files\Autodesk\Maya2026
DEVKIT_LOCATION  — Maya devkit 根目录
```

**解决方案：** `GaussianSplatting/GaussianSplatting.sln`

**本地构建脚本：** 仓库根目录 `_build_tmp.bat`（设置环境变量 + 调用 MSBuild Release|x64）。`build.bat` 是较早的版本。

**Post-build：** 项目会把 `GaussianSplatting.mll` 复制到仓库根目录（`$(SolutionDir)..\`）。

**Maya 设置：** 必须把 Viewport 2.0 渲染引擎切换为 DirectX 11：
`Windows → Settings/Preferences → Preferences → Display → Viewport 2.0 → Rendering Engine: DirectX 11`

## 架构

### 节点设计（v0.2 — 数据/渲染分离）

```
  gaussianSplatData (MPxNode)            gaussianSplat (MPxLocatorNode)
  ─────────────────────────              ──────────────────────────────
  filePath ── attributeAffects ──▶       inputData (message, 连入 DataNode)
  dataReady (compute 时加载 PLY)          pointSize
  outputData (message, 输出端)             renderMode (0=auto, 1=debug, 2=prod, 3=prod+fixedRadius)
  持有 GaussianData + 5 个共享 SRV  ◀─────  引用 DataNode 的 SRV, 由 DrawOverride 渲染
```

一个 DataNode 可被多个 gaussianSplat 渲染节点连接，PLY 数据和 GPU 输入缓冲（PositionWS/Scale/Rotation/Opacity/SHCoeffs）仅上传一次。

### 数据流

```
PLY 文件 ──▶ PLYReader ──▶ GaussianData
                                │
                                ▼
                      GaussianDataNode.compute (DG 触发)
                                │  uploadInputBuffersIfNeeded
                                ▼
         5 个 StructuredBuffer SRV (positionWS/scale/rotation/opacity/shCoeffs)
                                │  非拥有引用, 每帧从 DataNode 拷贝
                                ▼
              GaussianDrawOverride::prepareForDraw
                                │
                                ▼
                  GaussianDrawOverride::draw (DX11)
                      ┌──────────┴──────────┐
            Production 路径              Debug 路径
            (Compute+Sort+Ellipse)       (VS+GS+PS 圆点)
```

### 核心模块（`GaussianSplatting/src/`）

| 文件 | 行数 | 说明 |
|---|---|---|
| `plugin_main.cpp` | 189 | 注册 2 个节点 + DrawOverride，执行内嵌 MEL 构建 **"Gaussian Splatting" 顶部菜单**（Load PLY / Create Data Node / Create Render Node / Connect Selected） |
| `GaussianDataNode.{h,cpp}` | 86+194 | `MPxNode`，拥有 `GaussianData` 与 5 个共享 SRV。`compute` 在 `filePath` 变化时调用 PLYReader，打印 scale 诊断（检测缺失 scale_0/1/2 导致的 1m 半径问题） |
| `GaussianNode.{h,cpp}` | 46+78 | `MPxLocatorNode`，纯渲染代理；通过 `inputData` message attr 连接上游 DataNode；`boundingBox()` 从 DataNode 拿 AABB（支持选择框和视锥剔除） |
| `GaussianDrawOverride.{h,cpp}` | 194+1398 | **当前活跃渲染路径**。`GaussianDrawData`（MUserData）拥有 per-render-node 的 DX11 资源；`GaussianDrawOverride::draw` 根据 `renderMode` 选择管线。所有 HLSL 作为 `static const char*` 内嵌（见下文）。`isTransparent()` 返回 true 以保证 Maya 几何体先写深度 |
| `GaussianGeometryOverride.{h,cpp}` | 52+201 | 旧版 `MPxGeometryOverride`，**已保留但未使用**，由 DrawOverride 取代 |
| `PLYReader.{h,cpp}` | 12+357 | 支持 binary_little_endian 和 ASCII；在 `buildGPUArrays()` 里做：`exp(log_scale)`、四元数归一化、SH 系数从 PLY 的 planar 布局（f_rest[0..44]，所有 r 后所有 g 后所有 b）重新交织为 16 组 float3；同时计算 object-space AABB |
| `GaussianData.h` | 49 | POD：`GaussianSplat` 原始数据 + `GaussianData` 展平数组（positions/colors/scaleWS/rotationWS/opacityRaw/shCoeffs）+ bbox |

### 关键点：Shader 都在 C++ 字符串里

**实际运行的所有 HLSL 代码都以 `static const char*` 内嵌在 `GaussianDrawOverride.cpp` 顶部**，通过 `D3DCompile` 运行时编译：

| 变量 | 内容 | 源码位置 |
|---|---|---|
| `kDbgShaderSrc` | Debug 路径：VS+GS+PS，圆形 Splat（点→四边形→圆遮罩） | 第 27 行附近 |
| `kProdShaderSrc` | Production 渲染：VS 实例化（4 顶点/splat 从 `gSortedIndices` 读索引）+ PS（Gaussian 椭圆 alpha） | 第 91 行附近 |
| `kPreprocessCS` | 预处理 compute shader，`PreprocessKernel` 256 线程/组 | 第 164 行附近 |
| `kRadixSortCS` | 4 个 radix sort kernel，通过 `#define` 切换：`KEYGEN_KERNEL` / `COUNT_KERNEL` / `SCAN_KERNEL` / `SCATTER_KERNEL` | 第 340 行附近 |

**`GaussianSplatting/shaders/` 目录下的 `Preprocessing.compute` 和 `gaussianDebug.fx` 是遗留参考文件，`.vcxproj` 中已标记 `ExcludedFromBuild`，不参与构建**。修改任何 shader 都必须编辑 `GaussianDrawOverride.cpp` 里的字符串常量。

## 渲染管线细节

### Production 路径（`renderMode=0` auto 或 `=2` 强制）

每帧在 `GaussianDrawOverride::draw` 里执行以下顺序（约第 1180 行起）：

1. **更新 Preprocess CB**：world/view/proj 矩阵、cameraPos、tanHalfFov、filmW/H、splat 数、`debugFixedRadius`（`renderMode=3` 时设为 5，用于诊断）
2. **Dispatch `PreprocessKernel`**（256 线程/组，共 `⌈N/256⌉` 组）
   - 读 5 个输入 SRV（positionWS/scale/rotation/opacity/shCoeffs）
   - 写 5 个输出 UAV（positionSS/depth/radius/color/cov2D_opacity）
   - 做 view 空间剔除（`posVS.z >= -0.2` 丢弃）、object→world→view→clip 变换、3D→2D 协方差投影、radius 计算（`ceil(3√λmax)`，上限 1024）、4 阶 SH 求值
3. **GPU Radix Sort**：
   - **3a. KeyGen**（256 线程/组）：`gKeysA[i] = ~FloatToSortKey(depth[i])` → 降序 = back-to-front；`gValsA[i] = i`
   - **3b. 4 趟 8-bit radix**（pass=0..3，`shift = pass*8`），每趟：
     - **Count**：每 block 一个 256-bucket 直方图 → `gBlockHist[numBlocks×256]`
     - **Scan**（单组，256 线程）：每 digit 跨 block 做 exclusive prefix sum，再对 digit-total 做 exclusive scan，写回到 `gBlockHist` 作为全局起始偏移
     - **Scatter**：每个 block 用 `groupshared sLocalRank` + `InterlockedAdd` 局部计数，写到 `gBlockOff + rank` 的输出位置
     - Ping-pong：even pass A→B，odd pass B→A；4 趟（偶数）后结果在 A
   - Tile size：`SORT_GROUP_SIZE(256) × ITEMS_PER_THREAD(16) = 4096` 元素/block
4. **更新 Render CB**（只含 viewport 尺寸）
5. **DrawInstanced(4, N, 0, 0)**：
   - 拓扑 `TRIANGLESTRIP`，无 GS，无 InputLayout，无 VB
   - VS 用 `SV_VertexID`（0..3）作为 quad 角标，`SV_InstanceID` 作为绘制顺序
   - VS 做 `idx = gSortedIndices[iid]` 查排序后的 splat 索引，然后从 compute 输出 SRV 读取 posSS/radius/color/cov2D/depth
   - PS 用 inverse 2D covariance 做 EWA Gaussian alpha：`alpha = opacity * exp(-0.5 * (a·x² + 2b·xy + c·y²))`，`< 1/255` 丢弃

**CS 5.0 + 无 wave intrinsics**：所有 compute shader 都用 `cs_5_0` target 编译，避免 SM 6.x 依赖。排序靠 `groupshared` + `InterlockedAdd` 实现，不使用 `WaveActiveSum` 等 wave 函数。

### Debug 路径（`renderMode=1` 或 production fallback）

直接从共享输入 SRV 读 position/opacity/SH[0]：
- VS：SH degree-0 颜色（`sh[0] * 0.282095 + 0.5`）+ sigmoid opacity
- GS：每个点扩展为屏幕对齐 quad（`pointSize` 像素）
- PS：`clip(1 - r²)` 圆形 + 软边淡出

**用途**：快速可视化、检查 PLY 加载是否正确、点大小可在 AE 的 `pointSize` 滑杆上调。

### Maya 集成要点

- **`isTransparent() = true`**（`GaussianDrawOverride.h:177`）：让 splat 在透明 pass 绘制，Maya 不透明几何体的深度已经在 depth buffer 里 → 自动正确遮挡
- **变换支持**：每帧 `objPath.inclusiveMatrix()` 取 `worldMat`，传给 compute shader；shader 内做 `posWS = posOS * worldMat`（Maya row-major 约定）；协方差按 `cov_WS = W^T * cov_obj * W` 变换（`GaussianDrawOverride.cpp:293-296`）
- **包围盒**：`GaussianData::buildGPUArrays()` 里计算 object-space AABB，`GaussianDataNode::boundingBox()` 暴露给 Maya，支持选择框、视锥剔除、变换手柄自动适配
- **输入缓冲懒加载**：DataNode 的 DX11 buffer 在首次 `prepareForDraw` 时通过 `uploadInputBuffersIfNeeded` 上传，PLY 切换时自动释放重建
- **DX11 状态保存/恢复**：`draw()` 开头保存 Maya 的 blend/RS/DS/InputLayout/VS/GS/PS，结尾全部恢复，避免污染 Maya 渲染状态
- **SH 步长常量**：C++ 用 `kSHCoeffsPerSplat = 16`（`GaussianData.h:6`），作为 debug shader CB 的 `gSHStride` 传入

## 常量与 Magic Number 速查

| 名称 | 值 | 说明 |
|---|---|---|
| `kSHCoeffsPerSplat` | 16 | 每 splat 16 组 float3 SH（0–3 阶） |
| `kSortGroupSize` | 256 | Radix sort 线程组大小 |
| `kSortItemsPerThread` | 16 | 每线程处理的元素数 |
| `kSortTileSize` | 4096 | = 256 × 16，每 block 处理的元素数 |
| `kRadixSize` | 256 | 8-bit radix 桶数 |
| `PreprocessKernel` 组大小 | 256 | `⌈N/256⌉` dispatch |
| Radius 上限 | 1024 px | 超过直接剔除，避免 GPU 杀手级全屏 quad |
| Z 剔除阈值 | `posVS.z >= -0.2` | 近平面之后即剔除（Maya 右手系，负 z 朝前） |

## 常见陷阱

- **"splats 都是 1m 半径"**：PLY 缺少 `scale_0/1/2` 属性（COLMAP sparse 点云），DataNode 会在 Script Editor 打印 `<<< ALL 1.0 — PLY likely missing scale... >>>` 警告。要加载 **训练输出**，比如 `output/<scene>/point_cloud/iteration_30000/point_cloud.ply`，不是 COLMAP 的 sparse/points3D
- **`shaders/Preprocessing.compute` 与运行版不一致**：磁盘那份是老版本，有 `#pragma kernel` vs 入口名不一致、SH 末尾 `max(..., 1.0)` 写错为 0.0、`Get2DCovariance` 参数错误等问题。**运行时用的是 `kPreprocessCS` 字符串常量**，已修复。请勿拿磁盘的 `.compute` 当作权威源
- **Rotation 四元数存储顺序**：PLY 里是 `rot_0..rot_3` = `w,x,y,z`；在 HLSL 里 `float4` 读进来后是 `rotation.x=w, .y=x, .z=y, .w=z`（见 `kPreprocessCS` 里 `Get3DCovariance`），和直觉反的，改 shader 时需要注意
- **`renderMode=3`** 是诊断模式：强制使用 production 路径但覆盖 radius 为固定 5 像素，用来区分"排序错"还是"协方差错"
- **Sort 准备未就绪时自动回退**：`draw()` 里 `canProd = prodReady && sortReady && inputsReady && allocatedN==vertexCount && sortValsA_SRV`，任一失败就走 debug 路径
- **PLY 切换 → 重新分配**：`allocatedN != N` 时调用 `createComputeOutputs(device, N)`，会同时重建 sort buffer（keys A/B、vals A/B、blockHist）
- **编码警告 C4819**：中文注释触发 warning，可忽略

## 菜单用法

加载插件后顶部栏出现 **"Gaussian Splatting"** 菜单：

1. **Load PLY File...**：一键完成 `createNode gaussianSplatData` → 设置 filePath → 创建 transform + gaussianSplat shape → `connectAttr outputData → inputData`
2. **Create Data Node** / **Create Render Node**：分别创建空节点，方便手动连接多个 render node 到同一 data
3. **Connect Selected**：选中 data node 再选 render node（或其 transform），点击即完成连接

也可以纯 MEL：
```mel
string $d = `createNode gaussianSplatData`;
setAttr -type "string" ($d + ".filePath") "C:/path/to/point_cloud.ply";
string $r = `createNode gaussianSplat`;
connectAttr ($d + ".outputData") ($r + ".inputData");
```
