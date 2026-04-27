# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目简介

Maya 插件（C++/DX11），在 Autodesk Maya 2026 的 Viewport 2.0 中可视化、交互编辑 3D Gaussian Splatting 点云。

**当前版本：v0.3**

已实现功能：
- 完整生产级渲染管线：**Compute 预处理 → GPU Radix 排序 → 实例化椭圆 Splat 渲染**
- Debug 路径（简单圆点）作为回退
- Maya 几何体深度遮挡（GS ↔ Maya 几何体互相正确遮挡）
- 完整 Maya 变换（移动/旋转/缩放）
- **Marquee 框选系统**：VP2.0 context，框选 splat，soft-delete，Restore All，Save PLY

## 构建

**工具链：** Visual Studio 2022（v143），MSBuild，仅 x64。无自动化测试或 lint。

**构建前必须设置的环境变量：**
```
MAYA_LOCATION    — Maya 安装目录，例：C:\Program Files\Autodesk\Maya2026
DEVKIT_LOCATION  — Maya devkit 根目录
```

**解决方案：** `GaussianSplatting/GaussianSplatting.sln`

**本地构建脚本：** 仓库根目录 `_build_tmp.bat`（设置环境变量 + 调用 MSBuild Release|x64）。

**Post-build：** 项目把 `GaussianSplatting.mll` 复制到仓库根目录（`$(SolutionDir)..\`）。

**Maya 设置：** Viewport 2.0 必须切换为 DirectX 11：
`Windows → Settings/Preferences → Preferences → Display → Viewport 2.0 → Rendering Engine: DirectX 11`

## 架构

### 节点设计（v0.3 — 自包含单节点）

```
  gaussianSplat (MPxLocatorNode)
  ──────────────────────────────────────────
  filePath  → compute() 时触发 PLYReader
  dataReady (hidden bool output)
  pointSize (float, debug 圆点半径)
  renderMode (int 0-3)

  内部持有：
    GaussianData (CPU 原始数据)
    5 个输入 StructuredBuffer SRV
    1 个选择 mask Buffer (SRV+UAV)
    CPU mask shadow (std::vector<uint32_t>)
```

**v0.3 重要变化**：原来的 `gaussianSplatData`（MPxNode）已被**合并进 `gaussianSplat`**，不再存在。每个 `gaussianSplat` 节点完全自包含，互相独立，不可能意外共享数据。加载同一 PLY 两次 = 两个独立节点，各有独立 mask。

### 数据流

```
PLY 文件 ──▶ PLYReader ──▶ GaussianData (GaussianNode 内部)
                                │
                                ▼
                      GaussianNode::compute() (DG 触发，filePath 变化时)
                                │  uploadInputBuffersIfNeeded
                                ▼
         5 个 StructuredBuffer SRV + 1 个 selection mask buffer
                                │
                                ▼
              GaussianDrawOverride::prepareForDraw
                                │  触发 compute；上传 GPU buffer；
                                │  注册到 GaussianRenderManager
                                ▼
                  GaussianDrawOverride::draw (DX11)
                      ┌──────────┴──────────┐
            Production 路径              Debug 路径
          (Merged Preprocess+Sort+       (VS+GS+PS 圆点,
           Ellipse instanced draw)        per-node SRV)
```

### 核心模块（`GaussianSplatting/src/`）

| 文件 | 说明 |
|---|---|
| `plugin_main.cpp` | 注册 **1 个节点** + DrawOverride + 5 个命令 + context 命令；内嵌 MEL 构建菜单和 AE 模板 |
| `GaussianNode.{h,cpp}` | **自包含** MPxLocatorNode。持有 GaussianData、5 个输入 SRV、selection mask buffer。`compute()` 在 filePath 变化时加载 PLY；所有 GPU buffer 管理、mask 操作都在这里 |
| `GaussianDataNode.{h,cpp}` | **已废弃**，保留为空 stub，不注册任何节点 |
| `GaussianDrawOverride.{h,cpp}` | 渲染路径。`prepareForDraw` 直接从 `m_node`（GaussianNode）取 SRV；注册到 RenderManager |
| `GaussianRenderManager.{h,cpp}` | 单例，合并多个实例的 buffer 并执行一次 preprocess+sort+render。`RenderInstance.node` 是 `GaussianNode*` |
| `GaussianSelection.{h,cpp}` | 5 个 Maya 命令（`gsMarqueeSelect`/`gsClearSelection`/`gsDeleteSelected`/`gsRestoreAll`/`gsSavePLY`）+ `GSMarqueeContext`（VP2.0 框选工具） |
| `PLYReader.{h,cpp}` | binary_little_endian + ASCII；exp(log_scale)；四元数归一化；SH planar→interleaved |
| `GaussianData.h` | POD：`GaussianSplat` 原始数据 + `GaussianData` 展平数组 + bbox；`kMaskBitSelected=1, kMaskBitDeleted=2` |

### 关键点：所有 HLSL 内嵌在 GaussianRenderManager.cpp / GaussianDrawOverride.cpp

| 变量 | 位置 | 内容 |
|---|---|---|
| `kMergedPreprocessCS` | GaussianRenderManager.cpp | 合并多实例的预处理 CS（读 instanceID + worldMats）|
| `kProdShaderSrc` | GaussianRenderManager.cpp | Production VS+PS（实例化椭圆）|
| `kRadixSortCS` | GaussianRenderManager.cpp | 4 个 radix sort kernel |
| `kDbgShaderSrc` | GaussianDrawOverride.cpp | Debug VS+GS+PS（圆点）|

**`shaders/` 目录下的文件是遗留参考，不参与构建。** 修改 shader 必须编辑上述 C++ 字符串常量。

## 渲染管线细节

### Production 路径（renderMode=0 auto 或 =2）

每帧 `GaussianRenderManager::render()` 执行：

1. **buildMergedInputs**：将所有 RenderInstance 的 CPU 数据拼接上传为合并 buffer（位置/scale/rotation/opacity/SH/instanceID/worldMats）。有 signature 缓存，只在实例集变化时重建。
2. **updateMergedSelection**：按 maskVersion 检测变化，重建合并 selection mask buffer。
3. **Dispatch PreprocessKernel**（256线程/组）：per-splat 通过 instanceID 查 worldMat，做 object→world→view→clip，计算 2D 协方差，EWA radius，SH 颜色。删除位(bit1)=1 的 splat 输出 radius=0 跳过。
4. **GPU Radix Sort**（4 趟 8-bit，key=~depth 降序=back-to-front）
5. **DrawInstanced(4, N)**：TRIANGLESTRIP，VS 通过 SV_InstanceID 查排序索引，读 compute 输出 SRV。PS 做 Gaussian 椭圆 alpha。

### Debug 路径（renderMode=1 或 production 失败时自动回退）

per-node，直接读节点自己的 SRV（positionWS/opacity/SHCoeffs），VS+GS+PS，圆形 splat。

### Maya 集成要点

- **`isTransparent() = true`**：splat 在透明 pass 绘制，Maya 不透明几何体深度已在 depth buffer → 自动互相遮挡
- **变换**：`objPath.inclusiveMatrix()` → worldMat → 传给 compute shader；协方差按 `W^T * cov_obj * W` 变换
- **包围盒**：`GaussianNode::boundingBox()` 返回 PLY 加载时计算的 object-space AABB
- **DX11 状态保存/恢复**：`draw()` 开头保存 Maya 的 blend/RS/DS/InputLayout/VS/GS/PS，结尾全部恢复
- **实例 dedup**：`GaussianRenderManager::registerInstance` 按 `node` 指针去重，防止 prepareForDraw 多次调用导致计数膨胀

## 选择与编辑系统

### Marquee Context（GSMarqueeContext）

**关键架构**：Maya VP2.0（DX11 模式）**只调用 3 参数版本**的 `doPress/doDrag/doRelease`（带 `MUIDrawManager`）。1 参数 legacy 版本在 VP2.0 下不会被调用。`drawFeedback()` 每帧绘制黄色矩形。

```
doPress (3-arg)   → 存储 m_x0, m_y0；m_dragging=true
doDrag  (3-arg)   → 更新 m_x1, m_y1（绘制在 drawFeedback 里）
doRelease (3-arg) → 调用 runSelectionFromRect()
drawFeedback      → 如果 m_dragging，用 MUIDrawManager 画黄框

drawFeedback 用 MUIDrawManager::rect2d(center, MVector(0,1,0), hw, hh, false)
```

**激活 context 必须 delete 再 recreate**（防止 plugin reload 后旧 context 残留）：
```mel
if (`contextInfo -exists gsMarqueeCtx1`) deleteUI gsMarqueeCtx1;
gsMarqueeCtx gsMarqueeCtx1;
setToolTo gsMarqueeCtx1;
```

### 选择范围（collectRenderPairs）

1. 检查 Maya 当前选中的节点中是否有 `gaussianSplat` 类型（直接选 shape 或选 transform 下的 shape）
2. 有 → 只对选中的节点做框选
3. 无 → fall back 到场景中所有 `gaussianSplat` 节点

### CPU 选择逻辑（runSelection）

无 GPU CS，无 readback stall。纯 CPU：
1. 计算 `wvp = worldMat * viewProj`（row-major 矩阵乘法）
2. 每个 splat 做 `posNDC = pos * wvp / w`
3. 测试是否落在框选 NDC rect 内
4. 按 mode（replace/add/subtract/toggle）更新 `m_maskShadow`
5. `UpdateSubresource` 把 CPU shadow 写回 GPU buffer
6. `markMaskChanged()` 使 RenderManager 下一帧重建合并 selection

### Mask Buffer 格式

每 splat 一个 `uint32_t`：
- `bit 0 (kMaskBitSelected=1)` = 选中（黄色高亮）
- `bit 1 (kMaskBitDeleted=2)` = 软删除（不渲染）

preprocess CS 里：`if (mask & 2u) { gRadius[id.x] = 0.0f; return; }`

### 命令列表

| 命令 | 说明 |
|---|---|
| `gsMarqueeSelect -min x0 y0 -max x1 y1 -mo mode` | MEL 脚本框选（NDC 坐标） |
| `gsClearSelection` | 清除所有节点的选中位 |
| `gsDeleteSelected` | 软删除选中 splat（所有节点） |
| `gsRestoreAll [-node name]` | 清除 mask（可指定单个节点） |
| `gsSavePLY -file path [-node name]` | 导出非删除 splat 为 binary PLY |

## 常量与 Magic Number 速查

| 名称 | 值 | 说明 |
|---|---|---|
| `kSHCoeffsPerSplat` | 16 | 每 splat 16 组 float3 SH（0–3 阶）|
| `kMaskBitSelected` | 1 | mask bit 0 |
| `kMaskBitDeleted` | 2 | mask bit 1 |
| `kSortGroupSize` | 256 | Radix sort 线程组大小 |
| `kSortItemsPerThread` | 16 | 每线程元素数 |
| `kSortTileSize` | 4096 | = 256×16，每 block 元素数 |
| `kRadixSize` | 256 | 8-bit radix 桶数 |
| PreprocessKernel 组大小 | 256 | `⌈N/256⌉` dispatch |
| Radius 上限 | 1024 px | 超过直接剔除 |
| Z 剔除阈值 | `posVS.z >= -0.2` | 近平面之后剔除 |

## 常见陷阱

- **"splats 都是 1m 半径"**：PLY 缺少 `scale_0/1/2`（COLMAP sparse 点云）。用 3DGS 训练输出的 `point_cloud.ply`，不是 COLMAP 的 `points3D`
- **Rotation 四元数顺序**：PLY `rot_0..3 = w,x,y,z`；HLSL float4 读进来 `.x=w .y=x .z=y .w=z`
- **`renderMode=3`**：诊断模式，强制 production 路径但 radius 固定 5px，用于区分排序错 vs 协方差错
- **VP2.0 context 1-arg 不被调用**：`doPress(MEvent&)` 等在 VP2.0/DX11 下不触发，必须重写 3-arg 版本
- **Context reload 残留**：`unloadPlugin` 不销毁已创建的 context 对象，必须在激活前 `deleteUI gsMarqueeCtx1` 再重建
- **AE callCustom 参数格式**：`editorTemplate -callCustom "proc1" "proc2" ""` 传空串；proc 收到 `nodeName.`，用 `stringToStringArray($arg, ".")[0]` 提取节点名
- **编码警告 C4819**：中文注释触发，可忽略

## 菜单用法

加载插件后顶部栏出现 **"Gaussian Splatting"** 菜单：

1. **Load PLY File...**：创建独立的 `gaussianSplat` 节点并加载 PLY（每次加载都是独立副本）
2. **Create Gaussian Splat Node**：创建空节点，在 AE 里设置 filePath
3. **Marquee Select Tool**：激活框选工具；如果已选中某个 gaussianSplat 节点，只对该节点生效
4. **Clear Selection / Delete Selected / Restore All**：选择编辑操作
5. **Save PLY As...**：导出当前选中节点（或第一个节点）的非删除 splat

**Attribute Editor**（选中 gaussianSplat 节点）：
- `filePath`：直接在 AE 里修改即可切换 PLY
- `pointSize` / `renderMode`：debug 参数
- **Restore All** / **Delete Selected** / **Save PLY As...**：针对当前节点的编辑按钮

纯 MEL 用法：
```mel
string $t = `createNode transform -name "gaussianSplat1"`;
string $n = `createNode gaussianSplat -parent $t`;
setAttr -type "string" ($n + ".filePath") "C:/path/to/point_cloud.ply";
```
