# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**MayaGaussianSplatting** is a Maya C++ plugin (`.mll`) for real-time 3D Gaussian Splatting rendering inside Autodesk Maya Viewport 2.0 (VP2). Course project for CIS 6600.

Planned pipeline:
1. Load a 3DGS `.ply` file → `PLYReader` → CPU `GaussianData`
2. Upload position + colour vertex buffers via `MPxGeometryOverride` → VP2 GPU buffers
3. **Debug pass** (current): point cloud with HLSL GS-expanded quads (`gaussianDebug.fx`)
4. **Production pass** (future): DX12 compute shader (SM 6.5) for depth-sorted alpha-blended splat rendering

## Build System

Visual Studio 2022 (MSBuild, toolset v143), Windows 10.0 SDK.

**Solution:** `GaussianSplatting/GaussianSplatting.sln`

```bash
# Debug x64
msbuild GaussianSplatting/GaussianSplatting.sln /p:Configuration=Debug /p:Platform=x64

# Release x64
msbuild GaussianSplatting/GaussianSplatting.sln /p:Configuration=Release /p:Platform=x64
```

**Required environment variables** (set per-developer, not committed):

| Variable | Example | Purpose |
|---|---|---|
| `MAYA_SDK_DIR` | `C:\Program Files\Autodesk\Maya2024` | Maya devkit include + lib paths |
| `MAYA_PLUG_IN_PATH` | `C:\Users\<user>\Documents\maya\2024\plug-ins` | Post-build copy target + Maya plugin search path |
| `GAUSSIAN_SHADER_DIR` | same as `MAYA_PLUG_IN_PATH` | Runtime shader load path for `.fx` file |

After changing environment variables, **restart Visual Studio** for them to be picked up.

## Source Layout

```
GaussianSplatting/
├── src/
│   ├── plugin_main.cpp               # initializePlugin / uninitializePlugin
│   ├── GaussianData.h                # GaussianSplat struct + GaussianData (CPU + flattened GPU arrays)
│   ├── PLYReader.h / PLYReader.cpp   # Binary-LE and ASCII PLY parser
│   ├── GaussianNode.h / .cpp         # MPxLocatorNode  (filePath, pointSize attributes)
│   └── GaussianGeometryOverride.h / .cpp  # MPxGeometryOverride – VP2 render items + buffer upload
└── shaders/
    └── gaussianDebug.fx              # HLSL DX11: VS→GS(point→quad)→PS(circle clip + SH colour)
```

## Architecture

**Node registration:** `GaussianNode` (type id `0x00127A00`) is a `kLocatorNode`. Its draw classification `drawdb/geometry/gaussianSplat` links it to `GaussianGeometryOverride`.

**Data flow per frame:**
```
GaussianGeometryOverride::updateDG()
    └─ reads filePath plug → PLYReader::read() if path changed
       → fills GaussianData.splats + flattened positions[] / colors[]

GaussianGeometryOverride::updateRenderItems()
    └─ ensures "gaussianDebugPoints" NonShadedItem exists
    └─ sets gPointSize + gViewportSize shader parameters

GaussianGeometryOverride::populateGeometry()
    └─ iterates MGeometryRequirements, fills kPosition (float3) + kColor (float4) buffers
    └─ sequential index buffer 0..N-1
```

**Colour conversion** (in `PLYReader.cpp`):
```
rgb  = clamp(0.5 + 0.2821 * f_dc,  0, 1)   // SH degree-0 → linear colour
alpha = sigmoid(raw_opacity)                  // logit → [0,1]
```

**Shader** (`gaussianDebug.fx`) uses a Geometry Shader to expand each point into a screen-aligned quad sized by `gPointSize / gViewportSize`, then clips to a circle in the PS. This is DX11-only; OpenGL support requires a separate `.ogsfx` variant.

## Loading the Plugin in Maya

```mel
// MEL
loadPlugin "GaussianSplatting.mll";
string $node = `createNode gaussianSplat`;
setAttr -type "string" ($node + ".filePath") "C:/data/scene.ply";
setAttr ($node + ".pointSize") 4.0;
```
