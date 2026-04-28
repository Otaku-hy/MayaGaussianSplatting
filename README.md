# MayaGaussianSplatting

Maya 2026 plugin for real-time 3D Gaussian Splatting visualization &
editing 

---

## 1. Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Autodesk Maya | 2026 | x64 only |
| Maya Devkit   | 2026 | matching Maya version, downloaded from Autodesk |
| Visual Studio | 2022 (v143 toolset) | "Desktop development with C++" workload |
| CMake         | 3.20+ | needed for the new build system |
| Windows SDK   | 10 | comes with VS 2022 |

---

## 2. Environment variables (required)

Set these (Control Panel → System → Advanced → Environment Variables)
so both CMake and Visual Studio pick them up:

| Variable | Example value | Purpose |
|---|---|---|
| `MAYA_LOCATION`   | `C:\Program Files\Autodesk\Maya2026` | Maya install root — used for `include/` and `lib/` |
| `DEVKIT_LOCATION` | `C:\maya2026-devkit`                 | Maya devkit root — additional headers / libs |

> If either is missing, `cmake` will abort with a `FATAL_ERROR` telling you which one.

You can also pass them on the CMake command line instead:
```
cmake -DMAYA_LOCATION="C:/Program Files/Autodesk/Maya2026" -DDEVKIT_LOCATION="C:/maya2026-devkit" ...
```

---

## 3. Optional variable

| Variable | Default | Purpose |
|---|---|---|
| `PLUGIN_TARGET_PATH` | empty → build output dir | Where the `.mll` + `shaders/` get copied after every build. Set this to your Maya plugin folder so the manager can auto detect  the plugins. |

Examples:
```bash
# install to a personal Maya plugin folder
cmake -DPLUGIN_TARGET_PATH="C:/Users/<you>/Documents/maya/2026//My3DGSPlugin" ...
```

If left empty, post-build copies into the CMake build directory only. And you should put `.mod` file to your maya's module folder (See Chapter 5)

---

## 4. Configure & build

From the repo root:

```bash
# generate Visual Studio sln in ./build
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DPLUGIN_TARGET_PATH="C:/path/to/install"
```

Or open `build/GaussianSplatting.sln` in Visual Studio and build there
(`GaussianSplatting` is the default startup project).

After build, the post-build step copies:
- `GaussianSplatting.mll`
- `shaders/` directory

into `PLUGIN_TARGET_PATH`.

---

## 5. The `.mod` file

A Maya module descriptor is generated at the cmake binary folder the first time you
configure CMake (ONLY when PLUGIN_TARGET_PATH is set to default):

`3DGSPlugin_<Config>.mod`

Contents look like:
```
+ 3DGSPlugin 1.0 <plugin path>
plug-ins: <mll path>
```

Then, after each time you compile and build the project, the corresponding mod file will be copied to the project root file. 

If you do not set PLUGIN_TARGET_PATH, PLEASE copy the mod file to Maya Module directory, otherwise plugin manager cannot find our plugin.

## 6. Maya viewport setting (required)

The plugin uses DirectX 11 only. Switch Viewport 2.0 to DX11:

```
Windows → Settings/Preferences → Preferences
  → Display → Viewport 2.0
  → Rendering Engine: DirectX 11
```

Restart Maya after changing this.
