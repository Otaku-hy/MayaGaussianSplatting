#pragma once
#include <string>

// ===========================================================================
// ShaderLoader  --  loads HLSL source files from disk at runtime.
//
// Search order for "<name>.hlsl":
//   1. $GAUSSIAN_SHADER_DIR/<name>.hlsl   (env var override, dev-time hot edit)
//   2. <pluginDir>/shaders/<name>.hlsl    (deployed copy, the normal case)
//
// pluginDir is captured once in initializePlugin via SetPluginDir().
// ===========================================================================
namespace gs {

void        SetPluginDir(const std::string& dir);
const std::string& PluginDir();

// Returns full file contents. On failure returns empty string and logs an
// error via MGlobal::displayError describing where it looked.
std::string LoadShader(const char* nameDotHlsl);

} // namespace gs
