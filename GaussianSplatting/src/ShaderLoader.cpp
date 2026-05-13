#include "ShaderLoader.h"

#include <maya/MGlobal.h>

#include <cstdlib>
#include <fstream>
#include <sstream>

namespace gs {

namespace {
    std::string g_pluginDir;

    bool tryRead(const std::string& path, std::string& out) {
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open()) return false;
        std::ostringstream ss;
        ss << f.rdbuf();
        out = ss.str();
        return !out.empty();
    }
}

void SetPluginDir(const std::string& dir) {
    g_pluginDir = dir;
}

const std::string& PluginDir() { return g_pluginDir; }

std::string LoadShader(const char* nameDotHlsl) {
    std::string out;

    std::string envDir;
    if (const char* e = std::getenv("GAUSSIAN_SHADER_DIR")) envDir = e;

    std::string p1, p2;
    if (!envDir.empty()) {
        p1 = envDir;
        if (p1.back() != '/' && p1.back() != '\\') p1 += '/';
        p1 += nameDotHlsl;
        if (tryRead(p1, out)) return out;
    }
    if (!g_pluginDir.empty()) {
        p2 = g_pluginDir;
        if (p2.back() != '/' && p2.back() != '\\') p2 += '/';
        p2 += "shaders/";
        p2 += nameDotHlsl;
        if (tryRead(p2, out)) return out;
    }

    MString msg = MString("[GaussianSplat] Failed to load shader '") + nameDotHlsl + "'.";
    if (!p1.empty()) msg += MString("  Tried: ") + p1.c_str();
    if (!p2.empty()) msg += MString("  Tried: ") + p2.c_str();
    if (g_pluginDir.empty() && envDir.empty())
        msg += "  (plugin dir not set and GAUSSIAN_SHADER_DIR not defined)";
    MGlobal::displayError(msg);
    return {};
}

} // namespace gs
