#include "PLYReader.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstring>

// ---------------------------------------------------------------------------
// Colour helpers
// ---------------------------------------------------------------------------
static constexpr float kSH_C0 = 0.28209479177387814f;   // 1 / (2*sqrt(pi))

static inline float shToLinear(float sh) {
    float v = 0.5f + kSH_C0 * sh;
    return v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
}

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// ---------------------------------------------------------------------------
// GaussianData helpers
// ---------------------------------------------------------------------------
void GaussianData::buildGPUArrays() {
    positions.clear();
    colors.clear();
    positions.reserve(splats.size() * 3);
    colors.reserve(splats.size() * 4);

    for (const auto& s : splats) {
        positions.push_back(s.position[0]);
        positions.push_back(s.position[1]);
        positions.push_back(s.position[2]);

        colors.push_back(shToLinear(s.f_dc[0]));
        colors.push_back(shToLinear(s.f_dc[1]));
        colors.push_back(shToLinear(s.f_dc[2]));
        colors.push_back(sigmoid(s.opacity));
    }
}

void GaussianData::clear() {
    splats.clear();
    positions.clear();
    colors.clear();
}

// ---------------------------------------------------------------------------
// PLY format
// ---------------------------------------------------------------------------
enum class PLYFormat { ASCII, BinaryLE, Unknown };

struct PropDef {
    std::string name;
    int  byteSize = 4;
    bool isFloat  = false;
};

// ---------------------------------------------------------------------------
// PLYReader::read
// ---------------------------------------------------------------------------
bool PLYReader::read(const std::string& filepath,
                     GaussianData&      outData,
                     std::string&       errorMsg)
{
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        errorMsg = "Cannot open: " + filepath;
        return false;
    }

    // ---- parse header ----
    {
        std::string line;
        if (!std::getline(file, line) || line.find("ply") == std::string::npos) {
            errorMsg = "Not a PLY file";
            return false;
        }
    }

    PLYFormat format       = PLYFormat::Unknown;
    int       vertexCount  = 0;
    std::vector<PropDef> props;
    bool inVertexElem = false;

    for (std::string line; std::getline(file, line); ) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line == "end_header") break;

        std::istringstream ss(line);
        std::string tok;
        ss >> tok;

        if (tok == "format") {
            std::string fmt; ss >> fmt;
            if      (fmt == "ascii")                 format = PLYFormat::ASCII;
            else if (fmt == "binary_little_endian")  format = PLYFormat::BinaryLE;

        } else if (tok == "element") {
            std::string name; ss >> name;
            inVertexElem = (name == "vertex");
            if (inVertexElem) ss >> vertexCount;

        } else if (tok == "property" && inVertexElem) {
            PropDef p;
            std::string typeName;
            ss >> typeName >> p.name;
            if      (typeName == "float"   || typeName == "float32") { p.byteSize = 4; p.isFloat = true; }
            else if (typeName == "double"  || typeName == "float64") { p.byteSize = 8; p.isFloat = false; }
            else if (typeName == "uchar"   || typeName == "uint8")   { p.byteSize = 1; p.isFloat = false; }
            else if (typeName == "int"     || typeName == "int32"
                  || typeName == "uint"    || typeName == "uint32")  { p.byteSize = 4; p.isFloat = false; }
            else                                                      { p.byteSize = 4; p.isFloat = false; }
            props.push_back(p);
        }
    }

    if (format == PLYFormat::Unknown) { errorMsg = "Unknown PLY format";          return false; }
    if (vertexCount <= 0)             { errorMsg = "No vertices in PLY";           return false; }

    // ---- build property index map ----
    auto find = [&](const char* name) -> int {
        for (int i = 0; i < (int)props.size(); i++)
            if (props[i].name == name) return i;
        return -1;
    };

    int iX  = find("x"),      iY   = find("y"),     iZ  = find("z");
    int iR  = find("f_dc_0"), iG   = find("f_dc_1"), iB = find("f_dc_2");
    int iOp = find("opacity");

    if (iX < 0 || iY < 0 || iZ < 0) {
        errorMsg = "PLY missing position properties (x/y/z)";
        return false;
    }

    // byte offsets per property
    std::vector<int> offsets(props.size(), 0);
    int rowBytes = 0;
    for (int i = 0; i < (int)props.size(); i++) {
        offsets[i] = rowBytes;
        rowBytes   += props[i].byteSize;
    }

    // ---- helper: read float from a raw row buffer ----
    auto getf = [&](const char* row, int idx) -> float {
        if (idx < 0 || !props[idx].isFloat) return 0.f;
        float v;
        std::memcpy(&v, row + offsets[idx], 4);
        return v;
    };

    // ---- read vertices ----
    outData.clear();
    outData.splats.resize(vertexCount);

    if (format == PLYFormat::BinaryLE) {
        std::vector<char> row(rowBytes);
        for (int i = 0; i < vertexCount; i++) {
            file.read(row.data(), rowBytes);
            if (file.fail()) { errorMsg = "Unexpected EOF in binary data"; return false; }
            GaussianSplat& s = outData.splats[i];
            s.position[0] = getf(row.data(), iX);
            s.position[1] = getf(row.data(), iY);
            s.position[2] = getf(row.data(), iZ);
            s.f_dc[0]     = getf(row.data(), iR);
            s.f_dc[1]     = getf(row.data(), iG);
            s.f_dc[2]     = getf(row.data(), iB);
            s.opacity     = getf(row.data(), iOp);
        }
    } else { // ASCII
        for (int i = 0; i < vertexCount; i++) {
            std::string line;
            if (!std::getline(file, line)) { errorMsg = "Unexpected EOF in ASCII data"; return false; }
            if (!line.empty() && line.back() == '\r') line.pop_back();
            std::istringstream ss(line);
            std::vector<float> vals(props.size(), 0.f);
            for (auto& v : vals) ss >> v;
            GaussianSplat& s = outData.splats[i];
            s.position[0] = iX  >= 0 ? vals[iX]  : 0.f;
            s.position[1] = iY  >= 0 ? vals[iY]  : 0.f;
            s.position[2] = iZ  >= 0 ? vals[iZ]  : 0.f;
            s.f_dc[0]     = iR  >= 0 ? vals[iR]  : 0.f;
            s.f_dc[1]     = iG  >= 0 ? vals[iG]  : 0.f;
            s.f_dc[2]     = iB  >= 0 ? vals[iB]  : 0.f;
            s.opacity     = iOp >= 0 ? vals[iOp] : 0.f;
        }
    }

    outData.buildGPUArrays();
    return true;
}
