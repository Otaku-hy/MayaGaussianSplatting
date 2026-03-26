#include "PLYReader.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>

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

static inline float quatLen(const float q[4]) {
    return std::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
}

// ---------------------------------------------------------------------------
// GaussianData helpers
// ---------------------------------------------------------------------------
void GaussianData::buildGPUArrays() {
    const size_t N = splats.size();

    bboxMin[0] = bboxMin[1] = bboxMin[2] =  1e30f;
    bboxMax[0] = bboxMax[1] = bboxMax[2] = -1e30f;

    positions.clear();   positions.reserve(N * 3);
    colors.clear();      colors.reserve(N * 4);
    scaleWS.clear();     scaleWS.reserve(N * 3);
    rotationWS.clear();  rotationWS.reserve(N * 4);
    opacityRaw.clear();  opacityRaw.reserve(N);
    shCoeffs.clear();    shCoeffs.reserve(N * kSHCoeffsPerSplat * 3);

    for (const auto& s : splats) {
        // debug pass
        positions.push_back(s.position[0]);
        positions.push_back(s.position[1]);
        positions.push_back(s.position[2]);

        for (int k = 0; k < 3; ++k) {
            if (s.position[k] < bboxMin[k]) bboxMin[k] = s.position[k];
            if (s.position[k] > bboxMax[k]) bboxMax[k] = s.position[k];
        }

        colors.push_back(shToLinear(s.f_dc[0]));
        colors.push_back(shToLinear(s.f_dc[1]));
        colors.push_back(shToLinear(s.f_dc[2]));
        colors.push_back(sigmoid(s.opacity));

        // compute pass — scale: exp(log_scale)
        scaleWS.push_back(std::exp(s.scale[0]));
        scaleWS.push_back(std::exp(s.scale[1]));
        scaleWS.push_back(std::exp(s.scale[2]));

        // compute pass — rotation: normalised quaternion
        float len = quatLen(s.rotation);
        if (len < 1e-6f) len = 1.f;
        rotationWS.push_back(s.rotation[0] / len);
        rotationWS.push_back(s.rotation[1] / len);
        rotationWS.push_back(s.rotation[2] / len);
        rotationWS.push_back(s.rotation[3] / len);

        // compute pass — raw logit opacity
        opacityRaw.push_back(s.opacity);

        // compute pass — SH coefficients (16 float3 per splat)
        // group 0: f_dc
        shCoeffs.push_back(s.f_dc[0]);
        shCoeffs.push_back(s.f_dc[1]);
        shCoeffs.push_back(s.f_dc[2]);
        // groups 1..15: f_rest (45 floats = 15 groups × 3 channels)
        // f_rest is stored as all-red, then all-green, then all-blue in the PLY.
        // Re-interleave to float3 groups expected by the shader (r,g,b per group).
        // f_rest[0..14]  = red   for groups 1..15
        // f_rest[15..29] = green for groups 1..15
        // f_rest[30..44] = blue  for groups 1..15
        for (int g = 0; g < 15; ++g) {
            shCoeffs.push_back(s.f_rest[g]);        // red   channel, group g+1
            shCoeffs.push_back(s.f_rest[g + 15]);   // green channel, group g+1
            shCoeffs.push_back(s.f_rest[g + 30]);   // blue  channel, group g+1
        }
    }
}

void GaussianData::clear() {
    splats.clear();
    positions.clear();
    colors.clear();
    scaleWS.clear();
    rotationWS.clear();
    opacityRaw.clear();
    shCoeffs.clear();
}

// ---------------------------------------------------------------------------
// PLY format
// ---------------------------------------------------------------------------
enum class PLYFormat { ASCII, BinaryLE, Unknown };

struct PropDef {
    std::string name;
    std::string typeName;
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
            ss >> p.typeName >> p.name;
            if      (p.typeName == "float"   || p.typeName == "float32") { p.byteSize = 4; p.isFloat = true; }
            else if (p.typeName == "double"  || p.typeName == "float64") { p.byteSize = 8; p.isFloat = true; }
            else if (p.typeName == "uchar"   || p.typeName == "uint8")   { p.byteSize = 1; p.isFloat = false; }
            else if (p.typeName == "int"     || p.typeName == "int32"
                  || p.typeName == "uint"    || p.typeName == "uint32")  { p.byteSize = 4; p.isFloat = false; }
            else                                                          { p.byteSize = 4; p.isFloat = false; }
            props.push_back(p);
        }
    }

    if (format == PLYFormat::Unknown) { errorMsg = "Unknown PLY format";  return false; }
    if (vertexCount <= 0)             { errorMsg = "No vertices in PLY";   return false; }

    // ---- build property index map ----
    auto findProp = [&](const char* name) -> int {
        for (int i = 0; i < (int)props.size(); i++)
            if (props[i].name == name) return i;
        return -1;
    };

    int iX  = findProp("x"),      iY   = findProp("y"),     iZ  = findProp("z");
    int iR  = findProp("f_dc_0"), iG   = findProp("f_dc_1"), iB = findProp("f_dc_2");
    int iOp = findProp("opacity");

    // Fallback: uint8 red/green/blue when SH DC coefficients are absent
    int iRed = findProp("red"), iGreen = findProp("green"), iBlue = findProp("blue");
    bool useRGBFallback = (iR < 0 || iG < 0 || iB < 0) &&
                          (iRed >= 0 && iGreen >= 0 && iBlue >= 0);
    int iSX = findProp("scale_0"), iSY = findProp("scale_1"), iSZ = findProp("scale_2");
    int iRW = findProp("rot_0"),   iRX = findProp("rot_1"),
        iRY = findProp("rot_2"),   iRZ = findProp("rot_3");

    // f_rest_0 .. f_rest_44
    int iRest[45];
    for (int i = 0; i < 45; ++i) {
        char buf[16];
        std::snprintf(buf, sizeof(buf), "f_rest_%d", i);
        iRest[i] = findProp(buf);
    }

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
        if (props[idx].byteSize == 8) {          // double / float64
            double d;
            std::memcpy(&d, row + offsets[idx], 8);
            return static_cast<float>(d);
        }
        float v;
        std::memcpy(&v, row + offsets[idx], 4);
        return v;
    };

    // ---- helper: read uint8 from a raw row buffer ----
    auto getu8 = [&](const char* row, int idx) -> uint8_t {
        if (idx < 0) return 0;
        uint8_t v;
        std::memcpy(&v, row + offsets[idx], 1);
        return v;
    };

    // ---- diagnostic: log property discovery ----
    {
        auto propInfo = [&](int idx) -> std::string {
            if (idx < 0) return "MISSING";
            return props[idx].typeName + " (byte " + std::to_string(offsets[idx]) + ")";
        };
        fprintf(stderr, "[PLYReader] %d vertices, %d properties, %d bytes/row\n",
                vertexCount, (int)props.size(), rowBytes);
        fprintf(stderr, "[PLYReader] x=%s  y=%s  z=%s\n",
                propInfo(iX).c_str(), propInfo(iY).c_str(), propInfo(iZ).c_str());
        fprintf(stderr, "[PLYReader] f_dc_0=%s  f_dc_1=%s  f_dc_2=%s\n",
                propInfo(iR).c_str(), propInfo(iG).c_str(), propInfo(iB).c_str());
        fprintf(stderr, "[PLYReader] red=%s  green=%s  blue=%s  useRGBFallback=%d\n",
                propInfo(iRed).c_str(), propInfo(iGreen).c_str(), propInfo(iBlue).c_str(),
                (int)useRGBFallback);
        fprintf(stderr, "[PLYReader] opacity=%s  scale_0=%s  rot_0=%s\n",
                propInfo(iOp).c_str(), propInfo(iSX).c_str(), propInfo(iRW).c_str());
    }

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
            if (useRGBFallback) {
                // Convert uint8 [0,255] → linear [0,1] → SH DC space
                s.f_dc[0] = (getu8(row.data(), iRed)   / 255.f - 0.5f) / kSH_C0;
                s.f_dc[1] = (getu8(row.data(), iGreen) / 255.f - 0.5f) / kSH_C0;
                s.f_dc[2] = (getu8(row.data(), iBlue)  / 255.f - 0.5f) / kSH_C0;
            } else {
                s.f_dc[0] = getf(row.data(), iR);
                s.f_dc[1] = getf(row.data(), iG);
                s.f_dc[2] = getf(row.data(), iB);
            }
            s.opacity     = getf(row.data(), iOp);
            s.scale[0]    = getf(row.data(), iSX);
            s.scale[1]    = getf(row.data(), iSY);
            s.scale[2]    = getf(row.data(), iSZ);
            s.rotation[0] = getf(row.data(), iRW);
            s.rotation[1] = getf(row.data(), iRX);
            s.rotation[2] = getf(row.data(), iRY);
            s.rotation[3] = getf(row.data(), iRZ);
            for (int j = 0; j < 45; ++j)
                s.f_rest[j] = getf(row.data(), iRest[j]);
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
            if (useRGBFallback) {
                s.f_dc[0] = (vals[iRed]   / 255.f - 0.5f) / kSH_C0;
                s.f_dc[1] = (vals[iGreen] / 255.f - 0.5f) / kSH_C0;
                s.f_dc[2] = (vals[iBlue]  / 255.f - 0.5f) / kSH_C0;
            } else {
                s.f_dc[0] = iR >= 0 ? vals[iR] : 0.f;
                s.f_dc[1] = iG >= 0 ? vals[iG] : 0.f;
                s.f_dc[2] = iB >= 0 ? vals[iB] : 0.f;
            }
            s.opacity     = iOp >= 0 ? vals[iOp] : 0.f;
            s.scale[0]    = iSX >= 0 ? vals[iSX] : 0.f;
            s.scale[1]    = iSY >= 0 ? vals[iSY] : 0.f;
            s.scale[2]    = iSZ >= 0 ? vals[iSZ] : 0.f;
            s.rotation[0] = iRW >= 0 ? vals[iRW] : 1.f;
            s.rotation[1] = iRX >= 0 ? vals[iRX] : 0.f;
            s.rotation[2] = iRY >= 0 ? vals[iRY] : 0.f;
            s.rotation[3] = iRZ >= 0 ? vals[iRZ] : 0.f;
            for (int j = 0; j < 45; ++j)
                s.f_rest[j] = iRest[j] >= 0 ? vals[iRest[j]] : 0.f;
        }
    }

    // ---- diagnostic: sample first few splats ----
    {
        int nSample = std::min(5, vertexCount);
        for (int i = 0; i < nSample; ++i) {
            const auto& s = outData.splats[i];
            fprintf(stderr, "[PLYReader] splat[%d] pos=(%.3f,%.3f,%.3f) scale=(%.4f,%.4f,%.4f) "
                    "opacity=%.3f rot=(%.3f,%.3f,%.3f,%.3f) f_dc=(%.3f,%.3f,%.3f)\n",
                    i, s.position[0], s.position[1], s.position[2],
                    s.scale[0], s.scale[1], s.scale[2],
                    s.opacity,
                    s.rotation[0], s.rotation[1], s.rotation[2], s.rotation[3],
                    s.f_dc[0], s.f_dc[1], s.f_dc[2]);
        }
        // Scale statistics
        float sMin = 1e30f, sMax = -1e30f, sSum = 0.f;
        for (int i = 0; i < vertexCount; ++i) {
            for (int j = 0; j < 3; ++j) {
                float v = outData.splats[i].scale[j];
                sMin = std::min(sMin, v);
                sMax = std::max(sMax, v);
                sSum += v;
            }
        }
        fprintf(stderr, "[PLYReader] scale stats: min=%.4f max=%.4f avg=%.4f\n",
                sMin, sMax, sSum / (vertexCount * 3.0f));
        fprintf(stderr, "[PLYReader] If scale values are positive (e.g. 0.001~1.0), "
                "PLY may store LINEAR scale (not log-scale).\n");
    }

    outData.buildGPUArrays();
    return true;
}
