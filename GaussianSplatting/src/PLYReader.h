#pragma once
#include "GaussianData.h"
#include <string>

class PLYReader {
public:
    // Reads a 3DGS PLY file (binary_little_endian or ASCII).
    // Returns true on success; sets errorMsg on failure.
    static bool read(const std::string& filepath,
                     GaussianData&      outData,
                     std::string&       errorMsg);
};
