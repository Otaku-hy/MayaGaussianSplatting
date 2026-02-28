#pragma once
#include <vector>
#include <cstddef>

// Raw per-splat data as parsed from PLY
struct GaussianSplat {
    float position[3];  // x, y, z
    float f_dc[3];      // SH degree-0 color coefficients
    float opacity;      // raw logit opacity
    float scale[3];     // log scale (unused in debug pass but parsed)
    float rotation[4];  // quaternion w,x,y,z (unused in debug pass but parsed)
};

// CPU-side container + flattened GPU-ready arrays
struct GaussianData {
    std::vector<GaussianSplat> splats;

    // Flattened for upload: positions = [x0,y0,z0, x1,y1,z1, ...]
    std::vector<float> positions;
    // colors = [r0,g0,b0,a0, r1,g1,b1,a1, ...]  (a = sigmoid(opacity))
    std::vector<float> colors;

    size_t count() const { return splats.size(); }
    bool   empty() const { return splats.empty(); }

    // Rebuild flattened arrays from splats (call after loading)
    void buildGPUArrays();
    void clear();
};
