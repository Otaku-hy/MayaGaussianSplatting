#pragma once
#include <vector>
#include <cstddef>

// Number of SH coefficients per channel stored per splat (degree 0..3 = 16 groups)
static constexpr int kSHCoeffsPerSplat = 16;

// Raw per-splat data as parsed from PLY
struct GaussianSplat {
    float position[3];  // x, y, z  (world space)
    float f_dc[3];      // SH degree-0 color coefficients
    float f_rest[45];   // SH degree 1-3 rest coefficients (15 groups × 3 channels); 0 if absent
    float opacity;      // raw logit opacity
    float scale[3];     // log-scale (exp needed before use)
    float rotation[4];  // quaternion w,x,y,z
};

// CPU-side container + flattened GPU-ready arrays
struct GaussianData {
    std::vector<GaussianSplat> splats;

    // --- debug pass arrays (positions + RGBA color, flat) ---
    // positions = [x0,y0,z0, x1,y1,z1, ...]
    std::vector<float> positions;
    // colors    = [r0,g0,b0,a0, r1,g1,b1,a1, ...]  (a = sigmoid(opacity))
    std::vector<float> colors;

    // --- compute-shader / production pass arrays ---
    // Each is tightly packed, N elements, uploaded as StructuredBuffers.
    std::vector<float> scaleWS;     // float3 per splat: exp(log_scale)
    std::vector<float> rotationWS;  // float4 per splat: quaternion w,x,y,z (normalised)
    std::vector<float> opacityRaw;  // float  per splat: raw logit
    // SH coefficients: 16 float3 groups per splat, laid out as
    //   [sh0_r, sh0_g, sh0_b,  sh1_r, sh1_g, sh1_b, ...,  sh15_r, sh15_g, sh15_b]
    //   groups 0      = f_dc_0/1/2
    //   groups 1..15  = f_rest_0..44  (may be zeroed if PLY has no higher-order SH)
    std::vector<float> shCoeffs;    // float3 × 16 × N = 48 floats per splat

    size_t count() const { return splats.size(); }
    bool   empty() const { return splats.empty(); }

    // Rebuild all flattened arrays from splats (call after loading)
    void buildGPUArrays();
    void clear();
};
