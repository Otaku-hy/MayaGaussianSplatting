// Depth pass for GS -> Maya occlusion. Two kernels (compiled separately):
//   CLEAR_DEPTH_KERNEL -> ClearDepthKernel
//   DEPTH_PASS_KERNEL  -> DepthPassKernel  (atomic-min depth write per splat footprint)

cbuffer DepthCB : register(b0)
{
    uint  gViewportW;
    uint  gViewportH;
    uint  gSplatCount;
    uint  gRadiusCap;
    float gAlphaThreshold;
    float gDpadA, gDpadB, gDpadC;
};

RWTexture2D<uint> gDepthUAV : register(u0);

#ifdef CLEAR_DEPTH_KERNEL
[numthreads(16, 16, 1)]
void ClearDepthKernel(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= gViewportW || id.y >= gViewportH) return;
    gDepthUAV[id.xy] = asuint(1.0f);
}
#endif

#ifdef DEPTH_PASS_KERNEL
StructuredBuffer<float2> gPositionSS   : register(t0);
StructuredBuffer<float>  gRadius       : register(t1);
StructuredBuffer<float>  gDepth        : register(t2);
StructuredBuffer<float4> gCov2DOpacity : register(t3);

float sigmoid_approx(float x) { return 1.0f / (1.0f + exp(-x)); }

[numthreads(256, 1, 1)]
void DepthPassKernel(uint3 id : SV_DispatchThreadID)
{
    uint sidx = id.x;
    if (sidx >= gSplatCount) return;

    float r = gRadius[sidx];
    if (r <= 0.0f) return;

    float  depth  = gDepth[sidx];
    if (depth <= 0.0f || depth >= 1.0f) return;

    float4 cov4   = gCov2DOpacity[sidx];
    float  opacity = sigmoid_approx(cov4.w);
    if (opacity < gAlphaThreshold) return;

    float2 center = gPositionSS[sidx];

    float rCapped = min(r, (float)gRadiusCap);

    int minX = max(0, (int)floor(center.x - rCapped));
    int maxX = min((int)gViewportW - 1, (int)ceil (center.x + rCapped));
    int minY = max(0, (int)floor(center.y - rCapped));
    int maxY = min((int)gViewportH - 1, (int)ceil (center.y + rCapped));
    if (minX > maxX || minY > maxY) return;

    uint  depthBits = asuint(depth);
    float a = cov4.x, b = cov4.y, c = cov4.z;

    for (int y = minY; y <= maxY; y++) {
        for (int x = minX; x <= maxX; x++) {
            float dx = (float)x + 0.5f - center.x;
            float dy = (float)y + 0.5f - center.y;
            float power = -0.5f * (a*dx*dx + 2.0f*b*dx*dy + c*dy*dy);
            if (power > 0.0f) continue;
            float alpha = opacity * exp(power);
            if (alpha < gAlphaThreshold) continue;
            InterlockedMin(gDepthUAV[uint2(x, y)], depthBits);
        }
    }
}
#endif
