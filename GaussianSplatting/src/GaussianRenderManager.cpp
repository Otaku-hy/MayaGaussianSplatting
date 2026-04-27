#define NOMINMAX
#include "GaussianRenderManager.h"
#include "GaussianNode.h"
#include "GaussianData.h"

#include <maya/MGlobal.h>

#include <d3d11.h>
#include <d3dcompiler.h>
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")

#include <algorithm>
#include <cstring>
#include <cmath>
#include <functional>

// ===========================================================================
// Merged Preprocessing compute shader HLSL
//   Difference from single-instance version:
//     - worldMat removed from CB
//     - Added gInstanceID (t5) and gWorldMats (t6) StructuredBuffers
//     - PreprocessKernel looks up world matrix per-splat via instanceID
// ===========================================================================
static const char* kMergedPreprocessCS = R"HLSL(
StructuredBuffer<float3> gPositionWS  : register(t0);
StructuredBuffer<float3> gScale       : register(t1);
StructuredBuffer<float4> gRotation    : register(t2);
StructuredBuffer<float>  gOpacity     : register(t3);
StructuredBuffer<float3> gSHsCoeff    : register(t4);
StructuredBuffer<uint>   gInstanceID  : register(t5);
// World matrices stored as 4 x float4 rows (row-major, matching Maya convention)
struct Float4x4 { float4 r0, r1, r2, r3; };
StructuredBuffer<Float4x4> gWorldMats : register(t6);
// Per-splat selection mask: bit 0 = selected, bit 1 = deleted
StructuredBuffer<uint>   gMask        : register(t7);

RWStructuredBuffer<float2> gPositionSS    : register(u0);
RWStructuredBuffer<float>  gDepth         : register(u1);
RWStructuredBuffer<float>  gRadius        : register(u2);
RWStructuredBuffer<float3> gColor         : register(u3);
RWStructuredBuffer<float4> gCov2D_opacity : register(u4);

cbuffer PreprocessParams : register(b0)
{
    row_major float4x4 viewMat;
    row_major float4x4 projMat;
    float3   cameraPos;
    float    padding0;
    float2   tanHalfFov;
    int      filmWidth;
    int      filmHeight;
    uint     gGaussCounts;
    uint     debugFixedRadius;
    float    pad2; float pad3;
};

float3x3 Get3DCovariance(float3 scale, float4 rotation)
{
    float r = rotation.x;
    float x = rotation.y;
    float y = rotation.z;
    float z = rotation.w;

    float3x3 R = float3x3(
        1 - 2*(y*y + z*z),  2*(x*y - r*z),      2*(x*z + r*y),
        2*(x*y + r*z),      1 - 2*(x*x + z*z),  2*(y*z - r*x),
        2*(x*z - r*y),      2*(y*z + r*x),      1 - 2*(x*x + y*y)
    );
    float3x3 S = float3x3(
        scale.x*scale.x, 0, 0,
        0, scale.y*scale.y, 0,
        0, 0, scale.z*scale.z
    );
    return mul(R, mul(S, transpose(R)));
}

float3 Get2DCovariance(float3x3 cov3D, float3 meanVS)
{
    float focalX = 0.5f * (float)filmWidth  / tanHalfFov.x;
    float focalY = 0.5f * (float)filmHeight / tanHalfFov.y;

    float limX = 1.3f * tanHalfFov.x;
    float limY = 1.3f * tanHalfFov.y;
    float3 mv  = meanVS;
    mv.x = clamp(mv.x / mv.z, -limX, limX) * mv.z;
    mv.y = clamp(mv.y / mv.z, -limY, limY) * mv.z;

    float3x3 J = float3x3(
        focalX / mv.z, 0.0f,          -focalX * mv.x / (mv.z * mv.z),
        0.0f,          focalY / mv.z, -focalY * mv.y / (mv.z * mv.z),
        0.0f,          0.0f,           0.0f
    );

    float3x3 W   = transpose((float3x3)viewMat);
    float3x3 T   = mul(J, W);
    float3x3 cov = mul(T, mul(cov3D, transpose(T)));

    return float3(cov[0][0] + 0.3f, cov[0][1], cov[1][1] + 0.3f);
}

float3 ComputeSphericalHarmonics(uint idx, float3 position, float3 camPos)
{
    uint shBaseIdx = idx * 16;
    float3 dir = normalize(position - camPos);

    float3 shColor = gSHsCoeff[shBaseIdx + 0] * 0.282095f;

    shColor += gSHsCoeff[shBaseIdx + 1] * -0.488603f * dir.y;
    shColor += gSHsCoeff[shBaseIdx + 2] *  0.488603f * dir.z;
    shColor += gSHsCoeff[shBaseIdx + 3] * -0.488603f * dir.x;

    shColor += gSHsCoeff[shBaseIdx + 4] *  1.092548f * dir.x * dir.y;
    shColor += gSHsCoeff[shBaseIdx + 5] * -1.092548f * dir.y * dir.z;
    shColor += gSHsCoeff[shBaseIdx + 6] *  0.315392f * (3.0f * dir.z * dir.z - 1.0f);
    shColor += gSHsCoeff[shBaseIdx + 7] * -1.092548f * dir.x * dir.z;
    shColor += gSHsCoeff[shBaseIdx + 8] *  0.546274f * (dir.x * dir.x - dir.y * dir.y);

    shColor += gSHsCoeff[shBaseIdx + 9]  * -0.590044f * dir.y * (3.0f * dir.x * dir.x - dir.y * dir.y);
    shColor += gSHsCoeff[shBaseIdx + 10] *  2.890611f * dir.x * dir.y * dir.z;
    shColor += gSHsCoeff[shBaseIdx + 11] * -0.457046f * dir.y * (5.0f * dir.z * dir.z - 1.0f);
    shColor += gSHsCoeff[shBaseIdx + 12] *  0.373176f * (5.0f * dir.z * dir.z * dir.z - 3.0f * dir.z);
    shColor += gSHsCoeff[shBaseIdx + 13] * -0.457046f * dir.x * (5.0f * dir.z * dir.z - 1.0f);
    shColor += gSHsCoeff[shBaseIdx + 14] *  1.445305f * dir.z * (dir.x * dir.x - dir.y * dir.y);
    shColor += gSHsCoeff[shBaseIdx + 15] * -0.590044f * dir.x * (dir.x * dir.x - 3.0f * dir.y * dir.y);

    shColor += 0.5f;
    return max(shColor, 0.0f);
}

[numthreads(256, 1, 1)]
void PreprocessKernel(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= gGaussCounts) return;

    // Deleted splats: emit zero-radius so they are skipped downstream.
    uint mask = gMask[id.x];
    if (mask & 2u) { gRadius[id.x] = 0.0f; return; }

    // Look up per-splat world matrix via instance ID
    uint inst = gInstanceID[id.x];
    Float4x4 wm = gWorldMats[inst];
    float4x4 worldMat = float4x4(wm.r0, wm.r1, wm.r2, wm.r3);

    float3 posOS = gPositionWS[id.x];
    float3 posWS = mul(float4(posOS, 1.0f), worldMat).xyz;

    float4 posVS = mul(float4(posWS, 1.0f), viewMat);

    if (posVS.z >= -0.2f) {
        gRadius[id.x] = 0.0f;
        return;
    }

    float4 posCS  = mul(posVS, projMat);
    float2 posNDC = posCS.xy / posCS.w;
    float2 posSS  = (posNDC * 0.5f + 0.5f) * float2((float)filmWidth, (float)filmHeight);

    float3x3 cov3D_obj = Get3DCovariance(gScale[id.x], gRotation[id.x]);
    float3x3 W3x3      = (float3x3)worldMat;
    float3x3 cov3D     = mul(transpose(W3x3), mul(cov3D_obj, W3x3));

    float3   cov2D = Get2DCovariance(cov3D, posVS.xyz);

    float det    = cov2D.x * cov2D.z - cov2D.y * cov2D.y;
    if (det <= 0.0f) { gRadius[id.x] = 0.0f; return; }

    float mid    = 0.5f * (cov2D.x + cov2D.z);
    float lambda = mid + sqrt(max(0.01f, mid*mid - det));
    float radius = ceil(3.0f * sqrt(lambda));

    if (radius > 1024.0f) { gRadius[id.x] = 0.0f; return; }

    if (posSS.x + radius < 0.0f || posSS.x - radius > (float)filmWidth ||
        posSS.y + radius < 0.0f || posSS.y - radius > (float)filmHeight)
    {
        gRadius[id.x] = 0.0f;
        return;
    }

    float3 invCov = float3(cov2D.z, -cov2D.y, cov2D.x) / det;

    float3 color = ComputeSphericalHarmonics(id.x, posWS, cameraPos);

    gPositionSS[id.x]    = posSS;
    gDepth[id.x]         = posCS.z / posCS.w;

    if (debugFixedRadius > 0) {
        float fr = (float)debugFixedRadius;
        gRadius[id.x]        = fr;
        gColor[id.x]         = color;
        float invR2 = 1.0f / (fr * fr * 0.1111f);
        gCov2D_opacity[id.x] = float4(invR2, 0.0f, invR2, gOpacity[id.x]);
    } else {
        gRadius[id.x]        = radius;
        gColor[id.x]         = color;
        gCov2D_opacity[id.x] = float4(invCov, gOpacity[id.x]);
    }
}
)HLSL";

// Production render shader (same as single-instance, reads from sorted indices)
static const char* kProdShaderSrc = R"HLSL(
StructuredBuffer<float2> gPositionSS    : register(t0);
StructuredBuffer<float>  gRadius        : register(t1);
StructuredBuffer<float3> gColor         : register(t2);
StructuredBuffer<float4> gCov2D_Opacity : register(t3);
StructuredBuffer<float>  gDepth         : register(t4);
StructuredBuffer<uint>   gSortedIndices : register(t5);
StructuredBuffer<uint>   gMask          : register(t6);

cbuffer CBRender : register(b0)
{
    float2 gViewportSize;
    float2 pad;
};

struct PS_IN {
    float4 clipPos  : SV_Position;
    float3 color    : COLOR0;
    float  opacity  : TEXCOORD3;
    float2 pixelOff : TEXCOORD0;
    float3 invCov2D : TEXCOORD1;
    float  selected : TEXCOORD2;   // 0 or 1
};

float sigmoid_approx(float x) { return 1.0f / (1.0f + exp(-x)); }

PS_IN VS(uint cornerID : SV_VertexID, uint iid : SV_InstanceID)
{
    PS_IN o = (PS_IN)0;

    uint idx = gSortedIndices[iid];
    float r  = gRadius[idx];

    if (r <= 0.0f) { o.clipPos = float4(0, 0, -2, 1); return o; }

    uint  m    = gMask[idx];
    if (m & 2u) { o.clipPos = float4(0, 0, -2, 1); return o; }  // deleted

    float2 spos  = gPositionSS[idx];
    float3 col   = gColor[idx];
    float4 cov4  = gCov2D_Opacity[idx];
    float  depth = gDepth[idx];

    float2 ndc = spos / gViewportSize * 2.0f - 1.0f;

    static const float2 corners[4] = {
        float2(-1.f,  1.f), float2( 1.f,  1.f),
        float2(-1.f, -1.f), float2( 1.f, -1.f),
    };
    float2 corner  = corners[cornerID];
    float2 halfNDC = float2(r / gViewportSize.x, r / gViewportSize.y) * 2.0f;

    o.clipPos   = float4(ndc + corner * halfNDC, depth, 1.0f);
    o.color     = col;
    o.opacity   = sigmoid_approx(cov4.w);
    o.pixelOff  = corner * r;
    o.invCov2D  = cov4.xyz;
    o.selected  = (m & 1u) ? 1.0f : 0.0f;
    return o;
}

float4 PS(PS_IN i) : SV_Target
{
    float x = i.pixelOff.x, y = i.pixelOff.y;
    float a = i.invCov2D.x, b = i.invCov2D.y, c = i.invCov2D.z;
    float power = -0.5f * (a*x*x + 2.0f*b*x*y + c*y*y);
    if (power > 0.0f) discard;
    float alpha = i.opacity * exp(power);
    if (alpha < 1.0f / 255.0f) discard;

    float3 col = i.color;
    if (i.selected > 0.5f) {
        // Tint selected splats toward bright yellow; boost alpha a bit
        col   = lerp(col, float3(1.0f, 0.85f, 0.15f), 0.75f);
        alpha = min(1.0f, alpha * 1.5f + 0.15f);
    }
    return float4(col, alpha);
}
)HLSL";

// Radix sort CS (identical to single-instance)
static const char* kRadixSortCS = R"HLSL(
#define SORT_GROUP_SIZE 256
#define ITEMS_PER_THREAD 16
#define TILE_SIZE (SORT_GROUP_SIZE * ITEMS_PER_THREAD)
#define RADIX_SIZE 256

cbuffer SortCB : register(b0) {
    uint gNumElements;
    uint gNumBlocks;
    uint gShift;
    uint gPadSort;
};

uint FloatToSortKey(float f) {
    uint bits = asuint(f);
    uint mask = (-(int)(bits >> 31)) | 0x80000000u;
    return bits ^ mask;
}

#ifdef KEYGEN_KERNEL
StructuredBuffer<float>    gDepthIn   : register(t0);
RWStructuredBuffer<uint>   gKeysOut   : register(u0);
RWStructuredBuffer<uint>   gValsOut   : register(u1);

[numthreads(SORT_GROUP_SIZE, 1, 1)]
void KeyGenKernel(uint3 id : SV_DispatchThreadID) {
    if (id.x >= gNumElements) return;
    gKeysOut[id.x] = ~FloatToSortKey(gDepthIn[id.x]);
    gValsOut[id.x] = id.x;
}
#endif

#ifdef COUNT_KERNEL
StructuredBuffer<uint>     gKeysIn    : register(t0);
RWStructuredBuffer<uint>   gBlockHist : register(u0);

groupshared uint sHist[RADIX_SIZE];

[numthreads(SORT_GROUP_SIZE, 1, 1)]
void CountKernel(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    sHist[tid.x] = 0;
    GroupMemoryBarrierWithGroupSync();

    uint base = gid.x * TILE_SIZE;
    for (uint i = 0; i < ITEMS_PER_THREAD; i++) {
        uint idx = base + tid.x + i * SORT_GROUP_SIZE;
        if (idx < gNumElements) {
            uint digit = (gKeysIn[idx] >> gShift) & 0xFFu;
            InterlockedAdd(sHist[digit], 1u);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    gBlockHist[gid.x * RADIX_SIZE + tid.x] = sHist[tid.x];
}
#endif

#ifdef SCAN_KERNEL
RWStructuredBuffer<uint> gBlockHist : register(u0);

groupshared uint sDigitTotal[RADIX_SIZE];

[numthreads(RADIX_SIZE, 1, 1)]
void ScanKernel(uint3 tid : SV_GroupThreadID) {
    uint digit = tid.x;

    uint total = 0;
    for (uint b = 0; b < gNumBlocks; b++) {
        uint count = gBlockHist[b * RADIX_SIZE + digit];
        gBlockHist[b * RADIX_SIZE + digit] = total;
        total += count;
    }

    sDigitTotal[digit] = total;
    GroupMemoryBarrierWithGroupSync();

    if (digit == 0) {
        uint sum = 0;
        for (uint d = 0; d < RADIX_SIZE; d++) {
            uint t = sDigitTotal[d];
            sDigitTotal[d] = sum;
            sum += t;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    uint digitOff = sDigitTotal[digit];
    for (uint b2 = 0; b2 < gNumBlocks; b2++) {
        gBlockHist[b2 * RADIX_SIZE + digit] += digitOff;
    }
}
#endif

#ifdef SCATTER_KERNEL
StructuredBuffer<uint>     gKeysIn     : register(t0);
StructuredBuffer<uint>     gValsIn     : register(t1);
StructuredBuffer<uint>     gBlockOff   : register(t2);
RWStructuredBuffer<uint>   gKeysOut    : register(u0);
RWStructuredBuffer<uint>   gValsOut    : register(u1);

groupshared uint sLocalRank[RADIX_SIZE];

[numthreads(SORT_GROUP_SIZE, 1, 1)]
void ScatterKernel(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    sLocalRank[tid.x] = 0;
    GroupMemoryBarrierWithGroupSync();

    uint base = gid.x * TILE_SIZE;
    for (uint i = 0; i < ITEMS_PER_THREAD; i++) {
        uint idx = base + tid.x + i * SORT_GROUP_SIZE;
        if (idx < gNumElements) {
            uint key   = gKeysIn[idx];
            uint val   = gValsIn[idx];
            uint digit = (key >> gShift) & 0xFFu;

            uint rank;
            InterlockedAdd(sLocalRank[digit], 1u, rank);

            uint pos = gBlockOff[gid.x * RADIX_SIZE + digit] + rank;
            gKeysOut[pos] = key;
            gValsOut[pos] = val;
        }
    }
}
#endif
)HLSL";

// Depth pass CS (identical to single-instance)
static const char* kDepthPassCS = R"HLSL(
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
)HLSL";

// Selection CS: project each splat through worldMat then viewProj, test if
// projected NDC is inside [rectMin, rectMax], update mask bit 0 per mode.
// Bit 1 (deleted) is preserved; deleted splats are never newly selected.
static const char* kSelectCS = R"HLSL(
cbuffer SelectCB : register(b0) {
    row_major float4x4 worldMat;
    row_major float4x4 viewProj;
    float2 rectMin;     // NDC space [-1,1]
    float2 rectMax;
    uint   splatCount;
    uint   mode;        // 0=replace, 1=add, 2=subtract, 3=toggle
    uint   pad0, pad1;
};
StructuredBuffer<float3>   gPositionOS : register(t0);
RWStructuredBuffer<uint>   gMask       : register(u0);

[numthreads(256, 1, 1)]
void SelectKernel(uint3 id : SV_DispatchThreadID) {
    if (id.x >= splatCount) return;

    uint cur = gMask[id.x];
    if (cur & 2u) return;  // deleted: never touch

    float4 p = mul(float4(gPositionOS[id.x], 1.0f), worldMat);
    p = mul(p, viewProj);
    bool behind = (p.w <= 0.0f);
    float3 ndc = p.xyz / max(abs(p.w), 1e-6f) * sign(p.w);
    bool inRect = !behind &&
                  ndc.x >= rectMin.x && ndc.x <= rectMax.x &&
                  ndc.y >= rectMin.y && ndc.y <= rectMax.y;

    uint oldSel = cur & 1u;
    uint newSel;
    if      (mode == 0u) newSel = inRect ? 1u : 0u;                          // replace
    else if (mode == 1u) newSel = (inRect || (oldSel != 0u)) ? 1u : 0u;      // add
    else if (mode == 2u) newSel = (inRect ? 0u : oldSel);                    // subtract
    else                 newSel = inRect ? (oldSel ^ 1u) : oldSel;           // toggle

    gMask[id.x] = (cur & ~1u) | newSel;
}
)HLSL";

// Depth copy shader (identical to single-instance)
static const char* kDepthCopyShader = R"HLSL(
Texture2D<uint> gDepthSrc : register(t0);

struct VS_OUT { float4 pos : SV_Position; };

VS_OUT CopyVS(uint vid : SV_VertexID)
{
    VS_OUT o;
    float2 uv = float2((vid << 1) & 2, vid & 2);
    o.pos = float4(uv * 2.0f - 1.0f, 0.0f, 1.0f);
    o.pos.y = -o.pos.y;
    return o;
}

void CopyPS(VS_OUT i, out float outDepth : SV_Depth)
{
    uint2 pix = uint2(i.pos.xy);
    uint  bits = gDepthSrc.Load(int3(pix, 0));
    outDepth = asfloat(bits);
}
)HLSL";

// ===========================================================================
// Constants
// ===========================================================================
static const uint32_t kSortGroupSize      = 256;
static const uint32_t kSortItemsPerThread = 16;
static const uint32_t kSortTileSize       = kSortGroupSize * kSortItemsPerThread;
static const uint32_t kRadixSize          = 256;

// ===========================================================================
// CB layouts (must match HLSL)
// ===========================================================================

// Merged preprocess CB: NO worldMat (uses per-splat lookup instead)
struct CBPreprocessMerged {
    float    viewMat[16];
    float    projMat[16];
    float    cameraPos[3];
    float    padding0;
    float    tanHalfFov[2];
    int      filmWidth;
    int      filmHeight;
    uint32_t gaussCount;
    uint32_t debugFixedRadius;
    float    pad2, pad3;
};
static_assert(sizeof(CBPreprocessMerged) % 16 == 0, "");

struct CBRender {
    float vpWidth, vpHeight;
    float pad[2];
};
static_assert(sizeof(CBRender) % 16 == 0, "");

struct CBSort {
    uint32_t numElements;
    uint32_t numBlocks;
    uint32_t shift;
    uint32_t pad;
};
static_assert(sizeof(CBSort) % 16 == 0, "");

struct CBDepth {
    uint32_t viewportW;
    uint32_t viewportH;
    uint32_t splatCount;
    uint32_t radiusCap;
    float    alphaThreshold;
    float    pad0, pad1, pad2;
};
static_assert(sizeof(CBDepth) % 16 == 0, "");

struct CBSelect {
    float    worldMat[16];
    float    viewProj[16];
    float    rectMin[2];
    float    rectMax[2];
    uint32_t splatCount;
    uint32_t mode;
    uint32_t pad0, pad1;
};
static_assert(sizeof(CBSelect) % 16 == 0, "");

// ===========================================================================
// Utility: compile a shader stage
// ===========================================================================
static bool CompileStage(const char* src, size_t srcLen,
                         const char* entry, const char* target,
                         ID3DBlob** outBlob,
                         const D3D_SHADER_MACRO* defines = nullptr)
{
    UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;
#ifdef _DEBUG
    flags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif
    ID3DBlob* errBlob = nullptr;
    HRESULT hr = D3DCompile(src, srcLen, nullptr, defines, nullptr,
                            entry, target, flags, 0, outBlob, &errBlob);
    if (FAILED(hr)) {
        if (errBlob) {
            MGlobal::displayError(
                MString("[GS-Manager] Shader compile error (") + entry + "): " +
                static_cast<const char*>(errBlob->GetBufferPointer()));
            errBlob->Release();
        }
        return false;
    }
    if (errBlob) errBlob->Release();
    return true;
}

#define SAFE_RELEASE(p) do { if (p) { (p)->Release(); (p) = nullptr; } } while(0)

// ===========================================================================
// Singleton
// ===========================================================================
GaussianRenderManager& GaussianRenderManager::instance() {
    static GaussianRenderManager s;
    return s;
}

GaussianRenderManager::~GaussianRenderManager() {
    releaseAll();
}

// ===========================================================================
// Per-frame API
// ===========================================================================
void GaussianRenderManager::beginFrame(uint64_t frameStamp) {
    if (m_frameStamp != frameStamp) {
        m_frameStamp    = frameStamp;
        m_frameRendered = false;
        m_instances.clear();
        m_totalSplats   = 0;
    }
}

void GaussianRenderManager::registerInstance(const RenderInstance& inst) {
    // Deduplicate by dataNode: Maya can call prepareForDraw multiple times per
    // logical frame (picking/shadow/transparency passes). Prevent instance count
    // from multiplying while still allowing re-registration after beginFrame clears.
    for (const auto& existing : m_instances)
        if (existing.node == inst.node) return;
    m_instances.push_back(inst);
    m_totalSplats += inst.splatCount;
}

void GaussianRenderManager::setFrameData(const float viewMat[16], const float projMat[16],
                                          const float cameraPos[3], const float tanHalfFov[2],
                                          float vpWidth, float vpHeight)
{
    std::memcpy(m_viewMat, viewMat, 64);
    std::memcpy(m_projMat, projMat, 64);
    std::memcpy(m_cameraPos, cameraPos, 12);
    m_tanHalfFov[0] = tanHalfFov[0];
    m_tanHalfFov[1] = tanHalfFov[1];
    m_vpWidth  = vpWidth;
    m_vpHeight = vpHeight;
}

bool GaussianRenderManager::canRender() const {
    return m_pipelineReady && m_sortReady && m_totalSplats > 0;
}

// ===========================================================================
// createUAVBuffer
// ===========================================================================
bool GaussianRenderManager::createUAVBuffer(ID3D11Device* device, const char* name,
                                             uint32_t numElements, uint32_t stride,
                                             ID3D11Buffer** outBuf,
                                             ID3D11UnorderedAccessView** outUAV,
                                             ID3D11ShaderResourceView** outSRV)
{
    D3D11_BUFFER_DESC bd = {};
    bd.ByteWidth           = numElements * stride;
    bd.Usage               = D3D11_USAGE_DEFAULT;
    bd.BindFlags           = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
    bd.MiscFlags           = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    bd.StructureByteStride = stride;

    HRESULT hr = device->CreateBuffer(&bd, nullptr, outBuf);
    if (FAILED(hr)) {
        MGlobal::displayError(MString("[GS-Manager] CreateBuffer(UAV) failed: ") + name);
        return false;
    }

    D3D11_UNORDERED_ACCESS_VIEW_DESC uavd = {};
    uavd.ViewDimension       = D3D11_UAV_DIMENSION_BUFFER;
    uavd.Buffer.NumElements  = numElements;
    hr = device->CreateUnorderedAccessView(*outBuf, &uavd, outUAV);
    if (FAILED(hr)) { SAFE_RELEASE(*outBuf); return false; }

    D3D11_SHADER_RESOURCE_VIEW_DESC srvd = {};
    srvd.ViewDimension         = D3D11_SRV_DIMENSION_BUFFEREX;
    srvd.BufferEx.NumElements  = numElements;
    hr = device->CreateShaderResourceView(*outBuf, &srvd, outSRV);
    if (FAILED(hr)) { SAFE_RELEASE(*outBuf); SAFE_RELEASE(*outUAV); return false; }

    return true;
}

// ===========================================================================
// createSRVBuffer  (immutable or default, with initial data upload)
// ===========================================================================
bool GaussianRenderManager::createSRVBuffer(ID3D11Device* device, const char* name,
                                             const void* initData, uint32_t numElements,
                                             uint32_t stride,
                                             ID3D11Buffer** outBuf,
                                             ID3D11ShaderResourceView** outSRV)
{
    D3D11_BUFFER_DESC bd = {};
    bd.ByteWidth           = numElements * stride;
    bd.Usage               = D3D11_USAGE_DEFAULT;
    bd.BindFlags           = D3D11_BIND_SHADER_RESOURCE;
    bd.MiscFlags           = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    bd.StructureByteStride = stride;

    D3D11_SUBRESOURCE_DATA srd = {};
    srd.pSysMem = initData;

    HRESULT hr = device->CreateBuffer(&bd, initData ? &srd : nullptr, outBuf);
    if (FAILED(hr)) {
        MGlobal::displayError(MString("[GS-Manager] CreateBuffer(SRV) failed: ") + name);
        return false;
    }

    D3D11_SHADER_RESOURCE_VIEW_DESC srvd = {};
    srvd.ViewDimension         = D3D11_SRV_DIMENSION_BUFFEREX;
    srvd.BufferEx.NumElements  = numElements;
    hr = device->CreateShaderResourceView(*outBuf, &srvd, outSRV);
    if (FAILED(hr)) { SAFE_RELEASE(*outBuf); return false; }

    return true;
}

// ===========================================================================
// Pipeline init
// ===========================================================================
bool GaussianRenderManager::initPipeline(ID3D11Device* device) {
    HRESULT hr;

    // Compile merged preprocess CS
    {
        ID3DBlob* blob = nullptr;
        if (!CompileStage(kMergedPreprocessCS, strlen(kMergedPreprocessCS),
                          "PreprocessKernel", "cs_5_0", &blob))
            return false;
        hr = device->CreateComputeShader(blob->GetBufferPointer(), blob->GetBufferSize(),
                                          nullptr, &m_preprocessCS);
        blob->Release();
        if (FAILED(hr)) return false;
    }

    // Preprocess CB
    {
        D3D11_BUFFER_DESC cbd = {};
        cbd.ByteWidth = sizeof(CBPreprocessMerged);
        cbd.Usage = D3D11_USAGE_DYNAMIC;
        cbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        cbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        if (FAILED(device->CreateBuffer(&cbd, nullptr, &m_preprocessCB))) return false;
    }

    // Production render shaders
    {
        size_t srcLen = strlen(kProdShaderSrc);
        ID3DBlob* vsBlob = nullptr, *psBlob = nullptr;
        if (!CompileStage(kProdShaderSrc, srcLen, "VS", "vs_5_0", &vsBlob)) return false;
        if (!CompileStage(kProdShaderSrc, srcLen, "PS", "ps_5_0", &psBlob)) { vsBlob->Release(); return false; }

        hr = device->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), nullptr, &m_prodVS);
        if (FAILED(hr)) { vsBlob->Release(); psBlob->Release(); return false; }
        hr = device->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(), nullptr, &m_prodPS);
        vsBlob->Release(); psBlob->Release();
        if (FAILED(hr)) return false;
    }

    // Render CB
    {
        D3D11_BUFFER_DESC cbd = {};
        cbd.ByteWidth = sizeof(CBRender);
        cbd.Usage = D3D11_USAGE_DYNAMIC;
        cbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        cbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        if (FAILED(device->CreateBuffer(&cbd, nullptr, &m_prodCB))) return false;
    }

    // Render states
    {
        D3D11_BLEND_DESC bd = {};
        bd.RenderTarget[0].BlendEnable           = TRUE;
        bd.RenderTarget[0].SrcBlend              = D3D11_BLEND_SRC_ALPHA;
        bd.RenderTarget[0].DestBlend             = D3D11_BLEND_INV_SRC_ALPHA;
        bd.RenderTarget[0].BlendOp               = D3D11_BLEND_OP_ADD;
        bd.RenderTarget[0].SrcBlendAlpha         = D3D11_BLEND_ONE;
        bd.RenderTarget[0].DestBlendAlpha        = D3D11_BLEND_ZERO;
        bd.RenderTarget[0].BlendOpAlpha          = D3D11_BLEND_OP_ADD;
        bd.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
        if (FAILED(device->CreateBlendState(&bd, &m_blendState))) return false;
    }
    {
        D3D11_RASTERIZER_DESC rd = {};
        rd.FillMode = D3D11_FILL_SOLID; rd.CullMode = D3D11_CULL_NONE; rd.DepthClipEnable = TRUE;
        if (FAILED(device->CreateRasterizerState(&rd, &m_rsState))) return false;
    }
    {
        D3D11_DEPTH_STENCIL_DESC dsd = {};
        dsd.DepthEnable    = TRUE;
        dsd.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
        dsd.DepthFunc      = D3D11_COMPARISON_LESS;
        dsd.StencilEnable  = FALSE;
        if (FAILED(device->CreateDepthStencilState(&dsd, &m_dsState))) return false;
    }

    m_pipelineReady = true;
    MGlobal::displayInfo("[GS-Manager] Merged pipeline: OK");
    return true;
}

bool GaussianRenderManager::initSortPipeline(ID3D11Device* device) {
    size_t srcLen = strlen(kRadixSortCS);

    struct KernelDef { const char* define; const char* entry; ID3D11ComputeShader** out; };
    KernelDef kernels[] = {
        { "KEYGEN_KERNEL",  "KeyGenKernel",  &m_sortCS_keygen  },
        { "COUNT_KERNEL",   "CountKernel",   &m_sortCS_count   },
        { "SCAN_KERNEL",    "ScanKernel",    &m_sortCS_scan    },
        { "SCATTER_KERNEL", "ScatterKernel", &m_sortCS_scatter },
    };

    for (auto& k : kernels) {
        D3D_SHADER_MACRO defines[] = { { k.define, "1" }, { nullptr, nullptr } };
        ID3DBlob* blob = nullptr;
        if (!CompileStage(kRadixSortCS, srcLen, k.entry, "cs_5_0", &blob, defines)) return false;
        HRESULT hr = device->CreateComputeShader(blob->GetBufferPointer(),
                                                  blob->GetBufferSize(), nullptr, k.out);
        blob->Release();
        if (FAILED(hr)) return false;
    }

    D3D11_BUFFER_DESC cbd = {};
    cbd.ByteWidth = sizeof(CBSort);
    cbd.Usage = D3D11_USAGE_DYNAMIC;
    cbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    cbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    if (FAILED(device->CreateBuffer(&cbd, nullptr, &m_sortCB))) return false;

    m_sortReady = true;
    MGlobal::displayInfo("[GS-Manager] Sort pipeline: OK");
    return true;
}

bool GaussianRenderManager::initSelectPipeline(ID3D11Device* device) {
    size_t srcLen = strlen(kSelectCS);
    ID3DBlob* blob = nullptr;
    if (!CompileStage(kSelectCS, srcLen, "SelectKernel", "cs_5_0", &blob)) return false;
    HRESULT hr = device->CreateComputeShader(blob->GetBufferPointer(),
                                              blob->GetBufferSize(),
                                              nullptr, &m_selectCS);
    blob->Release();
    if (FAILED(hr)) return false;

    D3D11_BUFFER_DESC cbd = {};
    cbd.ByteWidth      = sizeof(CBSelect);
    cbd.Usage          = D3D11_USAGE_DYNAMIC;
    cbd.BindFlags      = D3D11_BIND_CONSTANT_BUFFER;
    cbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    if (FAILED(device->CreateBuffer(&cbd, nullptr, &m_selectCB))) return false;

    m_selectReady = true;
    MGlobal::displayInfo("[GS-Manager] Select pipeline: OK");
    return true;
}

// ---------------------------------------------------------------------------
// runSelection  --  dispatch selection CS on one data node's mask buffer.
// Called by gsMarqueeSelect command; rectMin/rectMax are in NDC space.
// ---------------------------------------------------------------------------
bool GaussianRenderManager::runSelection(ID3D11Device* /*device*/, ID3D11DeviceContext* ctx,
                                         GaussianNode* node,
                                         const float worldMat[16],
                                         const float viewProj[16],
                                         float rectMinX, float rectMinY,
                                         float rectMaxX, float rectMaxY,
                                         int mode)
{
    // CPU-only rect selection: no GPU dispatch, no readback stall.
    // We own m_maskShadow (always in sync) and upload via UpdateSubresource.
    if (!node || !node->areInputsReady() || !node->bufSelectionMask())
        return false;

    uint32_t N = node->splatCount();
    if (N == 0) return false;

    // wvp = worldMat * viewProj (row-major: pos_clip = pos_os * wvp)
    float wvp[16];
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) {
            float s = 0.f;
            for (int k = 0; k < 4; k++) s += worldMat[r*4+k] * viewProj[k*4+c];
            wvp[r*4+c] = s;
        }

    const auto& splats = node->gaussianData().splats;
    auto&       mask   = node->maskShadowMutable();

    for (uint32_t i = 0; i < N; i++) {
        uint32_t cur = mask[i];
        if (cur & 2u) continue;  // deleted: never touch

        const float* p = splats[i].position;
        // row-vector * wvp
        float cx = p[0]*wvp[0] + p[1]*wvp[4] + p[2]*wvp[8]  + wvp[12];
        float cy = p[0]*wvp[1] + p[1]*wvp[5] + p[2]*wvp[9]  + wvp[13];
        float cw = p[0]*wvp[3] + p[1]*wvp[7] + p[2]*wvp[11] + wvp[15];

        bool inRect = false;
        if (cw > 0.f) {
            float nx = cx / cw, ny = cy / cw;
            inRect = (nx >= rectMinX && nx <= rectMaxX &&
                      ny >= rectMinY && ny <= rectMaxY);
        }

        uint32_t oldSel = cur & 1u;
        uint32_t newSel;
        if      (mode == 0) newSel = inRect ? 1u : 0u;
        else if (mode == 1) newSel = (inRect || oldSel) ? 1u : 0u;
        else if (mode == 2) newSel = inRect ? 0u : oldSel;
        else                newSel = inRect ? (oldSel ^ 1u) : oldSel;

        mask[i] = (cur & ~1u) | newSel;
    }

    uint32_t selectedCount = 0;
    for (const auto& v : mask) if (v & 1u) selectedCount++;
    MGlobal::displayInfo(MString("[GS Select] ") + selectedCount + "/" + N +
                         " splats selected (mode=" + mode + ")");

    ctx->UpdateSubresource(node->bufSelectionMask(), 0, nullptr, mask.data(), 0, 0);
    node->markMaskChanged();
    m_selectionDirty = true;
    return true;
}

bool GaussianRenderManager::initDepthPassPipeline(ID3D11Device* device) {
    size_t csLen = strlen(kDepthPassCS);

    {
        D3D_SHADER_MACRO defs[] = { { "CLEAR_DEPTH_KERNEL", "1" }, { nullptr, nullptr } };
        ID3DBlob* blob = nullptr;
        if (!CompileStage(kDepthPassCS, csLen, "ClearDepthKernel", "cs_5_0", &blob, defs)) return false;
        HRESULT hr = device->CreateComputeShader(blob->GetBufferPointer(), blob->GetBufferSize(),
                                                  nullptr, &m_depthClearCS);
        blob->Release();
        if (FAILED(hr)) return false;
    }
    {
        D3D_SHADER_MACRO defs[] = { { "DEPTH_PASS_KERNEL", "1" }, { nullptr, nullptr } };
        ID3DBlob* blob = nullptr;
        if (!CompileStage(kDepthPassCS, csLen, "DepthPassKernel", "cs_5_0", &blob, defs)) return false;
        HRESULT hr = device->CreateComputeShader(blob->GetBufferPointer(), blob->GetBufferSize(),
                                                  nullptr, &m_depthPassCS);
        blob->Release();
        if (FAILED(hr)) return false;
    }
    {
        D3D11_BUFFER_DESC cbd = {};
        cbd.ByteWidth = sizeof(CBDepth);
        cbd.Usage = D3D11_USAGE_DYNAMIC;
        cbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        cbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        if (FAILED(device->CreateBuffer(&cbd, nullptr, &m_depthCB))) return false;
    }
    {
        size_t vsLen = strlen(kDepthCopyShader);
        ID3DBlob* vsBlob = nullptr, *psBlob = nullptr;
        if (!CompileStage(kDepthCopyShader, vsLen, "CopyVS", "vs_5_0", &vsBlob)) return false;
        if (!CompileStage(kDepthCopyShader, vsLen, "CopyPS", "ps_5_0", &psBlob)) { vsBlob->Release(); return false; }
        HRESULT hr = device->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), nullptr, &m_depthCopyVS);
        if (FAILED(hr)) { vsBlob->Release(); psBlob->Release(); return false; }
        hr = device->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(), nullptr, &m_depthCopyPS);
        vsBlob->Release(); psBlob->Release();
        if (FAILED(hr)) return false;
    }
    {
        D3D11_DEPTH_STENCIL_DESC dsd = {};
        dsd.DepthEnable = TRUE;
        dsd.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
        dsd.DepthFunc = D3D11_COMPARISON_LESS;
        dsd.StencilEnable = FALSE;
        if (FAILED(device->CreateDepthStencilState(&dsd, &m_depthWriteDS))) return false;
    }
    {
        D3D11_BLEND_DESC bd = {};
        bd.RenderTarget[0].BlendEnable = FALSE;
        bd.RenderTarget[0].RenderTargetWriteMask = 0;
        if (FAILED(device->CreateBlendState(&bd, &m_depthCopyBlend))) return false;
    }

    m_depthPassReady = true;
    MGlobal::displayInfo("[GS-Manager] Depth pass pipeline: OK");
    return true;
}

// ===========================================================================
// Buffer management
// ===========================================================================
bool GaussianRenderManager::createComputeOutputs(ID3D11Device* device, uint32_t N) {
    releaseComputeOutputs();

    if (!createUAVBuffer(device, "m_positionSS", N, sizeof(float)*2, &m_ubPositionSS, &m_uavPositionSS, &m_srvPositionSS)) return false;
    if (!createUAVBuffer(device, "m_depth",      N, sizeof(float),   &m_ubDepth,      &m_uavDepth,      &m_srvDepth))      return false;
    if (!createUAVBuffer(device, "m_radius",     N, sizeof(float),   &m_ubRadius,     &m_uavRadius,     &m_srvRadius))     return false;
    if (!createUAVBuffer(device, "m_color",      N, sizeof(float)*3, &m_ubColor,      &m_uavColor,      &m_srvColor))      return false;
    if (!createUAVBuffer(device, "m_cov2D",      N, sizeof(float)*4, &m_ubCov2D,      &m_uavCov2D,      &m_srvCov2D))      return false;

    if (m_sortReady) {
        if (!createSortBuffers(device, N))
            MGlobal::displayError("[GS-Manager] Sort buffer creation failed.");
    }

    m_mergedAllocN = N;
    MGlobal::displayInfo(MString("[GS-Manager] Compute outputs created: N=") + N);
    return true;
}

bool GaussianRenderManager::createSortBuffers(ID3D11Device* device, uint32_t N) {
    releaseSortBuffers();

    uint32_t numBlocks = (N + kSortTileSize - 1) / kSortTileSize;

    if (!createUAVBuffer(device, "sortKeysA", N, sizeof(uint32_t), &m_sortKeysA, &m_sortKeysA_UAV, &m_sortKeysA_SRV)) return false;
    if (!createUAVBuffer(device, "sortKeysB", N, sizeof(uint32_t), &m_sortKeysB, &m_sortKeysB_UAV, &m_sortKeysB_SRV)) return false;
    if (!createUAVBuffer(device, "sortValsA", N, sizeof(uint32_t), &m_sortValsA, &m_sortValsA_UAV, &m_sortValsA_SRV)) return false;
    if (!createUAVBuffer(device, "sortValsB", N, sizeof(uint32_t), &m_sortValsB, &m_sortValsB_UAV, &m_sortValsB_SRV)) return false;
    if (!createUAVBuffer(device, "sortBlockHist", numBlocks * kRadixSize, sizeof(uint32_t),
                         &m_sortBlockHist, &m_sortBlockHist_UAV, &m_sortBlockHist_SRV)) return false;

    return true;
}

bool GaussianRenderManager::createDepthTexture(ID3D11Device* device, uint32_t w, uint32_t h) {
    SAFE_RELEASE(m_depthTex);
    SAFE_RELEASE(m_depthTex_UAV);
    SAFE_RELEASE(m_depthTex_SRV);
    m_depthTexW = m_depthTexH = 0;

    D3D11_TEXTURE2D_DESC td = {};
    td.Width = w; td.Height = h; td.MipLevels = 1; td.ArraySize = 1;
    td.Format = DXGI_FORMAT_R32_UINT;
    td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;

    if (FAILED(device->CreateTexture2D(&td, nullptr, &m_depthTex))) return false;

    D3D11_UNORDERED_ACCESS_VIEW_DESC uavd = {};
    uavd.Format = DXGI_FORMAT_R32_UINT;
    uavd.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
    if (FAILED(device->CreateUnorderedAccessView(m_depthTex, &uavd, &m_depthTex_UAV))) {
        SAFE_RELEASE(m_depthTex); return false;
    }

    D3D11_SHADER_RESOURCE_VIEW_DESC srvd = {};
    srvd.Format = DXGI_FORMAT_R32_UINT;
    srvd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srvd.Texture2D.MipLevels = 1;
    if (FAILED(device->CreateShaderResourceView(m_depthTex, &srvd, &m_depthTex_SRV))) {
        SAFE_RELEASE(m_depthTex); SAFE_RELEASE(m_depthTex_UAV); return false;
    }

    m_depthTexW = w; m_depthTexH = h;
    return true;
}

// ===========================================================================
// buildMergedInputs  --  CPU concatenate all instance data, upload to GPU
//
// The large splat-data buffers (pos/scale/rotation/opacity/SH/instanceID)
// are only rebuilt when the instance set actually changes (different data
// nodes or different splat counts). World matrices are tiny and updated
// every frame since transforms can change.
// ===========================================================================
bool GaussianRenderManager::buildMergedInputs(ID3D11Device* device, ID3D11DeviceContext* ctx) {
    uint32_t N = m_totalSplats;
    uint32_t numInstances = (uint32_t)m_instances.size();

    if (N == 0 || numInstances == 0) return false;

    // Compute a signature of the current instance set to detect changes.
    // Hash = XOR-combine of (dataNode pointer, splatCount) per instance.
    size_t sig = 0;
    for (uint32_t i = 0; i < numInstances; i++) {
        auto ptr = reinterpret_cast<uintptr_t>(m_instances[i].node);
        sig ^= std::hash<uintptr_t>()(ptr) + 0x9e3779b9 + (sig << 6) + (sig >> 2);
        sig ^= std::hash<uint32_t>()(m_instances[i].splatCount) + 0x9e3779b9 + (sig << 6) + (sig >> 2);
    }

    bool needRebuild = (sig != m_cachedSignature) || !m_inputsUploaded;

    // --- Rebuild large buffers only when instance set changes ---
    if (needRebuild) {
        std::vector<float>    mergedPos;       mergedPos.reserve((size_t)N * 3);
        std::vector<float>    mergedScale;     mergedScale.reserve((size_t)N * 3);
        std::vector<float>    mergedRotation;  mergedRotation.reserve((size_t)N * 4);
        std::vector<float>    mergedOpacity;   mergedOpacity.reserve(N);
        std::vector<float>    mergedSH;        mergedSH.reserve((size_t)N * 48);
        std::vector<uint32_t> instanceIDs;     instanceIDs.reserve(N);

        for (uint32_t i = 0; i < numInstances; i++) {
            const RenderInstance& inst = m_instances[i];
            const GaussianData& gd = inst.node->gaussianData();
            uint32_t cnt = inst.splatCount;

            mergedPos.insert(mergedPos.end(),
                             gd.positions.begin(), gd.positions.begin() + (size_t)cnt * 3);
            mergedScale.insert(mergedScale.end(),
                               gd.scaleWS.begin(), gd.scaleWS.begin() + (size_t)cnt * 3);
            mergedRotation.insert(mergedRotation.end(),
                                  gd.rotationWS.begin(), gd.rotationWS.begin() + (size_t)cnt * 4);
            mergedOpacity.insert(mergedOpacity.end(),
                                 gd.opacityRaw.begin(), gd.opacityRaw.begin() + cnt);
            mergedSH.insert(mergedSH.end(),
                            gd.shCoeffs.begin(), gd.shCoeffs.begin() + (size_t)cnt * 48);
            instanceIDs.insert(instanceIDs.end(), cnt, i);
        }

        bool needRealloc = (m_mergedAllocN != N) || (m_mergedAllocInstances != numInstances);

        if (needRealloc) {
            releaseMergedInputs();

            if (!createSRVBuffer(device, "mergedPosWS", mergedPos.data(), N, sizeof(float)*3,
                                 &m_mergedPositionWS, &m_mergedSrvPosWS)) return false;
            if (!createSRVBuffer(device, "mergedScale", mergedScale.data(), N, sizeof(float)*3,
                                 &m_mergedScale, &m_mergedSrvScale)) return false;
            if (!createSRVBuffer(device, "mergedRotation", mergedRotation.data(), N, sizeof(float)*4,
                                 &m_mergedRotation, &m_mergedSrvRotation)) return false;
            if (!createSRVBuffer(device, "mergedOpacity", mergedOpacity.data(), N, sizeof(float),
                                 &m_mergedOpacity, &m_mergedSrvOpacity)) return false;
            if (!createSRVBuffer(device, "mergedSH", mergedSH.data(), N * kSHCoeffsPerSplat, sizeof(float)*3,
                                 &m_mergedSHCoeffs, &m_mergedSrvSH)) return false;
            if (!createSRVBuffer(device, "instanceID", instanceIDs.data(), N, sizeof(uint32_t),
                                 &m_instanceIDBuf, &m_instanceIDSrv)) return false;

            m_mergedAllocInstances = numInstances;

            // Also reallocate compute outputs and sort buffers
            if (!createComputeOutputs(device, N)) return false;
        } else {
            ctx->UpdateSubresource(m_mergedPositionWS, 0, nullptr, mergedPos.data(), 0, 0);
            ctx->UpdateSubresource(m_mergedScale, 0, nullptr, mergedScale.data(), 0, 0);
            ctx->UpdateSubresource(m_mergedRotation, 0, nullptr, mergedRotation.data(), 0, 0);
            ctx->UpdateSubresource(m_mergedOpacity, 0, nullptr, mergedOpacity.data(), 0, 0);
            ctx->UpdateSubresource(m_mergedSHCoeffs, 0, nullptr, mergedSH.data(), 0, 0);
            ctx->UpdateSubresource(m_instanceIDBuf, 0, nullptr, instanceIDs.data(), 0, 0);
        }

        m_cachedSignature = sig;
        m_inputsUploaded  = true;

        MGlobal::displayInfo(MString("[GS-Manager] Merged inputs rebuilt: ") +
                             N + " splats, " + numInstances + " instances");
    }

    // --- Always update world matrices (tiny: numInstances * 64 bytes) ---
    {
        std::vector<float> worldMats;
        worldMats.reserve((size_t)numInstances * 16);
        for (uint32_t i = 0; i < numInstances; i++) {
            for (int j = 0; j < 16; j++)
                worldMats.push_back(m_instances[i].worldMat[j]);
        }

        bool needReallocMats = (m_mergedAllocInstances != numInstances) || !m_worldMatsBuf;
        if (needReallocMats) {
            SAFE_RELEASE(m_worldMatsBuf);
            SAFE_RELEASE(m_worldMatsSrv);
            if (!createSRVBuffer(device, "worldMats", worldMats.data(), numInstances,
                                 sizeof(float)*16, &m_worldMatsBuf, &m_worldMatsSrv))
                return false;
        } else {
            ctx->UpdateSubresource(m_worldMatsBuf, 0, nullptr, worldMats.data(), 0, 0);
        }
    }

    return true;
}

// ===========================================================================
// updateMergedSelection  --  concat per-instance selection masks into the
// merged buffer using GPU-side CopySubresourceRegion. Skips work if no
// instance's mask has changed since last call and the instance set matches.
// ===========================================================================
bool GaussianRenderManager::updateMergedSelection(ID3D11Device* device,
                                                   ID3D11DeviceContext* ctx) {
    uint32_t N = m_totalSplats;
    uint32_t numInstances = (uint32_t)m_instances.size();
    if (N == 0 || numInstances == 0) return false;

    // Allocate / reallocate if size changed
    bool needRealloc = (!m_mergedSelection) ||
                        (m_instanceMaskVersions.size() != numInstances);
    if (needRealloc) {
        SAFE_RELEASE(m_mergedSelection);
        SAFE_RELEASE(m_srvMergedSelection);
        std::vector<uint32_t> zeroes(N, 0u);
        if (!createSRVBuffer(device, "mergedSelection", zeroes.data(),
                             N, sizeof(uint32_t),
                             &m_mergedSelection, &m_srvMergedSelection))
            return false;
        m_instanceMaskVersions.assign(numInstances, (uint64_t)-1);
        m_selectionDirty = true;
    }

    // Check if any per-instance mask version drifted
    bool anyChanged = m_selectionDirty;
    if (!anyChanged) {
        for (uint32_t i = 0; i < numInstances; i++) {
            if (m_instanceMaskVersions[i] != m_instances[i].node->maskVersion()) {
                anyChanged = true;
                break;
            }
        }
    }
    if (!anyChanged) return true;

    // Concatenate via CopySubresourceRegion (GPU-side byte copy per instance).
    uint32_t byteOffset = 0;
    for (uint32_t i = 0; i < numInstances; i++) {
        const RenderInstance& inst = m_instances[i];
        GaussianNode* dn = inst.node;
        uint32_t cnt = inst.splatCount;
        if (!dn->bufSelectionMask() || cnt == 0) {
            byteOffset += cnt * sizeof(uint32_t);
            continue;
        }
        D3D11_BOX box = {};
        box.left   = 0;
        box.right  = cnt * sizeof(uint32_t);
        box.top    = 0;
        box.bottom = 1;
        box.front  = 0;
        box.back   = 1;
        ctx->CopySubresourceRegion(m_mergedSelection, 0,
                                    byteOffset, 0, 0,
                                    dn->bufSelectionMask(), 0, &box);
        byteOffset += cnt * sizeof(uint32_t);
        m_instanceMaskVersions[i] = dn->maskVersion();
    }

    m_selectionDirty = false;
    return true;
}

// ===========================================================================
// render  --  main merged pipeline
// ===========================================================================
bool GaussianRenderManager::render(ID3D11Device* device, ID3D11DeviceContext* ctx, int renderMode) {
    if (m_frameRendered) return false;  // already done this frame
    m_frameRendered = true;

    if (m_instances.empty() || m_totalSplats == 0) return false;

    // Lazy pipeline init
    if (!m_pipelineReady) {
        if (!initPipeline(device)) return false;
    }
    if (!m_sortReady) {
        if (!initSortPipeline(device)) return false;
    }
    if (!m_depthPassReady) {
        initDepthPassPipeline(device);  // non-fatal
    }

    // Build/update merged input buffers
    if (!buildMergedInputs(device, ctx)) return false;

    // Refresh merged selection mask (concat per-instance masks)
    updateMergedSelection(device, ctx);

    uint32_t N = m_totalSplats;

    // Ensure depth texture matches viewport
    uint32_t vpW = (uint32_t)m_vpWidth;
    uint32_t vpH = (uint32_t)m_vpHeight;
    if (m_depthPassReady && (m_depthTexW != vpW || m_depthTexH != vpH) && vpW > 0 && vpH > 0) {
        createDepthTexture(device, vpW, vpH);
    }

    // -- 1. Update preprocess CB --
    {
        D3D11_MAPPED_SUBRESOURCE mapped;
        if (SUCCEEDED(ctx->Map(m_preprocessCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
            CBPreprocessMerged* cb = static_cast<CBPreprocessMerged*>(mapped.pData);
            std::memcpy(cb->viewMat,   m_viewMat,   64);
            std::memcpy(cb->projMat,   m_projMat,   64);
            std::memcpy(cb->cameraPos, m_cameraPos,  12);
            cb->padding0       = 0.f;
            cb->tanHalfFov[0]  = m_tanHalfFov[0];
            cb->tanHalfFov[1]  = m_tanHalfFov[1];
            cb->filmWidth      = (int)m_vpWidth;
            cb->filmHeight     = (int)m_vpHeight;
            cb->gaussCount     = N;
            cb->debugFixedRadius = (renderMode == 3) ? 5 : 0;
            cb->pad2 = cb->pad3 = 0.f;
            ctx->Unmap(m_preprocessCB, 0);
        }
    }

    // -- 2. Dispatch preprocess --
    {
        ID3D11ShaderResourceView* srvs[] = {
            m_mergedSrvPosWS, m_mergedSrvScale, m_mergedSrvRotation,
            m_mergedSrvOpacity, m_mergedSrvSH, m_instanceIDSrv, m_worldMatsSrv,
            m_srvMergedSelection
        };
        ID3D11UnorderedAccessView* uavs[] = {
            m_uavPositionSS, m_uavDepth, m_uavRadius, m_uavColor, m_uavCov2D
        };
        ctx->CSSetShader(m_preprocessCS, nullptr, 0);
        ctx->CSSetConstantBuffers(0, 1, &m_preprocessCB);
        ctx->CSSetShaderResources(0, 8, srvs);
        ctx->CSSetUnorderedAccessViews(0, 5, uavs, nullptr);

        ctx->Dispatch((N + 255) / 256, 1, 1);

        ID3D11UnorderedAccessView* nullUAVs[5] = {};
        ctx->CSSetUnorderedAccessViews(0, 5, nullUAVs, nullptr);
        ID3D11ShaderResourceView* nullSRVs[8] = {};
        ctx->CSSetShaderResources(0, 8, nullSRVs);
    }

    // -- 3. GPU Radix Sort --
    {
        uint32_t numBlocks = (N + kSortTileSize - 1) / kSortTileSize;

        // 3a. KeyGen
        {
            CBSort scb = { N, numBlocks, 0, 0 };
            D3D11_MAPPED_SUBRESOURCE mapped;
            if (SUCCEEDED(ctx->Map(m_sortCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
                std::memcpy(mapped.pData, &scb, sizeof(scb));
                ctx->Unmap(m_sortCB, 0);
            }
            ctx->CSSetShader(m_sortCS_keygen, nullptr, 0);
            ctx->CSSetConstantBuffers(0, 1, &m_sortCB);
            ID3D11ShaderResourceView* kgSRV[] = { m_srvDepth };
            ctx->CSSetShaderResources(0, 1, kgSRV);
            ID3D11UnorderedAccessView* kgUAV[] = { m_sortKeysA_UAV, m_sortValsA_UAV };
            ctx->CSSetUnorderedAccessViews(0, 2, kgUAV, nullptr);
            ctx->Dispatch((N + kSortGroupSize - 1) / kSortGroupSize, 1, 1);
            ID3D11ShaderResourceView*  n1[1] = {};
            ID3D11UnorderedAccessView* n2[2] = {};
            ctx->CSSetShaderResources(0, 1, n1);
            ctx->CSSetUnorderedAccessViews(0, 2, n2, nullptr);
        }

        // 3b. Four radix passes
        for (uint32_t pass = 0; pass < 4; pass++) {
            bool even = (pass % 2 == 0);
            auto keysInSRV  = even ? m_sortKeysA_SRV  : m_sortKeysB_SRV;
            auto keysOutUAV = even ? m_sortKeysB_UAV  : m_sortKeysA_UAV;
            auto valsInSRV  = even ? m_sortValsA_SRV  : m_sortValsB_SRV;
            auto valsOutUAV = even ? m_sortValsB_UAV  : m_sortValsA_UAV;

            {
                CBSort scb = { N, numBlocks, pass * 8, 0 };
                D3D11_MAPPED_SUBRESOURCE mapped;
                if (SUCCEEDED(ctx->Map(m_sortCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
                    std::memcpy(mapped.pData, &scb, sizeof(scb));
                    ctx->Unmap(m_sortCB, 0);
                }
            }

            // Count
            {
                ctx->CSSetShader(m_sortCS_count, nullptr, 0);
                ctx->CSSetConstantBuffers(0, 1, &m_sortCB);
                ctx->CSSetShaderResources(0, 1, &keysInSRV);
                ctx->CSSetUnorderedAccessViews(0, 1, &m_sortBlockHist_UAV, nullptr);
                ctx->Dispatch(numBlocks, 1, 1);
                ID3D11ShaderResourceView*  n1[1] = {};
                ID3D11UnorderedAccessView* n1u[1] = {};
                ctx->CSSetShaderResources(0, 1, n1);
                ctx->CSSetUnorderedAccessViews(0, 1, n1u, nullptr);
            }

            // Scan
            {
                ctx->CSSetShader(m_sortCS_scan, nullptr, 0);
                ctx->CSSetConstantBuffers(0, 1, &m_sortCB);
                ctx->CSSetUnorderedAccessViews(0, 1, &m_sortBlockHist_UAV, nullptr);
                ctx->Dispatch(1, 1, 1);
                ID3D11UnorderedAccessView* n1u[1] = {};
                ctx->CSSetUnorderedAccessViews(0, 1, n1u, nullptr);
            }

            // Scatter
            {
                ctx->CSSetShader(m_sortCS_scatter, nullptr, 0);
                ctx->CSSetConstantBuffers(0, 1, &m_sortCB);
                ID3D11ShaderResourceView* scSRV[] = { keysInSRV, valsInSRV, m_sortBlockHist_SRV };
                ctx->CSSetShaderResources(0, 3, scSRV);
                ID3D11UnorderedAccessView* scUAV[] = { keysOutUAV, valsOutUAV };
                ctx->CSSetUnorderedAccessViews(0, 2, scUAV, nullptr);
                ctx->Dispatch(numBlocks, 1, 1);
                ID3D11ShaderResourceView*  n3[3] = {};
                ID3D11UnorderedAccessView* n2[2] = {};
                ctx->CSSetShaderResources(0, 3, n3);
                ctx->CSSetUnorderedAccessViews(0, 2, n2, nullptr);
            }
        }
        ctx->CSSetShader(nullptr, nullptr, 0);
    }

    // -- 4. Update render CB --
    {
        D3D11_MAPPED_SUBRESOURCE mapped;
        if (SUCCEEDED(ctx->Map(m_prodCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
            CBRender* cb = static_cast<CBRender*>(mapped.pData);
            cb->vpWidth  = m_vpWidth;
            cb->vpHeight = m_vpHeight;
            cb->pad[0]   = cb->pad[1] = 0.f;
            ctx->Unmap(m_prodCB, 0);
        }
    }

    // -- 5. Render (sorted instanced draw) --
    {
        float blendFactor[] = { 1.f, 1.f, 1.f, 1.f };
        ctx->OMSetBlendState(m_blendState, blendFactor, 0xFFFFFFFF);
        ctx->RSSetState(m_rsState);
        ctx->OMSetDepthStencilState(m_dsState, 0);

        ID3D11ShaderResourceView* vsSRVs[] = {
            m_srvPositionSS, m_srvRadius, m_srvColor,
            m_srvCov2D, m_srvDepth, m_sortValsA_SRV, m_srvMergedSelection
        };
        ctx->IASetInputLayout(nullptr);
        ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
        ctx->VSSetShader(m_prodVS, nullptr, 0);
        ctx->VSSetConstantBuffers(0, 1, &m_prodCB);
        ctx->VSSetShaderResources(0, 7, vsSRVs);
        ctx->GSSetShader(nullptr, nullptr, 0);
        ctx->PSSetShader(m_prodPS, nullptr, 0);
        ctx->DrawInstanced(4, N, 0, 0);

        ID3D11ShaderResourceView* nullSRVs7[7] = {};
        ctx->VSSetShaderResources(0, 7, nullSRVs7);
    }

    // -- 6. Depth pass --
    if (m_depthPassReady && m_depthTex_UAV && m_depthTex_SRV &&
        m_depthTexW > 0 && m_depthTexH > 0)
    {
        uint32_t W = m_depthTexW;
        uint32_t H = m_depthTexH;

        // 6a. Update depth CB
        {
            D3D11_MAPPED_SUBRESOURCE mapped;
            if (SUCCEEDED(ctx->Map(m_depthCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
                CBDepth* cb = static_cast<CBDepth*>(mapped.pData);
                cb->viewportW      = W;
                cb->viewportH      = H;
                cb->splatCount     = N;
                cb->radiusCap      = 16;
                cb->alphaThreshold = 0.5f;
                cb->pad0 = cb->pad1 = cb->pad2 = 0.f;
                ctx->Unmap(m_depthCB, 0);
            }
        }

        // 6b. Clear depth UAV
        {
            ctx->CSSetShader(m_depthClearCS, nullptr, 0);
            ctx->CSSetConstantBuffers(0, 1, &m_depthCB);
            ctx->CSSetUnorderedAccessViews(0, 1, &m_depthTex_UAV, nullptr);
            ctx->Dispatch((W + 15) / 16, (H + 15) / 16, 1);
        }

        // 6c. Depth pass kernel
        {
            ID3D11ShaderResourceView* srvs[] = {
                m_srvPositionSS, m_srvRadius, m_srvDepth, m_srvCov2D
            };
            ctx->CSSetShader(m_depthPassCS, nullptr, 0);
            ctx->CSSetConstantBuffers(0, 1, &m_depthCB);
            ctx->CSSetShaderResources(0, 4, srvs);
            ctx->Dispatch((N + 255) / 256, 1, 1);

            ID3D11ShaderResourceView*  nullSRV4[4] = {};
            ID3D11UnorderedAccessView* nullUAV1[1] = {};
            ctx->CSSetShaderResources(0, 4, nullSRV4);
            ctx->CSSetUnorderedAccessViews(0, 1, nullUAV1, nullptr);
            ctx->CSSetShader(nullptr, nullptr, 0);
        }

        // 6d. Copy pass
        {
            float copyBF[] = { 1.f, 1.f, 1.f, 1.f };
            ctx->OMSetBlendState(m_depthCopyBlend, copyBF, 0xFFFFFFFF);
            ctx->OMSetDepthStencilState(m_depthWriteDS, 0);

            ctx->IASetInputLayout(nullptr);
            ID3D11Buffer* nullVB[1] = {};
            UINT nullStride[1] = { 0 };
            UINT nullOffset[1] = { 0 };
            ctx->IASetVertexBuffers(0, 1, nullVB, nullStride, nullOffset);
            ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

            ctx->VSSetShader(m_depthCopyVS, nullptr, 0);
            ctx->GSSetShader(nullptr, nullptr, 0);
            ctx->PSSetShader(m_depthCopyPS, nullptr, 0);
            ctx->PSSetShaderResources(0, 1, &m_depthTex_SRV);
            ctx->Draw(3, 0);

            ID3D11ShaderResourceView* nullPSSRV[1] = {};
            ctx->PSSetShaderResources(0, 1, nullPSSRV);
        }
    }

    return true;
}

// ===========================================================================
// Release helpers
// ===========================================================================
void GaussianRenderManager::releaseMergedInputs() {
    SAFE_RELEASE(m_mergedPositionWS); SAFE_RELEASE(m_mergedSrvPosWS);
    SAFE_RELEASE(m_mergedScale);      SAFE_RELEASE(m_mergedSrvScale);
    SAFE_RELEASE(m_mergedRotation);   SAFE_RELEASE(m_mergedSrvRotation);
    SAFE_RELEASE(m_mergedOpacity);    SAFE_RELEASE(m_mergedSrvOpacity);
    SAFE_RELEASE(m_mergedSHCoeffs);   SAFE_RELEASE(m_mergedSrvSH);
    SAFE_RELEASE(m_instanceIDBuf);    SAFE_RELEASE(m_instanceIDSrv);
    SAFE_RELEASE(m_worldMatsBuf);     SAFE_RELEASE(m_worldMatsSrv);
    SAFE_RELEASE(m_mergedSelection);  SAFE_RELEASE(m_srvMergedSelection);
    m_mergedAllocN = 0;
    m_mergedAllocInstances = 0;
    m_cachedSignature = 0;
    m_inputsUploaded  = false;
    m_instanceMaskVersions.clear();
    m_selectionDirty = true;
}

void GaussianRenderManager::releaseComputeOutputs() {
    SAFE_RELEASE(m_ubPositionSS); SAFE_RELEASE(m_uavPositionSS); SAFE_RELEASE(m_srvPositionSS);
    SAFE_RELEASE(m_ubDepth);      SAFE_RELEASE(m_uavDepth);      SAFE_RELEASE(m_srvDepth);
    SAFE_RELEASE(m_ubRadius);     SAFE_RELEASE(m_uavRadius);     SAFE_RELEASE(m_srvRadius);
    SAFE_RELEASE(m_ubColor);      SAFE_RELEASE(m_uavColor);      SAFE_RELEASE(m_srvColor);
    SAFE_RELEASE(m_ubCov2D);      SAFE_RELEASE(m_uavCov2D);      SAFE_RELEASE(m_srvCov2D);
    releaseSortBuffers();
    m_mergedAllocN = 0;
}

void GaussianRenderManager::releaseSortBuffers() {
    SAFE_RELEASE(m_sortKeysA); SAFE_RELEASE(m_sortKeysA_UAV); SAFE_RELEASE(m_sortKeysA_SRV);
    SAFE_RELEASE(m_sortKeysB); SAFE_RELEASE(m_sortKeysB_UAV); SAFE_RELEASE(m_sortKeysB_SRV);
    SAFE_RELEASE(m_sortValsA); SAFE_RELEASE(m_sortValsA_UAV); SAFE_RELEASE(m_sortValsA_SRV);
    SAFE_RELEASE(m_sortValsB); SAFE_RELEASE(m_sortValsB_UAV); SAFE_RELEASE(m_sortValsB_SRV);
    SAFE_RELEASE(m_sortBlockHist); SAFE_RELEASE(m_sortBlockHist_UAV); SAFE_RELEASE(m_sortBlockHist_SRV);
}

void GaussianRenderManager::releaseDepthPassResources() {
    SAFE_RELEASE(m_depthClearCS);
    SAFE_RELEASE(m_depthPassCS);
    SAFE_RELEASE(m_depthCB);
    SAFE_RELEASE(m_depthTex);
    SAFE_RELEASE(m_depthTex_UAV);
    SAFE_RELEASE(m_depthTex_SRV);
    SAFE_RELEASE(m_depthCopyVS);
    SAFE_RELEASE(m_depthCopyPS);
    SAFE_RELEASE(m_depthWriteDS);
    SAFE_RELEASE(m_depthCopyBlend);
    m_depthTexW = m_depthTexH = 0;
    m_depthPassReady = false;
}

void GaussianRenderManager::releasePipeline() {
    SAFE_RELEASE(m_preprocessCS);
    SAFE_RELEASE(m_preprocessCB);
    SAFE_RELEASE(m_prodVS);
    SAFE_RELEASE(m_prodPS);
    SAFE_RELEASE(m_prodCB);
    SAFE_RELEASE(m_blendState);
    SAFE_RELEASE(m_rsState);
    SAFE_RELEASE(m_dsState);
    SAFE_RELEASE(m_sortCS_keygen);
    SAFE_RELEASE(m_sortCS_count);
    SAFE_RELEASE(m_sortCS_scan);
    SAFE_RELEASE(m_sortCS_scatter);
    SAFE_RELEASE(m_sortCB);
    SAFE_RELEASE(m_selectCS);
    SAFE_RELEASE(m_selectCB);
    m_pipelineReady = false;
    m_sortReady     = false;
    m_selectReady   = false;
}

void GaussianRenderManager::releaseAll() {
    releaseMergedInputs();
    releaseComputeOutputs();
    releaseDepthPassResources();
    releasePipeline();
    m_instances.clear();
    m_totalSplats = 0;
    m_frameRendered = false;
}

#undef SAFE_RELEASE
