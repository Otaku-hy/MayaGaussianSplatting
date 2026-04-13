#include "GaussianDrawOverride.h"
#include "GaussianNode.h"
#include "GaussianDataNode.h"

#include <maya/MDagPath.h>
#include <maya/MFnDependencyNode.h>
#include <maya/MPlug.h>
#include <maya/MGlobal.h>
#include <maya/MDrawRegistry.h>
#include <maya/MFrameContext.h>
#include <maya/MDrawContext.h>
#include <maya/MViewport2Renderer.h>
#include <maya/MMatrix.h>

#include <d3d11.h>
#include <d3dcompiler.h>
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")

#include <algorithm>
#include <cstring>
#include <cmath>

// ===========================================================================
// Debug pipeline HLSL  (reads from shared StructuredBuffers, no sort)
// ===========================================================================
static const char* kDbgShaderSrc = R"HLSL(
StructuredBuffer<float3> gPositionWS : register(t0);
StructuredBuffer<float>  gOpacity    : register(t1);
StructuredBuffer<float3> gSHCoeffs   : register(t2);

cbuffer CBDebug : register(b0)
{
    row_major float4x4 gWVP;
    float  gPointSize;
    float  gVPWidth;
    float  gVPHeight;
    uint   gSHStride;    // = 16 (groups per splat), for indexing into gSHCoeffs
};

struct VS_OUT { float4 clip : SV_Position; float4 col : COLOR; };
struct GS_OUT { float4 clip : SV_Position; float4 col : COLOR; float2 uv : TEXCOORD0; };

VS_OUT VS(uint vid : SV_VertexID)
{
    float3 pos = gPositionWS[vid];
    // degree-0 SH color
    float3 col = gSHCoeffs[vid * gSHStride] * 0.282095f + 0.5f;
    col = max(col, 0.0f);
    // sigmoid opacity
    float alpha = 1.0f / (1.0f + exp(-gOpacity[vid]));

    VS_OUT o;
    o.clip = mul(float4(pos, 1.0f), gWVP);
    o.col  = float4(col, alpha);
    return o;
}

[maxvertexcount(4)]
void GS(point VS_OUT input[1], inout TriangleStream<GS_OUT> stream)
{
    float4 c = input[0].clip;
    float2 h = float2(gPointSize / gVPWidth, gPointSize / gVPHeight) * c.w;
    static const float2 corners[4] = {
        float2(-1.f,  1.f), float2( 1.f,  1.f),
        float2(-1.f, -1.f), float2( 1.f, -1.f),
    };
    GS_OUT o;
    o.col = input[0].col;
    [unroll]
    for (int k = 0; k < 4; k++) {
        o.clip = c + float4(corners[k] * h, 0.f, 0.f);
        o.uv   = corners[k];
        stream.Append(o);
    }
    stream.RestartStrip();
}

float4 PS(GS_OUT i) : SV_Target
{
    float r2 = dot(i.uv, i.uv);
    clip(1.0f - r2);
    float a = i.col.a * (1.0f - r2 * 0.4f);
    return float4(i.col.rgb, a);
}
)HLSL";

// ===========================================================================
// Production pipeline HLSL  (ellipse Gaussian splat, depth-sorted)
// ===========================================================================
static const char* kProdShaderSrc = R"HLSL(
StructuredBuffer<float2> gPositionSS    : register(t0);
StructuredBuffer<float>  gRadius        : register(t1);
StructuredBuffer<float3> gColor         : register(t2);
StructuredBuffer<float4> gCov2D_Opacity : register(t3);
StructuredBuffer<float>  gDepth         : register(t4);
StructuredBuffer<uint>   gSortedIndices : register(t5);

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
};

float sigmoid_approx(float x) { return 1.0f / (1.0f + exp(-x)); }

// Instanced rendering: 4 vertices per splat (triangle strip quad), no GS.
// SV_VertexID  = corner index (0..3)
// SV_InstanceID = splat draw order (0..N-1, maps to sorted index)
PS_IN VS(uint cornerID : SV_VertexID, uint iid : SV_InstanceID)
{
    PS_IN o = (PS_IN)0;

    uint idx = gSortedIndices[iid];
    float r  = gRadius[idx];

    if (r <= 0.0f) { o.clipPos = float4(0, 0, -2, 1); return o; }

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
    return float4(i.color, alpha);
}
)HLSL";

// ===========================================================================
// Preprocessing compute shader HLSL
// ===========================================================================
static const char* kPreprocessCS = R"HLSL(
StructuredBuffer<float3> gPositionWS  : register(t0);
StructuredBuffer<float3> gScale       : register(t1);
StructuredBuffer<float4> gRotation    : register(t2);
StructuredBuffer<float>  gOpacity     : register(t3);
StructuredBuffer<float3> gSHsCoeff    : register(t4);

RWStructuredBuffer<float2> gPositionSS    : register(u0);
RWStructuredBuffer<float>  gDepth         : register(u1);
RWStructuredBuffer<float>  gRadius        : register(u2);
RWStructuredBuffer<float3> gColor         : register(u3);
RWStructuredBuffer<float4> gCov2D_opacity : register(u4);

cbuffer PreprocessParams : register(b0)
{
    row_major float4x4 worldMat;
    row_major float4x4 viewMat;
    row_major float4x4 projMat;
    float3   cameraPos;
    float    padding0;
    float2   tanHalfFov;
    int      filmWidth;
    int      filmHeight;
    uint     gGaussCounts;
    uint     debugFixedRadius;   // non-zero = override radius with this value
    float    pad2; float pad3;
};

float3x3 Get3DCovariance(float3 scale, float4 rotation)
{
    // rotation stored as float4(w, x, y, z) from PLY (rot_0..rot_3)
    // HLSL float4: .x=w(scalar), .y=x, .z=y, .w=z
    float r = rotation.x;   // w (scalar part)
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
    // Pixel-space focal lengths (matching reference 3DGS implementation)
    float focalX = 0.5f * (float)filmWidth  / tanHalfFov.x;
    float focalY = 0.5f * (float)filmHeight / tanHalfFov.y;

    // Clamp x/z and y/z ratios to avoid instability at screen edges
    float limX = 1.3f * tanHalfFov.x;
    float limY = 1.3f * tanHalfFov.y;
    float3 mv  = meanVS;
    mv.x = clamp(mv.x / mv.z, -limX, limX) * mv.z;
    mv.y = clamp(mv.y / mv.z, -limY, limY) * mv.z;

    // Jacobian of pinhole projection (camera-space -> pixel-space)
    float3x3 J = float3x3(
        focalX / mv.z, 0.0f,          -focalX * mv.x / (mv.z * mv.z),
        0.0f,          focalY / mv.z, -focalY * mv.y / (mv.z * mv.z),
        0.0f,          0.0f,           0.0f
    );

    // Maya row-major convention: v' = v*M, so view rotation in col-vec form is M^T
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

    // Transform from object space to world space via the node's transform
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

    // Object-space 3D covariance, then transform to world space.
    // Maya row-major: posWS = posOS * worldMat, so column-vec rotation = worldMat^T.
    // cov_WS = worldMat^T * cov_obj * worldMat
    float3x3 cov3D_obj = Get3DCovariance(gScale[id.x], gRotation[id.x]);
    float3x3 W3x3      = (float3x3)worldMat;
    float3x3 cov3D     = mul(transpose(W3x3), mul(cov3D_obj, W3x3));

    float3   cov2D = Get2DCovariance(cov3D, posVS.xyz);

    float det    = cov2D.x * cov2D.z - cov2D.y * cov2D.y;
    if (det <= 0.0f) { gRadius[id.x] = 0.0f; return; }

    float mid    = 0.5f * (cov2D.x + cov2D.z);
    float lambda = mid + sqrt(max(0.01f, mid*mid - det));
    float radius = ceil(3.0f * sqrt(lambda));

    // Clamp radius to avoid GPU-killing full-screen quads
    if (radius > 1024.0f) { gRadius[id.x] = 0.0f; return; }

    // Screen-space frustum cull: discard if the splat's bounding quad
    // (center ± radius) lies entirely outside the viewport rectangle.
    // Saves sort/draw/VS work for off-screen splats.
    if (posSS.x + radius < 0.0f || posSS.x - radius > (float)filmWidth ||
        posSS.y + radius < 0.0f || posSS.y - radius > (float)filmHeight)
    {
        gRadius[id.x] = 0.0f;
        return;
    }

    float3 invCov = float3(cov2D.z, -cov2D.y, cov2D.x) / det;

    // SH evaluated in world space
    float3 color = ComputeSphericalHarmonics(id.x, posWS, cameraPos);

    gPositionSS[id.x]    = posSS;
    gDepth[id.x]         = posCS.z / posCS.w;

    if (debugFixedRadius > 0) {
        // Diagnostic: fixed small radius, ignore covariance
        float fr = (float)debugFixedRadius;
        gRadius[id.x]        = fr;
        gColor[id.x]         = color;
        // Use identity-like inv covariance for circular splat: a=1/r², b=0, c=1/r²
        float invR2 = 1.0f / (fr * fr * 0.1111f);  // 1/(r/3)^2 so Gaussian decays to ~0 at edge
        gCov2D_opacity[id.x] = float4(invR2, 0.0f, invR2, gOpacity[id.x]);
    } else {
        gRadius[id.x]        = radius;
        gColor[id.x]         = color;
        gCov2D_opacity[id.x] = float4(invCov, gOpacity[id.x]);
    }
}
)HLSL";

// ===========================================================================
// GPU Radix Sort compute shader  (CS 5.0, no wave intrinsics)
//
// 4 kernels, each compiled with a different preprocessor define.
// 8-bit radix, 4 passes, ping-pong key/value buffers.
// ===========================================================================
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
// -- Key generation: float depth -> sortable uint, init indices --
StructuredBuffer<float>    gDepthIn   : register(t0);
RWStructuredBuffer<uint>   gKeysOut   : register(u0);
RWStructuredBuffer<uint>   gValsOut   : register(u1);

[numthreads(SORT_GROUP_SIZE, 1, 1)]
void KeyGenKernel(uint3 id : SV_DispatchThreadID) {
    if (id.x >= gNumElements) return;
    gKeysOut[id.x] = ~FloatToSortKey(gDepthIn[id.x]);   // descending = back-to-front
    gValsOut[id.x] = id.x;
}
#endif

#ifdef COUNT_KERNEL
// -- Histogram: count 256 radix digits per block --
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
// -- Prefix sum: scan block histograms per digit + compute digit offsets --
RWStructuredBuffer<uint> gBlockHist : register(u0);

groupshared uint sDigitTotal[RADIX_SIZE];

[numthreads(RADIX_SIZE, 1, 1)]
void ScanKernel(uint3 tid : SV_GroupThreadID) {
    uint digit = tid.x;

    // Exclusive prefix sum across blocks for this digit
    uint total = 0;
    for (uint b = 0; b < gNumBlocks; b++) {
        uint count = gBlockHist[b * RADIX_SIZE + digit];
        gBlockHist[b * RADIX_SIZE + digit] = total;
        total += count;
    }

    sDigitTotal[digit] = total;
    GroupMemoryBarrierWithGroupSync();

    // Exclusive prefix sum across all 256 digits (thread 0, sequential)
    if (digit == 0) {
        uint sum = 0;
        for (uint d = 0; d < RADIX_SIZE; d++) {
            uint t = sDigitTotal[d];
            sDigitTotal[d] = sum;
            sum += t;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // Add digit offset to every block prefix sum
    uint digitOff = sDigitTotal[digit];
    for (uint b2 = 0; b2 < gNumBlocks; b2++) {
        gBlockHist[b2 * RADIX_SIZE + digit] += digitOff;
    }
}
#endif

#ifdef SCATTER_KERNEL
// -- Scatter: redistribute elements by digit to output --
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

// ===========================================================================
// Depth Pass compute shader  (method B: per-pixel representative depth)
//
// Two kernels sharing the same UAV:
//   ClearDepthKernel  — clears UAV texture to 1.0f (far)
//   DepthPassKernel   — one thread per splat; iterates the splat's screen
//                       AABB and atomic-mins its NDC depth into every pixel
//                       where the Gaussian alpha exceeds the threshold.
//
// Result is an R32_UINT texture where each pixel stores asuint(nearestDepth).
// The copy pass then writes those values into Maya's real depth buffer.
// ===========================================================================
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
    if (r <= 0.0f) return;   // already culled (near plane / off-screen / degenerate)

    float  depth  = gDepth[sidx];
    if (depth <= 0.0f || depth >= 1.0f) return;  // behind camera or at far

    float4 cov4   = gCov2DOpacity[sidx];
    float  opacity = sigmoid_approx(cov4.w);
    if (opacity < gAlphaThreshold) return;       // skip barely-visible splats

    float2 center = gPositionSS[sidx];

    // Cap iteration area to bound per-thread work. Very large splats only
    // contribute depth inside a capped region around their center.
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

            // asuint preserves ordering for positive floats; NDC depth in [0,1]
            // so InterlockedMin gives us the nearest contributing splat.
            InterlockedMin(gDepthUAV[uint2(x, y)], depthBits);
        }
    }
}
#endif
)HLSL";

// ===========================================================================
// Depth Copy shader  (full-screen triangle, writes SV_Depth from UAV texture)
// ===========================================================================
static const char* kDepthCopyShader = R"HLSL(
Texture2D<uint> gDepthSrc : register(t0);

struct VS_OUT { float4 pos : SV_Position; };

VS_OUT CopyVS(uint vid : SV_VertexID)
{
    // Full-screen triangle (covers NDC [-1,1]^2 with one triangle)
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
// Constant buffer layouts (must match HLSL)
// ===========================================================================
static const uint32_t kSortGroupSize     = 256;
static const uint32_t kSortItemsPerThread = 16;
static const uint32_t kSortTileSize      = kSortGroupSize * kSortItemsPerThread;  // 4096
static const uint32_t kRadixSize         = 256;

struct CBDebug {
    float wvp[16];
    float pointSize, vpWidth, vpHeight;
    uint32_t shStride;
};
static_assert(sizeof(CBDebug) % 16 == 0, "");

struct CBPreprocess {
    float    worldMat[16];
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
static_assert(sizeof(CBPreprocess) % 16 == 0, "");

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
                MString("[GaussianSplat] Shader compile error (") + entry + "): " +
                static_cast<const char*>(errBlob->GetBufferPointer()));
            errBlob->Release();
        }
        return false;
    }
    if (errBlob) errBlob->Release();
    return true;
}

// ===========================================================================
// GaussianDrawData
// ===========================================================================
GaussianDrawData::GaussianDrawData() : MUserData() {}
GaussianDrawData::~GaussianDrawData() { releaseAll(); }

#define SAFE_RELEASE(p) do { if (p) { (p)->Release(); (p) = nullptr; } } while(0)

void GaussianDrawData::releaseAll()
{
    releaseDebugResources();
    releaseProductionResources();
    releaseDepthPassResources();
    // shared SRV pointers are non-owning; just null them
    sharedSrvPositionWS = sharedSrvScale = sharedSrvRotation = nullptr;
    sharedSrvOpacity = sharedSrvSHCoeffs = nullptr;
    inputsReady = false;
}

void GaussianDrawData::releaseDebugResources()
{
    SAFE_RELEASE(dbgVS);
    SAFE_RELEASE(dbgGS);
    SAFE_RELEASE(dbgPS);
    SAFE_RELEASE(dbgCB);
    SAFE_RELEASE(blendState);
    SAFE_RELEASE(rsState);
    SAFE_RELEASE(dsState);
    dbgReady = false;
}

void GaussianDrawData::releaseProductionResources()
{
    SAFE_RELEASE(computeShader);
    SAFE_RELEASE(computeCB);
    SAFE_RELEASE(ubPositionSS);  SAFE_RELEASE(uavPositionSS); SAFE_RELEASE(srvPositionSS);
    SAFE_RELEASE(ubDepth);       SAFE_RELEASE(uavDepth);      SAFE_RELEASE(srvDepth);
    SAFE_RELEASE(ubRadius);      SAFE_RELEASE(uavRadius);     SAFE_RELEASE(srvRadius);
    SAFE_RELEASE(ubColor);       SAFE_RELEASE(uavColor);      SAFE_RELEASE(srvColor);
    SAFE_RELEASE(ubCov2D);       SAFE_RELEASE(uavCov2D);      SAFE_RELEASE(srvCov2D);
    SAFE_RELEASE(prodVS);
    SAFE_RELEASE(prodPS);
    SAFE_RELEASE(prodCB);
    releaseSortResources();
    prodReady  = false;
    allocatedN = 0;
}

void GaussianDrawData::releaseSortResources()
{
    SAFE_RELEASE(sortCS_keygen);  SAFE_RELEASE(sortCS_count);
    SAFE_RELEASE(sortCS_scan);    SAFE_RELEASE(sortCS_scatter);
    SAFE_RELEASE(sortCB);
    SAFE_RELEASE(sortKeysA); SAFE_RELEASE(sortKeysA_UAV); SAFE_RELEASE(sortKeysA_SRV);
    SAFE_RELEASE(sortKeysB); SAFE_RELEASE(sortKeysB_UAV); SAFE_RELEASE(sortKeysB_SRV);
    SAFE_RELEASE(sortValsA); SAFE_RELEASE(sortValsA_UAV); SAFE_RELEASE(sortValsA_SRV);
    SAFE_RELEASE(sortValsB); SAFE_RELEASE(sortValsB_UAV); SAFE_RELEASE(sortValsB_SRV);
    SAFE_RELEASE(sortBlockHist); SAFE_RELEASE(sortBlockHist_UAV); SAFE_RELEASE(sortBlockHist_SRV);
    sortReady = false;
}

void GaussianDrawData::releaseDepthPassResources()
{
    SAFE_RELEASE(depthClearCS);
    SAFE_RELEASE(depthPassCS);
    SAFE_RELEASE(depthCB);
    SAFE_RELEASE(depthTex);
    SAFE_RELEASE(depthTex_UAV);
    SAFE_RELEASE(depthTex_SRV);
    SAFE_RELEASE(depthCopyVS);
    SAFE_RELEASE(depthCopyPS);
    SAFE_RELEASE(depthWriteDS);
    SAFE_RELEASE(depthCopyBlend);
    depthTexW = depthTexH = 0;
    depthPassReady = false;
}

// ---------------------------------------------------------------------------
// createUAVBuffer  (with error reporting)
// ---------------------------------------------------------------------------
bool GaussianDrawData::createUAVBuffer(ID3D11Device* device,
                                        const char*   name,
                                        uint32_t      numElements,
                                        uint32_t      stride,
                                        ID3D11Buffer**              outBuf,
                                        ID3D11UnorderedAccessView** outUAV,
                                        ID3D11ShaderResourceView**  outSRV)
{
    uint32_t totalBytes = numElements * stride;

    D3D11_BUFFER_DESC bd = {};
    bd.ByteWidth           = totalBytes;
    bd.Usage               = D3D11_USAGE_DEFAULT;
    bd.BindFlags           = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
    bd.MiscFlags           = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    bd.StructureByteStride = stride;

    HRESULT hr = device->CreateBuffer(&bd, nullptr, outBuf);
    if (FAILED(hr)) {
        MGlobal::displayError(
            MString("[GaussianSplat] CreateBuffer(UAV) failed for '") + name +
            "' (" + (unsigned int)(totalBytes / 1024 / 1024) + " MB).");
        return false;
    }

    D3D11_UNORDERED_ACCESS_VIEW_DESC uavd = {};
    uavd.ViewDimension       = D3D11_UAV_DIMENSION_BUFFER;
    uavd.Buffer.FirstElement = 0;
    uavd.Buffer.NumElements  = numElements;
    hr = device->CreateUnorderedAccessView(*outBuf, &uavd, outUAV);
    if (FAILED(hr)) {
        MGlobal::displayError(
            MString("[GaussianSplat] CreateUAV failed for '") + name + "'.");
        SAFE_RELEASE(*outBuf);
        return false;
    }

    D3D11_SHADER_RESOURCE_VIEW_DESC srvd = {};
    srvd.ViewDimension         = D3D11_SRV_DIMENSION_BUFFEREX;
    srvd.BufferEx.FirstElement = 0;
    srvd.BufferEx.NumElements  = numElements;
    hr = device->CreateShaderResourceView(*outBuf, &srvd, outSRV);
    if (FAILED(hr)) {
        MGlobal::displayError(
            MString("[GaussianSplat] CreateSRV(UAV) failed for '") + name + "'.");
        SAFE_RELEASE(*outBuf);
        SAFE_RELEASE(*outUAV);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// initDebugPipeline
// ---------------------------------------------------------------------------
bool GaussianDrawData::initDebugPipeline(ID3D11Device* device)
{
    HRESULT hr;
    size_t srcLen = strlen(kDbgShaderSrc);

    ID3DBlob* vsBlob = nullptr, *gsBlob = nullptr, *psBlob = nullptr;
    if (!CompileStage(kDbgShaderSrc, srcLen, "VS", "vs_5_0", &vsBlob)) return false;
    if (!CompileStage(kDbgShaderSrc, srcLen, "GS", "gs_5_0", &gsBlob)) { vsBlob->Release(); return false; }
    if (!CompileStage(kDbgShaderSrc, srcLen, "PS", "ps_5_0", &psBlob)) { vsBlob->Release(); gsBlob->Release(); return false; }

    hr = device->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), nullptr, &dbgVS);
    if (FAILED(hr)) goto dbg_cleanup;
    hr = device->CreateGeometryShader(gsBlob->GetBufferPointer(), gsBlob->GetBufferSize(), nullptr, &dbgGS);
    if (FAILED(hr)) goto dbg_cleanup;
    hr = device->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(), nullptr, &dbgPS);
    if (FAILED(hr)) goto dbg_cleanup;

    // No InputLayout needed: VS uses SV_VertexID, no vertex buffers
    {
        D3D11_BUFFER_DESC cbd = {};
        cbd.ByteWidth = sizeof(CBDebug); cbd.Usage = D3D11_USAGE_DYNAMIC;
        cbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER; cbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        hr = device->CreateBuffer(&cbd, nullptr, &dbgCB);
        if (FAILED(hr)) goto dbg_cleanup;
    }
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
        hr = device->CreateBlendState(&bd, &blendState);
        if (FAILED(hr)) goto dbg_cleanup;
    }
    {
        D3D11_RASTERIZER_DESC rd = {};
        rd.FillMode = D3D11_FILL_SOLID; rd.CullMode = D3D11_CULL_NONE; rd.DepthClipEnable = TRUE;
        hr = device->CreateRasterizerState(&rd, &rsState);
        if (FAILED(hr)) goto dbg_cleanup;
    }
    {
        D3D11_DEPTH_STENCIL_DESC dsd = {};
        dsd.DepthEnable    = TRUE;
        dsd.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
        dsd.DepthFunc      = D3D11_COMPARISON_LESS;
        dsd.StencilEnable  = FALSE;
        hr = device->CreateDepthStencilState(&dsd, &dsState);
        if (FAILED(hr)) goto dbg_cleanup;
    }

    dbgReady = true;
dbg_cleanup:
    vsBlob->Release(); gsBlob->Release(); psBlob->Release();
    if (!dbgReady) MGlobal::displayError("[GaussianSplat] Debug pipeline init failed.");
    return dbgReady;
}

// ---------------------------------------------------------------------------
// initProductionPipeline
// ---------------------------------------------------------------------------
bool GaussianDrawData::initProductionPipeline(ID3D11Device* device)
{
    HRESULT hr;

    // -- Compile compute shader --
    {
        ID3DBlob* csBlob = nullptr;
        if (!CompileStage(kPreprocessCS, strlen(kPreprocessCS),
                          "PreprocessKernel", "cs_5_0", &csBlob))
        {
            MGlobal::displayError("[GaussianSplat] Preprocessing CS compile failed.");
            return false;
        }
        hr = device->CreateComputeShader(csBlob->GetBufferPointer(),
                                         csBlob->GetBufferSize(), nullptr, &computeShader);
        csBlob->Release();
        if (FAILED(hr)) {
            MGlobal::displayError("[GaussianSplat] CreateComputeShader failed.");
            return false;
        }
    }

    // -- Compute constant buffer --
    {
        D3D11_BUFFER_DESC cbd = {};
        cbd.ByteWidth      = sizeof(CBPreprocess);
        cbd.Usage          = D3D11_USAGE_DYNAMIC;
        cbd.BindFlags      = D3D11_BIND_CONSTANT_BUFFER;
        cbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        hr = device->CreateBuffer(&cbd, nullptr, &computeCB);
        if (FAILED(hr)) return false;
    }

    // -- Compile render shaders (VS + PS only, no GS — instanced rendering) --
    size_t srcLen = strlen(kProdShaderSrc);
    ID3DBlob* vsBlob = nullptr, *psBlob = nullptr;
    if (!CompileStage(kProdShaderSrc, srcLen, "VS", "vs_5_0", &vsBlob)) goto prod_fail;
    if (!CompileStage(kProdShaderSrc, srcLen, "PS", "ps_5_0", &psBlob)) { vsBlob->Release(); goto prod_fail; }

    hr = device->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), nullptr, &prodVS);
    if (FAILED(hr)) goto prod_blob_cleanup;
    hr = device->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(), nullptr, &prodPS);
    if (FAILED(hr)) goto prod_blob_cleanup;

    {
        D3D11_BUFFER_DESC cbd = {};
        cbd.ByteWidth      = sizeof(CBRender);
        cbd.Usage          = D3D11_USAGE_DYNAMIC;
        cbd.BindFlags      = D3D11_BIND_CONSTANT_BUFFER;
        cbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        hr = device->CreateBuffer(&cbd, nullptr, &prodCB);
        if (FAILED(hr)) goto prod_blob_cleanup;
    }

    vsBlob->Release(); psBlob->Release();
    prodReady = true;
    return true;

prod_blob_cleanup:
    vsBlob->Release(); psBlob->Release();
prod_fail:
    MGlobal::displayError("[GaussianSplat] Production pipeline init failed.");
    return false;
}

// ---------------------------------------------------------------------------
// initSortPipeline  (compile 4 radix sort kernels with preprocessor defines)
// ---------------------------------------------------------------------------
bool GaussianDrawData::initSortPipeline(ID3D11Device* device)
{
    size_t srcLen = strlen(kRadixSortCS);

    struct KernelDef { const char* define; const char* entry; ID3D11ComputeShader** out; };
    KernelDef kernels[] = {
        { "KEYGEN_KERNEL",  "KeyGenKernel",  &sortCS_keygen  },
        { "COUNT_KERNEL",   "CountKernel",   &sortCS_count   },
        { "SCAN_KERNEL",    "ScanKernel",    &sortCS_scan    },
        { "SCATTER_KERNEL", "ScatterKernel", &sortCS_scatter },
    };

    for (auto& k : kernels) {
        D3D_SHADER_MACRO defines[] = { { k.define, "1" }, { nullptr, nullptr } };
        ID3DBlob* blob = nullptr;
        if (!CompileStage(kRadixSortCS, srcLen, k.entry, "cs_5_0", &blob, defines)) {
            MGlobal::displayError(MString("[GaussianSplat] Sort shader compile failed: ") + k.entry);
            return false;
        }
        HRESULT hr = device->CreateComputeShader(blob->GetBufferPointer(),
                                                  blob->GetBufferSize(), nullptr, k.out);
        blob->Release();
        if (FAILED(hr)) {
            MGlobal::displayError(MString("[GaussianSplat] CreateComputeShader failed: ") + k.entry);
            return false;
        }
    }

    // Sort constant buffer
    D3D11_BUFFER_DESC cbd = {};
    cbd.ByteWidth      = sizeof(CBSort);
    cbd.Usage          = D3D11_USAGE_DYNAMIC;
    cbd.BindFlags      = D3D11_BIND_CONSTANT_BUFFER;
    cbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    if (FAILED(device->CreateBuffer(&cbd, nullptr, &sortCB))) {
        MGlobal::displayError("[GaussianSplat] Sort CB creation failed.");
        return false;
    }

    sortReady = true;
    return true;
}

// ---------------------------------------------------------------------------
// createSortBuffers  (allocate ping-pong key/value + block histogram)
// ---------------------------------------------------------------------------
bool GaussianDrawData::createSortBuffers(ID3D11Device* device, uint32_t N)
{
    // Release old sort buffers (keep shaders)
    SAFE_RELEASE(sortKeysA); SAFE_RELEASE(sortKeysA_UAV); SAFE_RELEASE(sortKeysA_SRV);
    SAFE_RELEASE(sortKeysB); SAFE_RELEASE(sortKeysB_UAV); SAFE_RELEASE(sortKeysB_SRV);
    SAFE_RELEASE(sortValsA); SAFE_RELEASE(sortValsA_UAV); SAFE_RELEASE(sortValsA_SRV);
    SAFE_RELEASE(sortValsB); SAFE_RELEASE(sortValsB_UAV); SAFE_RELEASE(sortValsB_SRV);
    SAFE_RELEASE(sortBlockHist); SAFE_RELEASE(sortBlockHist_UAV); SAFE_RELEASE(sortBlockHist_SRV);

    uint32_t numBlocks = (N + kSortTileSize - 1) / kSortTileSize;

    if (!createUAVBuffer(device, "sortKeysA", N, sizeof(uint32_t),
                         &sortKeysA, &sortKeysA_UAV, &sortKeysA_SRV)) return false;
    if (!createUAVBuffer(device, "sortKeysB", N, sizeof(uint32_t),
                         &sortKeysB, &sortKeysB_UAV, &sortKeysB_SRV)) return false;
    if (!createUAVBuffer(device, "sortValsA", N, sizeof(uint32_t),
                         &sortValsA, &sortValsA_UAV, &sortValsA_SRV)) return false;
    if (!createUAVBuffer(device, "sortValsB", N, sizeof(uint32_t),
                         &sortValsB, &sortValsB_UAV, &sortValsB_SRV)) return false;
    if (!createUAVBuffer(device, "sortBlockHist", numBlocks * kRadixSize, sizeof(uint32_t),
                         &sortBlockHist, &sortBlockHist_UAV, &sortBlockHist_SRV)) return false;

    MGlobal::displayInfo(MString("[GaussianSplat] Sort buffers created: ") +
                         N + " elements, " + numBlocks + " blocks, ~" +
                         (unsigned int)(N * 4 * 4 / 1024 / 1024) + " MB");
    return true;
}

// ---------------------------------------------------------------------------
// initDepthPassPipeline  (compile 2 compute kernels + VS/PS + DS/blend state)
// ---------------------------------------------------------------------------
bool GaussianDrawData::initDepthPassPipeline(ID3D11Device* device)
{
    size_t csLen = strlen(kDepthPassCS);
    HRESULT hr;

    // -- Clear kernel --
    {
        D3D_SHADER_MACRO defs[] = { { "CLEAR_DEPTH_KERNEL", "1" }, { nullptr, nullptr } };
        ID3DBlob* blob = nullptr;
        if (!CompileStage(kDepthPassCS, csLen, "ClearDepthKernel", "cs_5_0", &blob, defs)) return false;
        hr = device->CreateComputeShader(blob->GetBufferPointer(), blob->GetBufferSize(),
                                          nullptr, &depthClearCS);
        blob->Release();
        if (FAILED(hr)) { MGlobal::displayError("[GaussianSplat] CreateComputeShader(ClearDepth) failed."); return false; }
    }

    // -- Depth pass kernel --
    {
        D3D_SHADER_MACRO defs[] = { { "DEPTH_PASS_KERNEL", "1" }, { nullptr, nullptr } };
        ID3DBlob* blob = nullptr;
        if (!CompileStage(kDepthPassCS, csLen, "DepthPassKernel", "cs_5_0", &blob, defs)) return false;
        hr = device->CreateComputeShader(blob->GetBufferPointer(), blob->GetBufferSize(),
                                          nullptr, &depthPassCS);
        blob->Release();
        if (FAILED(hr)) { MGlobal::displayError("[GaussianSplat] CreateComputeShader(DepthPass) failed."); return false; }
    }

    // -- Depth CB --
    {
        D3D11_BUFFER_DESC cbd = {};
        cbd.ByteWidth      = sizeof(CBDepth);
        cbd.Usage          = D3D11_USAGE_DYNAMIC;
        cbd.BindFlags      = D3D11_BIND_CONSTANT_BUFFER;
        cbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        if (FAILED(device->CreateBuffer(&cbd, nullptr, &depthCB))) return false;
    }

    // -- Copy pass VS/PS --
    {
        size_t vsLen = strlen(kDepthCopyShader);
        ID3DBlob* vsBlob = nullptr, *psBlob = nullptr;
        if (!CompileStage(kDepthCopyShader, vsLen, "CopyVS", "vs_5_0", &vsBlob)) return false;
        if (!CompileStage(kDepthCopyShader, vsLen, "CopyPS", "ps_5_0", &psBlob)) { vsBlob->Release(); return false; }

        hr = device->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), nullptr, &depthCopyVS);
        if (FAILED(hr)) { vsBlob->Release(); psBlob->Release(); return false; }
        hr = device->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(), nullptr, &depthCopyPS);
        vsBlob->Release(); psBlob->Release();
        if (FAILED(hr)) return false;
    }

    // -- Depth-write DS state (used by the copy pass only) --
    {
        D3D11_DEPTH_STENCIL_DESC dsd = {};
        dsd.DepthEnable    = TRUE;
        dsd.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
        dsd.DepthFunc      = D3D11_COMPARISON_LESS;   // only overwrite when GS is nearer
        dsd.StencilEnable  = FALSE;
        if (FAILED(device->CreateDepthStencilState(&dsd, &depthWriteDS))) return false;
    }

    // -- Blend state for copy pass: mask out color writes (depth only) --
    {
        D3D11_BLEND_DESC bd = {};
        bd.RenderTarget[0].BlendEnable           = FALSE;
        bd.RenderTarget[0].RenderTargetWriteMask = 0; // no color writes
        if (FAILED(device->CreateBlendState(&bd, &depthCopyBlend))) return false;
    }

    depthPassReady = true;
    return true;
}

// ---------------------------------------------------------------------------
// createDepthTexture  (UAV + SRV, R32_UINT, viewport sized)
// ---------------------------------------------------------------------------
bool GaussianDrawData::createDepthTexture(ID3D11Device* device, uint32_t w, uint32_t h)
{
    SAFE_RELEASE(depthTex);
    SAFE_RELEASE(depthTex_UAV);
    SAFE_RELEASE(depthTex_SRV);
    depthTexW = depthTexH = 0;

    D3D11_TEXTURE2D_DESC td = {};
    td.Width          = w;
    td.Height         = h;
    td.MipLevels      = 1;
    td.ArraySize      = 1;
    td.Format         = DXGI_FORMAT_R32_UINT;
    td.SampleDesc.Count = 1;
    td.Usage          = D3D11_USAGE_DEFAULT;
    td.BindFlags      = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;

    HRESULT hr = device->CreateTexture2D(&td, nullptr, &depthTex);
    if (FAILED(hr)) {
        MGlobal::displayError(
            MString("[GaussianSplat] CreateTexture2D(depthTex) failed: ") + (int)w + "x" + (int)h);
        return false;
    }

    D3D11_UNORDERED_ACCESS_VIEW_DESC uavd = {};
    uavd.Format        = DXGI_FORMAT_R32_UINT;
    uavd.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
    uavd.Texture2D.MipSlice = 0;
    if (FAILED(device->CreateUnorderedAccessView(depthTex, &uavd, &depthTex_UAV))) {
        SAFE_RELEASE(depthTex);
        return false;
    }

    D3D11_SHADER_RESOURCE_VIEW_DESC srvd = {};
    srvd.Format        = DXGI_FORMAT_R32_UINT;
    srvd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srvd.Texture2D.MipLevels = 1;
    if (FAILED(device->CreateShaderResourceView(depthTex, &srvd, &depthTex_SRV))) {
        SAFE_RELEASE(depthTex);
        SAFE_RELEASE(depthTex_UAV);
        return false;
    }

    depthTexW = w;
    depthTexH = h;
    MGlobal::displayInfo(MString("[GaussianSplat] Depth UAV texture created: ") + (int)w + "x" + (int)h);
    return true;
}

// ---------------------------------------------------------------------------
// createComputeOutputs
// ---------------------------------------------------------------------------
bool GaussianDrawData::createComputeOutputs(ID3D11Device* device, uint32_t N)
{
    // Release previous outputs
    SAFE_RELEASE(ubPositionSS); SAFE_RELEASE(uavPositionSS); SAFE_RELEASE(srvPositionSS);
    SAFE_RELEASE(ubDepth);      SAFE_RELEASE(uavDepth);      SAFE_RELEASE(srvDepth);
    SAFE_RELEASE(ubRadius);     SAFE_RELEASE(uavRadius);     SAFE_RELEASE(srvRadius);
    SAFE_RELEASE(ubColor);      SAFE_RELEASE(uavColor);      SAFE_RELEASE(srvColor);
    SAFE_RELEASE(ubCov2D);      SAFE_RELEASE(uavCov2D);      SAFE_RELEASE(srvCov2D);
    allocatedN = 0;

    if (!createUAVBuffer(device, "positionSS", N, sizeof(float)*2, &ubPositionSS, &uavPositionSS, &srvPositionSS)) return false;
    if (!createUAVBuffer(device, "depth",      N, sizeof(float),   &ubDepth,      &uavDepth,      &srvDepth))      return false;
    if (!createUAVBuffer(device, "radius",     N, sizeof(float),   &ubRadius,     &uavRadius,     &srvRadius))     return false;
    if (!createUAVBuffer(device, "color",      N, sizeof(float)*3, &ubColor,      &uavColor,      &srvColor))      return false;
    if (!createUAVBuffer(device, "cov2D",      N, sizeof(float)*4, &ubCov2D,      &uavCov2D,      &srvCov2D))      return false;

    // Allocate GPU sort buffers (ping-pong + block histogram)
    if (sortReady) {
        if (!createSortBuffers(device, N)) {
            MGlobal::displayError("[GaussianSplat] Sort buffer creation failed.");
            // Non-fatal: production pipeline won't render, falls back to debug
        }
    }

    allocatedN = N;
    return true;
}

// ===========================================================================
// GaussianDrawOverride
// ===========================================================================
MHWRender::MPxDrawOverride*
GaussianDrawOverride::creator(const MObject& obj) { return new GaussianDrawOverride(obj); }

GaussianDrawOverride::GaussianDrawOverride(const MObject& obj)
    : MHWRender::MPxDrawOverride(obj, GaussianDrawOverride::draw)
{
    MFnDependencyNode fn(obj);
    m_node = dynamic_cast<GaussianNode*>(fn.userNode());
}

MHWRender::DrawAPI GaussianDrawOverride::supportedDrawAPIs() const
{
    return MHWRender::kDirectX11;
}

bool GaussianDrawOverride::isBounded(const MDagPath& objPath, const MDagPath&) const
{
    if (!m_node) return false;
    GaussianDataNode* dn = m_node->findConnectedDataNode();
    return (dn && dn->hasData());
}

MBoundingBox GaussianDrawOverride::boundingBox(const MDagPath& objPath, const MDagPath&) const
{
    if (!m_node) return MBoundingBox(MPoint(-1,-1,-1), MPoint(1,1,1));
    GaussianDataNode* dn = m_node->findConnectedDataNode();
    if (!dn || !dn->hasData()) return MBoundingBox(MPoint(-1,-1,-1), MPoint(1,1,1));
    return dn->boundingBox();
}

// ---------------------------------------------------------------------------
// prepareForDraw
// ---------------------------------------------------------------------------
MUserData* GaussianDrawOverride::prepareForDraw(
    const MDagPath&                 objPath,
    const MDagPath&                 /*cameraPath*/,
    const MHWRender::MFrameContext& frameContext,
    MUserData*                      oldData)
{
    GaussianDrawData* data = dynamic_cast<GaussianDrawData*>(oldData);
    if (!data) data = new GaussianDrawData();
    if (!m_node) return data;

    MHWRender::MRenderer* renderer = MHWRender::MRenderer::theRenderer();
    ID3D11Device* device = static_cast<ID3D11Device*>(renderer->GPUDeviceHandle());

    // Init pipelines (lazy, once)
    if (!data->dbgReady  && device) {
        if (data->initDebugPipeline(device))
            MGlobal::displayInfo("[GaussianSplat] Debug pipeline: OK");
        else
            MGlobal::displayError("[GaussianSplat] Debug pipeline: FAILED");
    }
    if (!data->prodReady && device) {
        if (data->initProductionPipeline(device))
            MGlobal::displayInfo("[GaussianSplat] Production pipeline: OK");
        else
            MGlobal::displayError("[GaussianSplat] Production pipeline: FAILED");
    }
    if (!data->sortReady && device) {
        if (data->initSortPipeline(device))
            MGlobal::displayInfo("[GaussianSplat] Sort pipeline: OK");
        else
            MGlobal::displayError("[GaussianSplat] Sort pipeline: FAILED");
    }
    if (!data->depthPassReady && device) {
        if (data->initDepthPassPipeline(device))
            MGlobal::displayInfo("[GaussianSplat] Depth pass pipeline: OK");
        else
            MGlobal::displayError("[GaussianSplat] Depth pass pipeline: FAILED");
    }

    // Find connected data node
    GaussianDataNode* dataNode = m_node->findConnectedDataNode();

    // Clear shared SRV refs each frame (re-fetch below if data node exists)
    data->sharedSrvPositionWS = nullptr;
    data->sharedSrvScale      = nullptr;
    data->sharedSrvRotation   = nullptr;
    data->sharedSrvOpacity    = nullptr;
    data->sharedSrvSHCoeffs   = nullptr;
    data->inputsReady         = false;
    data->vertexCount         = 0;

    if (!dataNode) return data;

    // Trigger data node compute (DG evaluation)
    MPlug dataReadyPlug(dataNode->thisMObject(), GaussianDataNode::aDataReady);
    dataReadyPlug.asBool();

    if (!dataNode->hasData()) return data;

    // Lazy GPU upload of shared input buffers (in data node)
    if (device) {
        dataNode->uploadInputBuffersIfNeeded(device);
    }

    if (!dataNode->areInputsReady()) return data;

    // Copy non-owning SRV pointers from data node
    data->sharedSrvPositionWS = dataNode->srvPositionWS();
    data->sharedSrvScale      = dataNode->srvScale();
    data->sharedSrvRotation   = dataNode->srvRotation();
    data->sharedSrvOpacity    = dataNode->srvOpacity();
    data->sharedSrvSHCoeffs   = dataNode->srvSHCoeffs();
    data->inputsReady         = true;

    uint32_t N = dataNode->splatCount();
    data->vertexCount = N;

    // Allocate per-instance compute outputs + sort buffers if count changed
    if (data->allocatedN != N && device) {
        if (data->createComputeOutputs(device, N))
            MGlobal::displayInfo(MString("[GaussianSplat] Compute outputs created: allocatedN=") + data->allocatedN);
        else
            MGlobal::displayError("[GaussianSplat] createComputeOutputs FAILED");
    }

    // Matrices
    MStatus status;
    MMatrix world    = objPath.inclusiveMatrix();
    MMatrix viewMat  = frameContext.getMatrix(MHWRender::MFrameContext::kViewMtx, &status);
    MMatrix projMat  = frameContext.getMatrix(MHWRender::MFrameContext::kProjectionMtx, &status);
    MMatrix viewProj = frameContext.getMatrix(MHWRender::MFrameContext::kViewProjMtx, &status);
    MMatrix wvp      = world * viewProj;

    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) {
            data->wvp[r*4+c]      = (float)wvp[r][c];
            data->worldMat[r*4+c] = (float)world[r][c];
            data->viewMat[r*4+c]  = (float)viewMat[r][c];
            data->projMat[r*4+c]  = (float)projMat[r][c];
        }

    // Camera position (world space)
    MMatrix invView = viewMat.inverse();
    data->cameraPos[0] = (float)invView[3][0];
    data->cameraPos[1] = (float)invView[3][1];
    data->cameraPos[2] = (float)invView[3][2];

    // Viewport
    int ox, oy, vpW, vpH;
    frameContext.getViewportDimensions(ox, oy, vpW, vpH);
    data->vpWidth  = (float)vpW;
    data->vpHeight = (float)vpH;

    // Recreate depth UAV texture if viewport size changed
    if (device && data->depthPassReady &&
        ((uint32_t)vpW != data->depthTexW || (uint32_t)vpH != data->depthTexH) &&
        vpW > 0 && vpH > 0)
    {
        data->createDepthTexture(device, (uint32_t)vpW, (uint32_t)vpH);
    }

    // tanHalfFov
    data->tanHalfFov[0] = (float)(1.0 / projMat[0][0]);
    data->tanHalfFov[1] = (float)(1.0 / projMat[1][1]);

    // Point size
    MPlug psPlug(m_node->thisMObject(), GaussianNode::aPointSize);
    data->pointSize = psPlug.asFloat();

    // Render mode (0=auto, 1=debug, 2=production)
    MPlug rmPlug(m_node->thisMObject(), GaussianNode::aRenderMode);
    data->renderMode = rmPlug.asInt();

    // One-time diagnostic
    {
        static bool diagDone = false;
        if (!diagDone && data->vertexCount > 0) {
            diagDone = true;
            MString msg("[GaussianSplat] DIAG: prodReady=");
            msg += (int)data->prodReady;
            msg += " sortReady=";   msg += (int)data->sortReady;
            msg += " dbgReady=";    msg += (int)data->dbgReady;
            msg += " inputsReady="; msg += (int)data->inputsReady;
            msg += " allocatedN=";  msg += data->allocatedN;
            msg += " vertexCount="; msg += data->vertexCount;
            msg += " sortValsA=";   msg += (data->sortValsA_SRV ? "yes" : "null");
            MGlobal::displayInfo(msg);

            // Matrix diagnostics
            MString m2("[GaussianSplat] projMat diag: ");
            m2 += data->projMat[0]; m2 += ", ";
            m2 += data->projMat[5]; m2 += ", ";
            m2 += data->projMat[10]; m2 += ", ";
            m2 += data->projMat[15];
            MGlobal::displayInfo(m2);

            MString m3("[GaussianSplat] tanHalfFov: ");
            m3 += data->tanHalfFov[0]; m3 += ", ";
            m3 += data->tanHalfFov[1];
            m3 += " viewport: ";
            m3 += data->vpWidth; m3 += "x"; m3 += data->vpHeight;
            MGlobal::displayInfo(m3);

            MString m4("[GaussianSplat] viewMat row2: ");
            m4 += data->viewMat[8]; m4 += ", ";
            m4 += data->viewMat[9]; m4 += ", ";
            m4 += data->viewMat[10]; m4 += ", ";
            m4 += data->viewMat[11];
            MGlobal::displayInfo(m4);

            MString m5("[GaussianSplat] cameraPos: ");
            m5 += data->cameraPos[0]; m5 += ", ";
            m5 += data->cameraPos[1]; m5 += ", ";
            m5 += data->cameraPos[2];
            MGlobal::displayInfo(m5);

            // Focal length check
            float focalX = 0.5f * data->vpWidth / data->tanHalfFov[0];
            float focalY = 0.5f * data->vpHeight / data->tanHalfFov[1];
            MString m6("[GaussianSplat] focalPx: ");
            m6 += focalX; m6 += ", "; m6 += focalY;
            MGlobal::displayInfo(m6);
        }
    }

    return data;
}

// ---------------------------------------------------------------------------
// draw
// ---------------------------------------------------------------------------
void GaussianDrawOverride::draw(const MHWRender::MDrawContext& context,
                                const MUserData*               userData)
{
    const GaussianDrawData* data = static_cast<const GaussianDrawData*>(userData);
    if (!data || data->vertexCount == 0) return;

    MHWRender::MRenderer* renderer = MHWRender::MRenderer::theRenderer();
    ID3D11Device* device = static_cast<ID3D11Device*>(renderer->GPUDeviceHandle());
    if (!device) return;

    ID3D11DeviceContext* ctx = nullptr;
    device->GetImmediateContext(&ctx);
    if (!ctx) return;

    // Save Maya DX11 state
    ID3D11BlendState*       prevBlend  = nullptr; float prevBF[4]; UINT prevSM;
    ID3D11RasterizerState*  prevRS     = nullptr;
    ID3D11DepthStencilState* prevDS    = nullptr; UINT prevDSRef;
    ID3D11InputLayout*      prevLayout = nullptr;
    ID3D11VertexShader*     prevVS     = nullptr;
    ID3D11GeometryShader*   prevGS     = nullptr;
    ID3D11PixelShader*      prevPS     = nullptr;
    ctx->OMGetBlendState(&prevBlend, prevBF, &prevSM);
    ctx->RSGetState(&prevRS);
    ctx->OMGetDepthStencilState(&prevDS, &prevDSRef);
    ctx->IAGetInputLayout(&prevLayout);
    ctx->VSGetShader(&prevVS, nullptr, nullptr);
    ctx->GSGetShader(&prevGS, nullptr, nullptr);
    ctx->PSGetShader(&prevPS, nullptr, nullptr);

    float blendFactor[] = { 1.f, 1.f, 1.f, 1.f };
    ctx->OMSetBlendState(data->blendState, blendFactor, 0xFFFFFFFF);
    ctx->RSSetState(data->rsState);
    ctx->OMSetDepthStencilState(data->dsState, 0);

    // -----------------------------------------------------------------------
    // Determine which render path to use
    // -----------------------------------------------------------------------
    bool canProd = data->prodReady && data->sortReady && data->inputsReady
                   && data->allocatedN == data->vertexCount
                   && data->sortValsA_SRV;
    bool canDbg  = data->dbgReady && data->inputsReady;

    bool useProd = false;
    if (data->renderMode == 1)      useProd = false;         // force debug
    else if (data->renderMode == 2) useProd = canProd;       // force production
    else if (data->renderMode == 3) useProd = canProd;       // diagnostic: prod with fixed radius
    else                            useProd = canProd;       // auto

    // One-time draw-path diagnostic
    {
        static int drawDiagCount = 0;
        if (drawDiagCount < 3) {
            drawDiagCount++;
            MString msg("[GaussianSplat] DRAW frame ");
            msg += drawDiagCount;
            msg += ": useProd="; msg += (int)useProd;
            msg += " canProd=";  msg += (int)canProd;
            msg += " canDbg=";   msg += (int)canDbg;
            msg += " renderMode="; msg += data->renderMode;
            msg += " N=";       msg += data->vertexCount;
            MGlobal::displayInfo(msg);
        }
    }

    // -----------------------------------------------------------------------
    // Production path  (compute preprocess + GPU radix sort + ellipse render)
    // -----------------------------------------------------------------------
    if (useProd)
    {
        uint32_t N = data->vertexCount;

        // -- 1. Update compute constant buffer --
        {
            D3D11_MAPPED_SUBRESOURCE mapped;
            if (SUCCEEDED(ctx->Map(data->computeCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
                CBPreprocess* cb = static_cast<CBPreprocess*>(mapped.pData);
                std::memcpy(cb->worldMat,   data->worldMat,   64);
                std::memcpy(cb->viewMat,    data->viewMat,    64);
                std::memcpy(cb->projMat,    data->projMat,    64);
                std::memcpy(cb->cameraPos,  data->cameraPos,  12);
                cb->padding0    = 0.f;
                cb->tanHalfFov[0] = data->tanHalfFov[0];
                cb->tanHalfFov[1] = data->tanHalfFov[1];
                cb->filmWidth   = (int)data->vpWidth;
                cb->filmHeight  = (int)data->vpHeight;
                cb->gaussCount  = N;
                cb->debugFixedRadius = (data->renderMode == 3) ? 5 : 0;
                cb->pad2 = cb->pad3 = 0.f;
                ctx->Unmap(data->computeCB, 0);
            }
        }

        // -- 2. Dispatch Preprocessing compute shader --
        {
            ID3D11ShaderResourceView* srvs[] = {
                data->sharedSrvPositionWS, data->sharedSrvScale, data->sharedSrvRotation,
                data->sharedSrvOpacity, data->sharedSrvSHCoeffs
            };
            ID3D11UnorderedAccessView* uavs[] = {
                data->uavPositionSS, data->uavDepth, data->uavRadius,
                data->uavColor, data->uavCov2D
            };
            ctx->CSSetShader(data->computeShader, nullptr, 0);
            ctx->CSSetConstantBuffers(0, 1, &data->computeCB);
            ctx->CSSetShaderResources(0, 5, srvs);
            ctx->CSSetUnorderedAccessViews(0, 5, uavs, nullptr);

            ctx->Dispatch((N + 255) / 256, 1, 1);

            ID3D11UnorderedAccessView* nullUAVs5[5] = {};
            ctx->CSSetUnorderedAccessViews(0, 5, nullUAVs5, nullptr);
            ID3D11ShaderResourceView* nullSRVs5[5] = {};
            ctx->CSSetShaderResources(0, 5, nullSRVs5);
        }

        // -- 3. GPU Radix Sort (depth back-to-front) --
        {
            uint32_t numBlocks = (N + kSortTileSize - 1) / kSortTileSize;

            // 3a. Key generation: float depth -> sortable uint + init indices
            {
                CBSort scb = { N, numBlocks, 0, 0 };
                D3D11_MAPPED_SUBRESOURCE mapped;
                if (SUCCEEDED(ctx->Map(data->sortCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
                    std::memcpy(mapped.pData, &scb, sizeof(scb));
                    ctx->Unmap(data->sortCB, 0);
                }
                ctx->CSSetShader(data->sortCS_keygen, nullptr, 0);
                ctx->CSSetConstantBuffers(0, 1, &data->sortCB);
                ID3D11ShaderResourceView* kgSRV[] = { data->srvDepth };
                ctx->CSSetShaderResources(0, 1, kgSRV);
                ID3D11UnorderedAccessView* kgUAV[] = { data->sortKeysA_UAV, data->sortValsA_UAV };
                ctx->CSSetUnorderedAccessViews(0, 2, kgUAV, nullptr);
                ctx->Dispatch((N + kSortGroupSize - 1) / kSortGroupSize, 1, 1);
                // Unbind
                ID3D11ShaderResourceView*  nullSRV1[1] = {};
                ID3D11UnorderedAccessView* nullUAV2[2] = {};
                ctx->CSSetShaderResources(0, 1, nullSRV1);
                ctx->CSSetUnorderedAccessViews(0, 2, nullUAV2, nullptr);
            }

            // 3b. Four radix passes (8-bit digit each)
            for (uint32_t pass = 0; pass < 4; pass++) {
                bool even = (pass % 2 == 0);
                auto keysInSRV  = even ? data->sortKeysA_SRV  : data->sortKeysB_SRV;
                auto keysOutUAV = even ? data->sortKeysB_UAV  : data->sortKeysA_UAV;
                auto valsInSRV  = even ? data->sortValsA_SRV  : data->sortValsB_SRV;
                auto valsOutUAV = even ? data->sortValsB_UAV  : data->sortValsA_UAV;

                // Update sort CB for this pass
                {
                    CBSort scb = { N, numBlocks, pass * 8, 0 };
                    D3D11_MAPPED_SUBRESOURCE mapped;
                    if (SUCCEEDED(ctx->Map(data->sortCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
                        std::memcpy(mapped.pData, &scb, sizeof(scb));
                        ctx->Unmap(data->sortCB, 0);
                    }
                }

                // Count (histogram per block)
                {
                    ctx->CSSetShader(data->sortCS_count, nullptr, 0);
                    ctx->CSSetConstantBuffers(0, 1, &data->sortCB);
                    ctx->CSSetShaderResources(0, 1, &keysInSRV);
                    ctx->CSSetUnorderedAccessViews(0, 1, &data->sortBlockHist_UAV, nullptr);
                    ctx->Dispatch(numBlocks, 1, 1);
                    ID3D11ShaderResourceView*  n1[1] = {};
                    ID3D11UnorderedAccessView* n1u[1] = {};
                    ctx->CSSetShaderResources(0, 1, n1);
                    ctx->CSSetUnorderedAccessViews(0, 1, n1u, nullptr);
                }

                // Scan (prefix sum across blocks per digit)
                {
                    ctx->CSSetShader(data->sortCS_scan, nullptr, 0);
                    ctx->CSSetConstantBuffers(0, 1, &data->sortCB);
                    ctx->CSSetUnorderedAccessViews(0, 1, &data->sortBlockHist_UAV, nullptr);
                    ctx->Dispatch(1, 1, 1);
                    ID3D11UnorderedAccessView* n1u[1] = {};
                    ctx->CSSetUnorderedAccessViews(0, 1, n1u, nullptr);
                }

                // Scatter (redistribute by digit)
                {
                    ctx->CSSetShader(data->sortCS_scatter, nullptr, 0);
                    ctx->CSSetConstantBuffers(0, 1, &data->sortCB);
                    ID3D11ShaderResourceView* scSRV[] = { keysInSRV, valsInSRV, data->sortBlockHist_SRV };
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
            // After 4 passes (even count), sorted result is back in A
            ctx->CSSetShader(nullptr, nullptr, 0);
        }

        // -- 4. Update render constant buffer --
        {
            D3D11_MAPPED_SUBRESOURCE mapped;
            if (SUCCEEDED(ctx->Map(data->prodCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
                CBRender* cb = static_cast<CBRender*>(mapped.pData);
                cb->vpWidth  = data->vpWidth;
                cb->vpHeight = data->vpHeight;
                cb->pad[0]   = cb->pad[1] = 0.f;
                ctx->Unmap(data->prodCB, 0);
            }
        }

        // -- 5. Render pass (sorted via gSortedIndices SRV) --
        {
            ID3D11ShaderResourceView* vsSRVs[] = {
                data->srvPositionSS, data->srvRadius, data->srvColor,
                data->srvCov2D, data->srvDepth, data->sortValsA_SRV
            };
            ctx->IASetInputLayout(nullptr);
            ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
            ctx->VSSetShader(data->prodVS, nullptr, 0);
            ctx->VSSetConstantBuffers(0, 1, &data->prodCB);
            ctx->VSSetShaderResources(0, 6, vsSRVs);
            ctx->GSSetShader(nullptr, nullptr, 0);
            ctx->PSSetShader(data->prodPS, nullptr, 0);
            ctx->DrawInstanced(4, N, 0, 0);

            ID3D11ShaderResourceView* nullSRVs6[6] = {};
            ctx->VSSetShaderResources(0, 6, nullSRVs6);
        }

        // -- 6. Depth pass: write per-pixel representative depth into a UAV
        //         texture via compute (atomic min), then copy that into Maya's
        //         real depth buffer so GS can occlude Maya transparent/virtual
        //         geometry drawn later in the frame.
        //         Runs AFTER the main alpha-blend render to avoid self-occlusion.
        if (data->depthPassReady && data->depthTex_UAV && data->depthTex_SRV &&
            data->depthTexW > 0 && data->depthTexH > 0)
        {
            uint32_t W = data->depthTexW;
            uint32_t H = data->depthTexH;

            // 6a. Update depth CB
            {
                D3D11_MAPPED_SUBRESOURCE mapped;
                if (SUCCEEDED(ctx->Map(data->depthCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
                    CBDepth* cb = static_cast<CBDepth*>(mapped.pData);
                    cb->viewportW      = W;
                    cb->viewportH      = H;
                    cb->splatCount     = N;
                    cb->radiusCap      = 16;     // per-splat AABB iteration cap
                    cb->alphaThreshold = 0.5f;   // only dense splats contribute depth
                    cb->pad0 = cb->pad1 = cb->pad2 = 0.f;
                    ctx->Unmap(data->depthCB, 0);
                }
            }

            // 6b. Clear depth UAV to asuint(1.0f)
            {
                ctx->CSSetShader(data->depthClearCS, nullptr, 0);
                ctx->CSSetConstantBuffers(0, 1, &data->depthCB);
                ctx->CSSetUnorderedAccessViews(0, 1, &data->depthTex_UAV, nullptr);
                ctx->Dispatch((W + 15) / 16, (H + 15) / 16, 1);
            }

            // 6c. Depth pass: each thread (one per splat) atomic-mins its
            //       NDC depth into the pixels its Gaussian covers.
            {
                ID3D11ShaderResourceView* srvs[] = {
                    data->srvPositionSS, data->srvRadius,
                    data->srvDepth,      data->srvCov2D
                };
                ctx->CSSetShader(data->depthPassCS, nullptr, 0);
                ctx->CSSetConstantBuffers(0, 1, &data->depthCB);
                ctx->CSSetShaderResources(0, 4, srvs);
                // UAV (slot 0) still bound from clear pass
                ctx->Dispatch((N + 255) / 256, 1, 1);

                // Unbind CS resources
                ID3D11ShaderResourceView*  nullSRV4[4] = {};
                ID3D11UnorderedAccessView* nullUAV1[1] = {};
                ctx->CSSetShaderResources(0, 4, nullSRV4);
                ctx->CSSetUnorderedAccessViews(0, 1, nullUAV1, nullptr);
                ctx->CSSetShader(nullptr, nullptr, 0);
            }

            // 6d. Copy pass: full-screen triangle writes asfloat(bits) to
            //       SV_Depth. DepthFunc=LESS ensures Maya's already-written
            //       opaque depth still wins where opaque geo is in front.
            {
                float copyBF[] = { 1.f, 1.f, 1.f, 1.f };
                ctx->OMSetBlendState(data->depthCopyBlend, copyBF, 0xFFFFFFFF);
                ctx->OMSetDepthStencilState(data->depthWriteDS, 0);

                ctx->IASetInputLayout(nullptr);
                ID3D11Buffer* nullVB[1] = {};
                UINT nullStride[1] = { 0 };
                UINT nullOffset[1] = { 0 };
                ctx->IASetVertexBuffers(0, 1, nullVB, nullStride, nullOffset);
                ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

                ctx->VSSetShader(data->depthCopyVS, nullptr, 0);
                ctx->GSSetShader(nullptr, nullptr, 0);
                ctx->PSSetShader(data->depthCopyPS, nullptr, 0);
                ctx->PSSetShaderResources(0, 1, &data->depthTex_SRV);
                ctx->Draw(3, 0);

                // Unbind copy-pass resources
                ID3D11ShaderResourceView* nullPSSRV[1] = {};
                ctx->PSSetShaderResources(0, 1, nullPSSRV);
            }
        }
    }
    // -----------------------------------------------------------------------
    // Debug fallback path  (circles from shared StructuredBuffers)
    // -----------------------------------------------------------------------
    else if (canDbg)
    {
        // Update debug CB
        {
            D3D11_MAPPED_SUBRESOURCE mapped;
            if (SUCCEEDED(ctx->Map(data->dbgCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
                CBDebug* cb = static_cast<CBDebug*>(mapped.pData);
                std::memcpy(cb->wvp, data->wvp, sizeof(cb->wvp));
                cb->pointSize = data->pointSize;
                cb->vpWidth   = data->vpWidth;
                cb->vpHeight  = data->vpHeight;
                cb->shStride  = (uint32_t)kSHCoeffsPerSplat;
                ctx->Unmap(data->dbgCB, 0);
            }
        }

        // Bind shared StructuredBuffers as VS SRVs
        ID3D11ShaderResourceView* vsSRVs[] = {
            data->sharedSrvPositionWS, data->sharedSrvOpacity, data->sharedSrvSHCoeffs
        };
        ctx->IASetInputLayout(nullptr);
        ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);
        ctx->VSSetShader(data->dbgVS, nullptr, 0);
        ctx->VSSetConstantBuffers(0, 1, &data->dbgCB);
        ctx->VSSetShaderResources(0, 3, vsSRVs);
        ctx->GSSetShader(data->dbgGS, nullptr, 0);
        ctx->GSSetConstantBuffers(0, 1, &data->dbgCB);
        ctx->PSSetShader(data->dbgPS, nullptr, 0);
        ctx->Draw(data->vertexCount, 0);

        // Unbind
        ID3D11ShaderResourceView* nullSRVs[3] = {};
        ctx->VSSetShaderResources(0, 3, nullSRVs);
    }

    // Restore Maya state
    ctx->IASetInputLayout(prevLayout);
    ctx->VSSetShader(prevVS, nullptr, 0);
    ctx->GSSetShader(prevGS, nullptr, 0);
    ctx->PSSetShader(prevPS, nullptr, 0);
    ctx->OMSetBlendState(prevBlend, prevBF, prevSM);
    ctx->RSSetState(prevRS);
    ctx->OMSetDepthStencilState(prevDS, prevDSRef);

    auto safeRelease = [](IUnknown* p) { if (p) p->Release(); };
    safeRelease(prevBlend); safeRelease(prevRS); safeRelease(prevDS);
    safeRelease(prevLayout); safeRelease(prevVS); safeRelease(prevGS); safeRelease(prevPS);
    ctx->Release();
}
