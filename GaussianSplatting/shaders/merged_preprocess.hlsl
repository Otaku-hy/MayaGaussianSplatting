// Merged preprocessing CS.
//   - worldMat is per-splat via gInstanceID -> gWorldMats[]
//   - Outputs: positionSS, depth, radius, color, cov2D+opacity
//   - Skips deleted splats (mask bit 1) by emitting radius=0

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
