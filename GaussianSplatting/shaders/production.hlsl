// Production render: instanced ellipse splat (4 vertices per splat).
// VS reads sorted indices and per-splat preprocess outputs;
// PS evaluates Gaussian alpha and tints selected splats.

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
