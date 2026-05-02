// Debug pipeline (renderMode=1 or production fallback): point->quad via GS,
// flat circle splat with degree-0 SH color and sigmoid opacity.

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
