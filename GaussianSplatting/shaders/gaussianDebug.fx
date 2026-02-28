// gaussianDebug.fx
// Debug point-cloud shader for Gaussian Splatting, DirectX 11 / Maya VP2.
//
// Pipeline:  VS (transform) -> GS (expand point -> screen-aligned quad) -> PS (circle + color)
//
// NOTE: This is the *debug* pass only.  The production Gaussian splatting
//       pass will use a DX12 compute shader (SM 6.5) for depth-sorted alpha
//       blending and will replace the GS here with a proper splat rasterizer.

// ---------------------------------------------------------------------------
// Constant buffer / semantics
// ---------------------------------------------------------------------------
float4x4 gWVPXf     : WorldViewProjection;
float4x4 gProjXf    : Projection;          // needed to unproject half-size to NDC
float    gPointSize = 4.0;                 // screen-space radius in pixels
float2   gViewportSize = { 1280.0, 720.0 };// set from Maya's render target size

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------
struct VS_IN {
    float3 Pm : POSITION;
    float4 Cm : COLOR0;   // rgb = base SH colour, a = sigmoid(opacity)
};

struct VS_OUT {
    float4 clipPos : SV_Position;
    float4 color   : COLOR0;
};

struct GS_OUT {
    float4 clipPos  : SV_Position;
    float4 color    : COLOR0;
    float2 quadUV   : TEXCOORD0;  // [-1,1] within the quad, used for circle clip
};

// ---------------------------------------------------------------------------
// Vertex Shader  —  just transform and forward
// ---------------------------------------------------------------------------
VS_OUT VS(VS_IN input) {
    VS_OUT o;
    o.clipPos = mul(float4(input.Pm, 1.0f), gWVPXf);
    o.color   = input.Cm;
    return o;
}

// ---------------------------------------------------------------------------
// Geometry Shader  —  expand one point into a screen-aligned quad (2 tris)
// ---------------------------------------------------------------------------
[maxvertexcount(4)]
void GS(point VS_OUT input[1],
        inout TriangleStream<GS_OUT> stream)
{
    float4 center = input[0].clipPos;
    float4 col    = input[0].color;

    // Convert half-size from pixels to NDC space
    float2 halfSizeNDC = float2(gPointSize / gViewportSize.x,
                                gPointSize / gViewportSize.y) * center.w;

    // Quad corners in NDC offsets (CCW)
    static const float2 corners[4] = {
        float2(-1.0f,  1.0f),   // top-left
        float2( 1.0f,  1.0f),   // top-right
        float2(-1.0f, -1.0f),   // bottom-left
        float2( 1.0f, -1.0f),   // bottom-right
    };

    GS_OUT o;
    o.color = col;
    [unroll]
    for (int i = 0; i < 4; i++) {
        o.clipPos = center + float4(corners[i] * halfSizeNDC, 0.0f, 0.0f);
        o.quadUV  = corners[i];
        stream.Append(o);
    }
    stream.RestartStrip();
}

// ---------------------------------------------------------------------------
// Pixel Shader  —  circular mask + soft edge fade
// ---------------------------------------------------------------------------
float4 PS(GS_OUT input) : SV_Target
{
    float r2 = dot(input.quadUV, input.quadUV);
    clip(1.0f - r2);                    // discard outside the circle

    float alpha = input.color.a * (1.0f - r2 * 0.4f);  // soft edge
    return float4(input.color.rgb, alpha);
}

// ---------------------------------------------------------------------------
// Technique (DX11)
// ---------------------------------------------------------------------------
technique11 Main {
    pass p0 {
        SetVertexShader  (CompileShader(vs_5_0, VS()));
        SetGeometryShader(CompileShader(gs_5_0, GS()));
        SetPixelShader   (CompileShader(ps_5_0, PS()));
    }
}
