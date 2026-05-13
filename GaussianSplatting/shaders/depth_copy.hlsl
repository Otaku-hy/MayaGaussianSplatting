// Full-screen triangle that writes the GS-pass depth (R32_UINT) into the
// scene's depth buffer via SV_Depth. PS is rendered with no color writes.

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
