// Marquee selection CS (currently unused: gsMarqueeSelect runs on CPU,
// see GaussianRenderManager::runSelection). Kept for parity / future GPU path.
//
// Project each splat through worldMat then viewProj; test if NDC is inside
// [rectMin, rectMax]; update mask bit 0 per `mode`. Bit 1 (deleted) preserved.

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
