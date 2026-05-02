// 4-pass 8-bit radix sort. Each pass compiled with a different define:
//   KEYGEN_KERNEL  -> KeyGenKernel
//   COUNT_KERNEL   -> CountKernel
//   SCAN_KERNEL    -> ScanKernel
//   SCATTER_KERNEL -> ScatterKernel

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
