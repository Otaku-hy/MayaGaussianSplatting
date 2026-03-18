#pragma once

#define GROUP_SIZE 256
// MAX_WAVES = GROUP_SIZE / min_laneCount = 64 / 8 = 8
// (groupshared array must be sized at compile-time)
#define MAX_WAVES  8



/////////////////////////////
/*****Reduce & Scan*****/
/////////////////////////////

groupshared uint sBlockReducesMem[MAX_WAVES];
groupshared uint sBlockScansMem[MAX_WAVES];

uint WavePrefixSumExclusive(uint value)
{
    uint pfx = WavePrefixSum(value);
    return pfx;
}

uint WavePrefixSumInclusive(uint value)
{
    uint pfx = WavePrefixSum(value);
    uint inc = pfx + value;
    return inc;
}

uint BlockReduce(uint threadIdx, uint value)
{
    uint waveIdx = threadIdx / WaveGetLaneCount();

    uint warpSum = WaveActiveSum(value);
    if (WaveIsFirstLane())
        sBlockReducesMem[waveIdx] = warpSum;

    GroupMemoryBarrierWithGroupSync();

    uint waveVal = WaveGetLaneIndex() < MAX_WAVES ? sBlockReducesMem[WaveGetLaneIndex()] : 0;
    uint blockSum = WaveActiveSum(waveVal);

    return blockSum;
}

uint BlockScanExclusive(uint threadIdx, uint value)
{
    uint waveIdx = threadIdx / WaveGetLaneCount();

    uint pfxInWarp = WavePrefixSumExclusive(value);
    if (WaveGetLaneIndex() == (WaveGetLaneCount() - 1))
        sBlockScansMem[waveIdx] = pfxInWarp + value;

    GroupMemoryBarrierWithGroupSync();

    uint waveVal = WaveGetLaneIndex() < MAX_WAVES ? sBlockScansMem[WaveGetLaneIndex()] : 0;
    uint warpPfx = WavePrefixSum(waveVal);
    uint currentWarpPfx = WaveReadLaneAt(warpPfx, waveIdx);

    return currentWarpPfx + pfxInWarp;
}

uint BlockScanInclusive(uint threadIdx, uint value)
{
    uint waveIdx = threadIdx / WaveGetLaneCount();

    uint pfxInWarp = WavePrefixSumInclusive(value);
    if (WaveGetLaneIndex() == (WaveGetLaneCount() - 1))
        sBlockScansMem[waveIdx] = pfxInWarp;

    GroupMemoryBarrierWithGroupSync();

    uint waveVal = WaveGetLaneIndex() < MAX_WAVES ? sBlockScansMem[WaveGetLaneIndex()] : 0;
    uint warpPfx = WavePrefixSum(waveVal);
    uint currentWarpPfx = WaveReadLaneAt(warpPfx, waveIdx);

    return currentWarpPfx + pfxInWarp;
}

// one uint per block, represent the state of current block
// || 2bits - flag || 30bits - sum ||
// flags: 00 - X not available, 01 - A block sum avaiialbe, 10 - P prefix sum available

inline uint PackBlockState(uint flag, uint sum)
{
    return (flag << 30) | (sum & 0x3FFFFFFF);
}

inline uint UnpackBlockFlag(uint packed)
{
    return packed >> 30;
}

inline uint UnpackBlockSum(uint packed)
{
    return packed & 0x3FFFFFFF;
}

///////////////////////////////
/********Wave extension********/
//////////////////////////////

// Note: Any operation without "active" prefix will perform undefined on inactive lanes, so we need to mask them out manually
////// NEED opt!!!
#define IMPLEMENT_WAVEMATCH(T) \
uint WaveMatch32(T value){ \
    uint activeMask = WaveActiveBallot(true).x; \
    uint matchMask = 0; \
    uint waveCount = WaveGetLaneCount(); \
    for (uint i = 0; i < waveCount; i++) { \
        T laneVal = WaveReadLaneAt(value, i); \
        matchMask = matchMask | (laneVal == value ? (1 << i) : 0); \
    } \
    return matchMask & activeMask; \
}

// optimized version for 8-bit data - only used for shader model 6.5 is not supported
uint WaveMatch32_8bits(uint value){
    uint matchMask = WaveActiveBallot(true).x;
    for (uint bit = 0; bit < 8; bit++){
        bool predicate = (value & (1 << bit)) != 0;
        uint bitMatchMask = WaveActiveBallot(predicate).x;
        bitMatchMask = predicate ? bitMatchMask : ~bitMatchMask;
        matchMask = matchMask & bitMatchMask;
    }
    return matchMask;
}

uint WaveMultiPrefixCountBits32(bool bit, uint mask)
{
    uint bitMask = WaveActiveBallot(bit).x;
    
    uint laneIdx = WaveGetLaneIndex();
    uint laneMask = (1 << laneIdx) - 1;

    uint multipleMask = laneMask & mask;
    uint prefix = countbits(multipleMask & bitMask); 

    return prefix;
}

///////////////////////////////
/********Histogram********/
//////////////////////////////

#define RADIX 8 // 8bits per pass
#define SORT_COUNT 4 //(32 / RADIX)
#define BUCKETS 256 // (1 << RADIX)

groupshared uint sHistogramWaves[BUCKETS * MAX_WAVES];
groupshared uint sHistogramMultiPlace[BUCKETS * SORT_COUNT];
groupshared uint sBlockOffsets[BUCKETS]; // inter-block exclusive prefix sum, one per bucket

//optimized buffer, read current digit place's global histogram to L1 cache
groupshared uint sGlobalBinOffsets[BUCKETS];

uint ExtractDigitOpt_Radix8(uint value, uint digitPlace)
{
#if defined (DIGIT_PLACE_0)
    return (value ) & (BUCKETS - 1);
#elif defined (DIGIT_PLACE_1)
    return (value >> 8) & (BUCKETS - 1);
#elif defined (DIGIT_PLACE_2)
    return (value >> 16) & (BUCKETS - 1);
#else
    return (value >> 24) & (BUCKETS - 1);
#endif
}

uint4 ExtractDigit4Opt_Radix8(uint value)
{
    uint4 extract4;
    extract4.x = (value) & (BUCKETS - 1);
    extract4.y = (value >> 8) & (BUCKETS - 1);
    extract4.z = (value >> 16) & (BUCKETS - 1);
    extract4.w = (value >> 24) & (BUCKETS - 1);
    return extract4;
}

uint ExtractDigit(uint value, uint digitPlace)
{
    return (value >> (digitPlace * RADIX)) & (BUCKETS - 1);
}

void ClearHistogram(uint threadIdx)
{
    for (uint i = 0; i < MAX_WAVES; i++)
        sHistogramWaves[threadIdx + i * BUCKETS] = 0;
}

uint BuildHistogram(uint digit, uint waveIdx)
{
    uint sameMask = WaveMatch32_8bits(digit);
    uint sameCount = countbits(sameMask);
    uint leaderLane = firstbitlow(sameMask);

    uint baseOffsetInTile = 0;
    if (WaveGetLaneIndex() == leaderLane)
    {
        baseOffsetInTile = sHistogramWaves[digit + waveIdx * BUCKETS];
        sHistogramWaves[digit + waveIdx * BUCKETS] += sameCount;
    }

    baseOffsetInTile = WaveReadLaneAt(baseOffsetInTile, leaderLane); // broadcast leader's base offset to all lanes in the same digit
    uint idxInSame = WaveMultiPrefixCountBits32(true, sameMask);
    return baseOffsetInTile + idxInSame;
}

uint ScanHistogram(uint threadIdx)
{
    uint prefix = 0;
    for (uint wave = 0; wave < MAX_WAVES; wave++)
    {
        uint bucketSum = sHistogramWaves[threadIdx + wave * BUCKETS];
        sHistogramWaves[threadIdx + wave * BUCKETS] = prefix;
        prefix += bucketSum;
    }
    return prefix;
}

void BuildHistogramMultiPlace(uint value)
{
    uint4 digitVec = ExtractDigit4Opt_Radix8(value);
    uint digts[4] = { digitVec.x, digitVec.y, digitVec.z, digitVec.w };

    for(uint digitPlace = 0; digitPlace < SORT_COUNT; digitPlace++)
    {
        uint digit = digts[digitPlace];
        uint sameMask = WaveMatch32_8bits(digit);
        uint sameCount = countbits(sameMask);
        uint leaderLane = firstbitlow(sameMask);
        if (WaveGetLaneIndex() == leaderLane)
            InterlockedAdd(sHistogramMultiPlace[digitPlace * BUCKETS + digit], sameCount);
    }
}


 
