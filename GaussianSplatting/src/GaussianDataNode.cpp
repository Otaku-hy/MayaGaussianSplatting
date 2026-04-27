#include "GaussianDataNode.h"
#include "PLYReader.h"

#include <maya/MFnTypedAttribute.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnMessageAttribute.h>
#include <maya/MFnData.h>
#include <maya/MGlobal.h>

#include <algorithm>
#include <cstring>

// ---------------------------------------------------------------------------
// Static member definitions
// ---------------------------------------------------------------------------
MTypeId GaussianDataNode::typeId    { 0x00127A01 };
MString GaussianDataNode::typeName  { "gaussianSplatData" };

MObject GaussianDataNode::aFilePath;
MObject GaussianDataNode::aDataReady;
MObject GaussianDataNode::aOutputData;

// ---------------------------------------------------------------------------
void* GaussianDataNode::creator() { return new GaussianDataNode(); }

GaussianDataNode::~GaussianDataNode() { releaseInputBuffers(); }

MStatus GaussianDataNode::initialize() {
    MFnTypedAttribute   tAttr;
    MFnNumericAttribute nAttr;
    MFnMessageAttribute mAttr;

    // filePath
    aFilePath = tAttr.create("filePath", "fp", MFnData::kString);
    tAttr.setUsedAsFilename(true);
    tAttr.setStorable(true);
    CHECK_MSTATUS_AND_RETURN_IT(addAttribute(aFilePath));

    // dataReady (dummy output, triggers compute)
    aDataReady = nAttr.create("dataReady", "dr", MFnNumericData::kBoolean, false);
    nAttr.setWritable(false);
    nAttr.setStorable(false);
    nAttr.setHidden(true);
    CHECK_MSTATUS_AND_RETURN_IT(addAttribute(aDataReady));

    // outputData (message attribute for render node connections)
    aOutputData = mAttr.create("outputData", "od");
    mAttr.setStorable(false);
    CHECK_MSTATUS_AND_RETURN_IT(addAttribute(aOutputData));

    attributeAffects(aFilePath, aDataReady);

    return MS::kSuccess;
}

// ---------------------------------------------------------------------------
// compute  --  load PLY when filePath changes
// ---------------------------------------------------------------------------
MStatus GaussianDataNode::compute(const MPlug& plug, MDataBlock& dataBlock) {
    if (plug != aDataReady)
        return MS::kUnknownParameter;

    MString newPath = dataBlock.inputValue(aFilePath).asString();

    if (newPath != m_loadedPath) {
        m_data.clear();
        releaseInputBuffers();
        m_loadedPath = newPath;

        if (newPath.length() > 0) {
            std::string err;
            if (PLYReader::read(newPath.asChar(), m_data, err)) {
                MGlobal::displayInfo(MString("[GaussianSplatData] Loaded ") +
                                     (unsigned int)m_data.count() + " splats.");

                // --- Scale diagnostics (visible in Script Editor) ---
                if (!m_data.scaleWS.empty()) {
                    float sMin = 1e30f, sMax = -1e30f, sSum = 0.f;
                    int   allOnes = 0;
                    for (float v : m_data.scaleWS) {
                        if (v < sMin) sMin = v;
                        if (v > sMax) sMax = v;
                        sSum += v;
                        if (v > 0.999f && v < 1.001f) allOnes++;
                    }
                    float avg = sSum / (float)m_data.scaleWS.size();
                    bool  missingScale = (allOnes == (int)m_data.scaleWS.size());

                    MString ms("[GaussianSplatData] Scale (after exp): min=");
                    ms += sMin; ms += " max="; ms += sMax;
                    ms += " avg="; ms += avg;
                    if (missingScale)
                        ms += "  <<< ALL 1.0 — PLY likely missing scale_0/1/2, splats will be 1m radius! >>>";
                    MGlobal::displayInfo(ms);
                }

                m_inputsDirty = true;
            } else {
                MGlobal::displayError(MString("[GaussianSplatData] ") + err.c_str());
            }
        }
    }

    dataBlock.outputValue(aDataReady).setBool(!m_data.empty());
    dataBlock.setClean(plug);
    return MS::kSuccess;
}

// ---------------------------------------------------------------------------
// GPU input buffer helpers
// ---------------------------------------------------------------------------
#define SAFE_RELEASE(p) do { if (p) { (p)->Release(); (p) = nullptr; } } while(0)

void GaussianDataNode::releaseInputBuffers()
{
    SAFE_RELEASE(m_sbPositionWS); SAFE_RELEASE(m_srvPositionWS);
    SAFE_RELEASE(m_sbScale);      SAFE_RELEASE(m_srvScale);
    SAFE_RELEASE(m_sbRotation);   SAFE_RELEASE(m_srvRotation);
    SAFE_RELEASE(m_sbOpacity);    SAFE_RELEASE(m_srvOpacity);
    SAFE_RELEASE(m_sbSHCoeffs);   SAFE_RELEASE(m_srvSHCoeffs);
    releaseSelectionMaskBuffer();
    m_inputsReady = false;
}

void GaussianDataNode::releaseSelectionMaskBuffer()
{
    SAFE_RELEASE(m_uavSelectionMask);
    SAFE_RELEASE(m_srvSelectionMask);
    SAFE_RELEASE(m_sbSelectionMask);
    m_maskShadow.clear();
}

bool GaussianDataNode::createSelectionMaskBuffer(ID3D11Device* device, uint32_t N)
{
    releaseSelectionMaskBuffer();
    m_maskShadow.assign(N, 0u);

    D3D11_BUFFER_DESC bd = {};
    bd.ByteWidth           = N * sizeof(uint32_t);
    bd.Usage               = D3D11_USAGE_DEFAULT;
    bd.BindFlags           = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    bd.MiscFlags           = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    bd.StructureByteStride = sizeof(uint32_t);

    D3D11_SUBRESOURCE_DATA init = { m_maskShadow.data(), 0, 0 };
    if (FAILED(device->CreateBuffer(&bd, &init, &m_sbSelectionMask))) {
        MGlobal::displayError("[GaussianSplatData] CreateBuffer failed for selectionMask.");
        return false;
    }

    D3D11_SHADER_RESOURCE_VIEW_DESC srvd = {};
    srvd.ViewDimension         = D3D11_SRV_DIMENSION_BUFFEREX;
    srvd.BufferEx.FirstElement = 0;
    srvd.BufferEx.NumElements  = N;
    if (FAILED(device->CreateShaderResourceView(m_sbSelectionMask, &srvd, &m_srvSelectionMask))) {
        releaseSelectionMaskBuffer();
        return false;
    }

    D3D11_UNORDERED_ACCESS_VIEW_DESC uavd = {};
    uavd.ViewDimension      = D3D11_UAV_DIMENSION_BUFFER;
    uavd.Buffer.FirstElement = 0;
    uavd.Buffer.NumElements  = N;
    if (FAILED(device->CreateUnorderedAccessView(m_sbSelectionMask, &uavd, &m_uavSelectionMask))) {
        releaseSelectionMaskBuffer();
        return false;
    }

    m_maskVersion++;
    return true;
}

// ---------------------------------------------------------------------------
// Selection mask helpers
// ---------------------------------------------------------------------------
void GaussianDataNode::restoreAll(ID3D11DeviceContext* ctx)
{
    if (!m_sbSelectionMask || m_maskShadow.empty()) return;
    std::fill(m_maskShadow.begin(), m_maskShadow.end(), 0u);
    ctx->UpdateSubresource(m_sbSelectionMask, 0, nullptr, m_maskShadow.data(), 0, 0);
    m_maskVersion++;
}

void GaussianDataNode::clearSelection(ID3D11DeviceContext* ctx)
{
    if (!m_sbSelectionMask || m_maskShadow.empty()) return;
    for (auto& v : m_maskShadow) v &= ~kMaskBitSelected;
    ctx->UpdateSubresource(m_sbSelectionMask, 0, nullptr, m_maskShadow.data(), 0, 0);
    m_maskVersion++;
}

void GaussianDataNode::deleteSelected(ID3D11DeviceContext* ctx)
{
    if (!m_sbSelectionMask || m_maskShadow.empty()) return;
    // NOTE: operates on CPU shadow. If a selection CS has written the GPU
    // buffer since last readback, the caller should readbackMask() first.
    uint32_t numDeleted = 0;
    for (auto& v : m_maskShadow) {
        if (v & kMaskBitSelected) {
            v = (v & ~kMaskBitSelected) | kMaskBitDeleted;
            numDeleted++;
        }
    }
    ctx->UpdateSubresource(m_sbSelectionMask, 0, nullptr, m_maskShadow.data(), 0, 0);
    m_maskVersion++;
    MGlobal::displayInfo(MString("[GaussianSplatData] Soft-deleted ") +
                         (unsigned)numDeleted + " splats.");
}

bool GaussianDataNode::readbackMask(ID3D11Device* device, ID3D11DeviceContext* ctx)
{
    if (!m_sbSelectionMask || m_maskShadow.empty()) return false;

    ID3D11Buffer* staging = nullptr;
    D3D11_BUFFER_DESC bd = {};
    bd.ByteWidth      = (UINT)(m_maskShadow.size() * sizeof(uint32_t));
    bd.Usage          = D3D11_USAGE_STAGING;
    bd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    if (FAILED(device->CreateBuffer(&bd, nullptr, &staging))) return false;

    ctx->CopyResource(staging, m_sbSelectionMask);

    D3D11_MAPPED_SUBRESOURCE mapped;
    if (FAILED(ctx->Map(staging, 0, D3D11_MAP_READ, 0, &mapped))) {
        staging->Release();
        return false;
    }
    std::memcpy(m_maskShadow.data(), mapped.pData, bd.ByteWidth);
    ctx->Unmap(staging, 0);
    staging->Release();
    return true;
}

bool GaussianDataNode::createSRVBuffer(ID3D11Device* device,
                                        const char*   name,
                                        const void*   initData,
                                        uint32_t      numElements,
                                        uint32_t      stride,
                                        ID3D11Buffer**             outBuf,
                                        ID3D11ShaderResourceView** outSRV)
{
    uint32_t totalBytes = numElements * stride;

    D3D11_BUFFER_DESC bd = {};
    bd.ByteWidth           = totalBytes;
    bd.Usage               = D3D11_USAGE_DEFAULT;
    bd.BindFlags           = D3D11_BIND_SHADER_RESOURCE;
    bd.MiscFlags           = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    bd.StructureByteStride = stride;

    D3D11_SUBRESOURCE_DATA init = { initData, 0, 0 };
    HRESULT hr = device->CreateBuffer(&bd, initData ? &init : nullptr, outBuf);
    if (FAILED(hr)) {
        MGlobal::displayError(
            MString("[GaussianSplatData] CreateBuffer failed for '") + name +
            "' (" + (unsigned int)(totalBytes / 1024 / 1024) + " MB).");
        return false;
    }

    D3D11_SHADER_RESOURCE_VIEW_DESC srvd = {};
    srvd.ViewDimension         = D3D11_SRV_DIMENSION_BUFFEREX;
    srvd.BufferEx.FirstElement = 0;
    srvd.BufferEx.NumElements  = numElements;
    hr = device->CreateShaderResourceView(*outBuf, &srvd, outSRV);
    if (FAILED(hr)) {
        MGlobal::displayError(
            MString("[GaussianSplatData] CreateSRV failed for '") + name + "'.");
        SAFE_RELEASE(*outBuf);
        return false;
    }
    return true;
}

bool GaussianDataNode::uploadInputBuffersIfNeeded(ID3D11Device* device)
{
    if (!m_inputsDirty || m_data.empty()) return m_inputsReady;

    releaseInputBuffers();

    uint32_t N = (uint32_t)m_data.count();
    MGlobal::displayInfo(MString("[GaussianSplatData] Uploading GPU buffers for ") + N + " splats...");

    if (!createSRVBuffer(device, "positionWS", m_data.positions.data(),  N, sizeof(float)*3, &m_sbPositionWS, &m_srvPositionWS)) return false;
    if (!createSRVBuffer(device, "scale",      m_data.scaleWS.data(),    N, sizeof(float)*3, &m_sbScale,      &m_srvScale))      return false;
    if (!createSRVBuffer(device, "rotation",   m_data.rotationWS.data(), N, sizeof(float)*4, &m_sbRotation,   &m_srvRotation))   return false;
    if (!createSRVBuffer(device, "opacity",    m_data.opacityRaw.data(), N, sizeof(float),   &m_sbOpacity,    &m_srvOpacity))    return false;
    if (!createSRVBuffer(device, "shCoeffs",   m_data.shCoeffs.data(), N * kSHCoeffsPerSplat, sizeof(float)*3, &m_sbSHCoeffs, &m_srvSHCoeffs)) return false;

    // Selection mask: zero-initialized, one uint per splat
    if (!createSelectionMaskBuffer(device, N)) return false;

    m_inputsReady = true;
    m_inputsDirty = false;
    MGlobal::displayInfo("[GaussianSplatData] GPU buffers uploaded successfully.");
    return true;
}

#undef SAFE_RELEASE

// ---------------------------------------------------------------------------
// boundingBox  --  object-space AABB of the loaded splats
// ---------------------------------------------------------------------------
MBoundingBox GaussianDataNode::boundingBox() const
{
    if (m_data.empty())
        return MBoundingBox(MPoint(-1, -1, -1), MPoint(1, 1, 1));
    return MBoundingBox(
        MPoint(m_data.bboxMin[0], m_data.bboxMin[1], m_data.bboxMin[2]),
        MPoint(m_data.bboxMax[0], m_data.bboxMax[1], m_data.bboxMax[2]));
}
