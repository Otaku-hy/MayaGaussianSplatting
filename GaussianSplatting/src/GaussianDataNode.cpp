#include "GaussianDataNode.h"
#include "PLYReader.h"

#include <maya/MFnTypedAttribute.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnMessageAttribute.h>
#include <maya/MFnData.h>
#include <maya/MGlobal.h>

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
    m_inputsReady = false;
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

    m_inputsReady = true;
    m_inputsDirty = false;
    MGlobal::displayInfo("[GaussianSplatData] GPU buffers uploaded successfully.");
    return true;
}

#undef SAFE_RELEASE
