#pragma once
#include <maya/MPxDrawOverride.h>
#include <maya/MUserData.h>
#include <maya/MBoundingBox.h>
#include <maya/MString.h>
#include <maya/MObject.h>

#include <d3d11.h>
#include <cstdint>

#include "GaussianData.h"

class GaussianNode;

// ===========================================================================
// GaussianDrawData  --  owns all DX11 resources for one render node.
// ===========================================================================
class GaussianDrawData : public MUserData {
public:
    GaussianDrawData();
    ~GaussianDrawData() override;

    // -----------------------------------------------------------------------
    // Per-frame CPU data written by prepareForDraw
    // -----------------------------------------------------------------------
    float        wvp[16]      = {};
    float        worldMat[16] = {};
    float        viewMat[16]  = {};
    float        projMat[16]  = {};
    float        cameraPos[3] = {};
    float        tanHalfFov[2]= {};
    float        pointSize    = 10.f;
    float        vpWidth      = 1280.f;
    float        vpHeight     = 720.f;
    unsigned int vertexCount  = 0;

    // -----------------------------------------------------------------------
    // Debug pipeline  (VS+GS+PS, reads raw input SRVs)
    // -----------------------------------------------------------------------
    ID3D11VertexShader*    dbgVS         = nullptr;
    ID3D11GeometryShader*  dbgGS         = nullptr;
    ID3D11PixelShader*     dbgPS         = nullptr;
    ID3D11Buffer*          dbgCB         = nullptr;

    // Shared render states
    ID3D11BlendState*      blendState    = nullptr;
    ID3D11RasterizerState* rsState       = nullptr;
    ID3D11DepthStencilState* dsState     = nullptr;

    bool dbgReady = false;

    // -----------------------------------------------------------------------
    // Input StructuredBuffers (owned, uploaded from GaussianNode::m_data)
    // -----------------------------------------------------------------------
    ID3D11Buffer*             sbPositionWS  = nullptr;
    ID3D11ShaderResourceView* srvPositionWS = nullptr;
    ID3D11Buffer*             sbScale       = nullptr;
    ID3D11ShaderResourceView* srvScale      = nullptr;
    ID3D11Buffer*             sbRotation    = nullptr;
    ID3D11ShaderResourceView* srvRotation   = nullptr;
    ID3D11Buffer*             sbOpacity     = nullptr;
    ID3D11ShaderResourceView* srvOpacity    = nullptr;
    ID3D11Buffer*             sbSHCoeffs    = nullptr;
    ID3D11ShaderResourceView* srvSHCoeffs   = nullptr;

    bool     inputsReady = false;
    MString  loadedPath;            // path that was uploaded

    // -----------------------------------------------------------------------
    // Production pipeline  (Compute preprocess + sort + render)
    // -----------------------------------------------------------------------

    // -- Compute shader --
    ID3D11ComputeShader*   computeShader = nullptr;
    ID3D11Buffer*          computeCB     = nullptr;

    // -- Compute output UAVs --
    ID3D11Buffer*              ubPositionSS = nullptr;
    ID3D11UnorderedAccessView* uavPositionSS= nullptr;
    ID3D11ShaderResourceView*  srvPositionSS= nullptr;

    ID3D11Buffer*              ubDepth      = nullptr;
    ID3D11UnorderedAccessView* uavDepth     = nullptr;
    ID3D11ShaderResourceView*  srvDepth     = nullptr;

    ID3D11Buffer*              ubRadius     = nullptr;
    ID3D11UnorderedAccessView* uavRadius    = nullptr;
    ID3D11ShaderResourceView*  srvRadius    = nullptr;

    ID3D11Buffer*              ubColor      = nullptr;
    ID3D11UnorderedAccessView* uavColor     = nullptr;
    ID3D11ShaderResourceView*  srvColor     = nullptr;

    ID3D11Buffer*              ubCov2D      = nullptr;
    ID3D11UnorderedAccessView* uavCov2D     = nullptr;
    ID3D11ShaderResourceView*  srvCov2D     = nullptr;

    // -- Render shader (VS + GS + PS, one vertex per splat) --
    ID3D11VertexShader*    prodVS    = nullptr;
    ID3D11GeometryShader*  prodGS    = nullptr;
    ID3D11PixelShader*     prodPS    = nullptr;
    ID3D11Buffer*          prodCB    = nullptr;

    bool     prodReady  = false;
    uint32_t allocatedN = 0;

    // -----------------------------------------------------------------------
    // GPU Radix Sort pipeline  (CS 5.0)
    // -----------------------------------------------------------------------
    ID3D11ComputeShader*       sortCS_keygen    = nullptr;
    ID3D11ComputeShader*       sortCS_count     = nullptr;
    ID3D11ComputeShader*       sortCS_scan      = nullptr;
    ID3D11ComputeShader*       sortCS_scatter   = nullptr;
    ID3D11Buffer*              sortCB           = nullptr;

    ID3D11Buffer*              sortKeysA        = nullptr;
    ID3D11UnorderedAccessView* sortKeysA_UAV    = nullptr;
    ID3D11ShaderResourceView*  sortKeysA_SRV    = nullptr;
    ID3D11Buffer*              sortKeysB        = nullptr;
    ID3D11UnorderedAccessView* sortKeysB_UAV    = nullptr;
    ID3D11ShaderResourceView*  sortKeysB_SRV    = nullptr;

    ID3D11Buffer*              sortValsA        = nullptr;
    ID3D11UnorderedAccessView* sortValsA_UAV    = nullptr;
    ID3D11ShaderResourceView*  sortValsA_SRV    = nullptr;
    ID3D11Buffer*              sortValsB        = nullptr;
    ID3D11UnorderedAccessView* sortValsB_UAV    = nullptr;
    ID3D11ShaderResourceView*  sortValsB_SRV    = nullptr;

    ID3D11Buffer*              sortBlockHist    = nullptr;
    ID3D11UnorderedAccessView* sortBlockHist_UAV= nullptr;
    ID3D11ShaderResourceView*  sortBlockHist_SRV= nullptr;

    bool sortReady = false;

    // -----------------------------------------------------------------------
    // Init helpers
    // -----------------------------------------------------------------------
    bool initDebugPipeline(ID3D11Device* device);
    bool initProductionPipeline(ID3D11Device* device);
    bool initSortPipeline(ID3D11Device* device);
    bool createSortBuffers(ID3D11Device* device, uint32_t N);
    bool createComputeOutputs(ID3D11Device* device, uint32_t N);

    bool uploadInputBuffers(ID3D11Device* device, const GaussianData& data);

    void releaseAll();

private:
    void releaseDebugResources();
    void releaseProductionResources();
    void releaseSortResources();
    void releaseInputBuffers();

    bool createUAVBuffer(ID3D11Device* device,
                         const char*   name,
                         uint32_t      numElements,
                         uint32_t      stride,
                         ID3D11Buffer**              outBuf,
                         ID3D11UnorderedAccessView** outUAV,
                         ID3D11ShaderResourceView**  outSRV);

    static bool createSRVBuffer(ID3D11Device* device,
                                const char*   name,
                                const void*   initData,
                                uint32_t      numElements,
                                uint32_t      stride,
                                ID3D11Buffer**             outBuf,
                                ID3D11ShaderResourceView** outSRV);

    GaussianDrawData(const GaussianDrawData&)            = delete;
    GaussianDrawData& operator=(const GaussianDrawData&) = delete;
};

// ===========================================================================
// GaussianDrawOverride
// ===========================================================================
class GaussianDrawOverride : public MHWRender::MPxDrawOverride {
public:
    static MHWRender::MPxDrawOverride* creator(const MObject& obj);

    explicit GaussianDrawOverride(const MObject& obj);
    ~GaussianDrawOverride() override = default;

    MHWRender::DrawAPI supportedDrawAPIs() const override;

    bool         isBounded(const MDagPath&, const MDagPath&) const override { return false; }
    MBoundingBox boundingBox(const MDagPath&, const MDagPath&) const override { return {}; }

    MUserData* prepareForDraw(const MDagPath&                   objPath,
                              const MDagPath&                   cameraPath,
                              const MHWRender::MFrameContext&   frameContext,
                              MUserData*                        oldData) override;

    bool hasUIDrawables() const override { return false; }

    static void draw(const MHWRender::MDrawContext& context,
                     const MUserData*               data);

private:
    GaussianNode* m_node = nullptr;
};
