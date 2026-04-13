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
// GaussianDrawData  --  owns per-render-node DX11 resources.
//
// Input StructuredBuffers now live in GaussianDataNode (shared).
// This class holds non-owning SRV pointers copied each frame, plus
// per-instance compute outputs, sort buffers, and pipeline state.
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
    float        pointSize    = 4.f;
    float        vpWidth      = 1280.f;
    float        vpHeight     = 720.f;
    unsigned int vertexCount  = 0;
    int          renderMode   = 0;   // 0=auto, 1=debug, 2=production

    // -----------------------------------------------------------------------
    // Debug pipeline  (VS+GS+PS, reads from shared StructuredBuffers)
    // -----------------------------------------------------------------------
    ID3D11VertexShader*    dbgVS         = nullptr;
    ID3D11GeometryShader*  dbgGS         = nullptr;
    ID3D11PixelShader*     dbgPS         = nullptr;
    ID3D11Buffer*          dbgCB         = nullptr;

    // Shared render states (used by both debug and production)
    ID3D11BlendState*      blendState    = nullptr;
    ID3D11RasterizerState* rsState       = nullptr;
    ID3D11DepthStencilState* dsState     = nullptr;

    bool dbgReady = false;

    // -----------------------------------------------------------------------
    // Non-owning references to shared input SRVs (from GaussianDataNode).
    // Refreshed every frame in prepareForDraw. Do NOT Release() these.
    // -----------------------------------------------------------------------
    ID3D11ShaderResourceView* sharedSrvPositionWS = nullptr;
    ID3D11ShaderResourceView* sharedSrvScale      = nullptr;
    ID3D11ShaderResourceView* sharedSrvRotation   = nullptr;
    ID3D11ShaderResourceView* sharedSrvOpacity    = nullptr;
    ID3D11ShaderResourceView* sharedSrvSHCoeffs   = nullptr;
    bool inputsReady = false;

    // -----------------------------------------------------------------------
    // Production pipeline  (Compute preprocess + ellipse render)
    // -----------------------------------------------------------------------

    // -- Compute shader --
    ID3D11ComputeShader*   computeShader = nullptr;
    ID3D11Buffer*          computeCB     = nullptr;

    // -- Output UAVs (written by compute each frame) --
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

    // -- Render shader --
    ID3D11VertexShader*    prodVS    = nullptr;
    ID3D11PixelShader*     prodPS    = nullptr;
    ID3D11Buffer*          prodCB    = nullptr;

    bool     prodReady  = false;
    uint32_t allocatedN = 0;

    // -----------------------------------------------------------------------
    // GPU Radix Sort pipeline  (CS 5.0, no wave intrinsics)
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
    // Depth Pass pipeline  (method B: compute writes per-pixel rep. depth
    // into a UAV texture; copy pass transfers into Maya's depth buffer so
    // GS objects can occlude Maya transparent/virtual geometry drawn later)
    // -----------------------------------------------------------------------
    ID3D11ComputeShader*       depthClearCS  = nullptr;
    ID3D11ComputeShader*       depthPassCS   = nullptr;
    ID3D11Buffer*              depthCB       = nullptr;

    ID3D11Texture2D*           depthTex      = nullptr;
    ID3D11UnorderedAccessView* depthTex_UAV  = nullptr;
    ID3D11ShaderResourceView*  depthTex_SRV  = nullptr;
    uint32_t                   depthTexW     = 0;
    uint32_t                   depthTexH     = 0;

    // Copy pass (full-screen triangle writing SV_Depth)
    ID3D11VertexShader*        depthCopyVS   = nullptr;
    ID3D11PixelShader*         depthCopyPS   = nullptr;
    ID3D11DepthStencilState*   depthWriteDS  = nullptr;
    ID3D11BlendState*          depthCopyBlend = nullptr;    // color writes masked out

    bool depthPassReady = false;

    // -----------------------------------------------------------------------
    // Init / upload helpers
    // -----------------------------------------------------------------------
    bool initDebugPipeline(ID3D11Device* device);
    bool initProductionPipeline(ID3D11Device* device);
    bool initSortPipeline(ID3D11Device* device);
    bool initDepthPassPipeline(ID3D11Device* device);
    bool createSortBuffers(ID3D11Device* device, uint32_t N);
    bool createDepthTexture(ID3D11Device* device, uint32_t w, uint32_t h);

    bool createComputeOutputs(ID3D11Device* device, uint32_t N);

    void releaseAll();

private:
    void releaseDebugResources();
    void releaseProductionResources();
    void releaseSortResources();
    void releaseDepthPassResources();

    bool createUAVBuffer(ID3D11Device* device,
                         const char*   name,
                         uint32_t      numElements,
                         uint32_t      stride,
                         ID3D11Buffer**              outBuf,
                         ID3D11UnorderedAccessView** outUAV,
                         ID3D11ShaderResourceView**  outSRV);

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

    // Drawn in the transparent pass so opaque Maya geometry depth is already
    // present in the depth buffer, enabling correct mutual occlusion.
    bool isTransparent() const override { return true; }

    bool         isBounded(const MDagPath& objPath, const MDagPath&) const override;
    MBoundingBox boundingBox(const MDagPath& objPath, const MDagPath&) const override;

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
