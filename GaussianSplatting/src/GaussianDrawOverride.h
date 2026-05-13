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
// GaussianDrawData  --  per-render-node DX11 resources.
//
// Production/sort/depth resources now live in GaussianRenderManager (singleton).
// This class keeps only debug pipeline resources and per-frame CPU data.
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
    int          renderMode   = 0;   // 0=auto, 1=debug, 2=production, 3=diagnostic

    // -----------------------------------------------------------------------
    // Debug pipeline  (VS+GS+PS, reads from shared StructuredBuffers)
    // -----------------------------------------------------------------------
    ID3D11VertexShader*    dbgVS         = nullptr;
    ID3D11GeometryShader*  dbgGS         = nullptr;
    ID3D11PixelShader*     dbgPS         = nullptr;
    ID3D11Buffer*          dbgCB         = nullptr;

    // Shared render states (used by debug path)
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

    // Whether this instance was registered with the RenderManager this frame
    bool registeredWithManager = false;

    // -----------------------------------------------------------------------
    // Init / release
    // -----------------------------------------------------------------------
    bool initDebugPipeline(ID3D11Device* device);
    void releaseAll();

private:
    void releaseDebugResources();

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
