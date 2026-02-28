#pragma once
#include <maya/MPxDrawOverride.h>
#include <maya/MUserData.h>
#include <maya/MBoundingBox.h>
#include <maya/MString.h>
#include <maya/MObject.h>

#include <d3d11.h>

class GaussianNode;

// ---------------------------------------------------------------------------
// GaussianDrawData  —  MUserData subclass that owns all DX11 resources.
//
// Lifetime managed by MSharedPtr<MUserData> (Maya 2026+).
// DX11 objects are created lazily in prepareForDraw() and released in dtor.
// ---------------------------------------------------------------------------
class GaussianDrawData : public MUserData {
public:
    GaussianDrawData();
    ~GaussianDrawData() override;

    // ---- per-frame CPU data (written by prepareForDraw each frame) ----
    float        wvp[16]     = {};      // row-major world-view-projection
    float        pointSize   = 4.f;
    float        vpWidth     = 1280.f;
    float        vpHeight    = 720.f;
    unsigned int vertexCount = 0;

    // ---- persistent DX11 resources ----
    ID3D11VertexShader*    vs          = nullptr;
    ID3D11GeometryShader*  gs          = nullptr;
    ID3D11PixelShader*     ps          = nullptr;
    ID3D11InputLayout*     inputLayout = nullptr;
    ID3D11Buffer*          posBuf      = nullptr;   // float3 per vertex
    ID3D11Buffer*          colBuf      = nullptr;   // float4 per vertex
    ID3D11Buffer*          constBuf    = nullptr;   // CBPerObject
    ID3D11BlendState*      blendState  = nullptr;
    ID3D11RasterizerState* rsState     = nullptr;

    bool shadersReady  = false;
    bool vertexDirty   = true;

    // Initialise VS / GS / PS / input-layout / states from inline HLSL
    bool initShaders(ID3D11Device* device);

    // (Re)upload position and colour arrays to GPU vertex buffers
    bool uploadVertices(ID3D11Device*  device,
                        const float*   positions,
                        const float*   colors,
                        unsigned int   count);

    void releaseAll();

private:
    GaussianDrawData(const GaussianDrawData&)            = delete;
    GaussianDrawData& operator=(const GaussianDrawData&) = delete;
};

// ---------------------------------------------------------------------------
// GaussianDrawOverride  —  MPxDrawOverride for GaussianNode.
//
// prepareForDraw()  runs on the main thread: reads the DG, computes WVP,
//                   (re)builds GPU buffers when PLY data changes.
// draw()            runs on the render thread: issues raw DX11 draw calls.
// ---------------------------------------------------------------------------
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
    GaussianNode* m_node       = nullptr;
    MString       m_loadedPath;           // last PLY path uploaded to GPU
};
