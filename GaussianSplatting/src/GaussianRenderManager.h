#pragma once
#include <d3d11.h>
#include <cstdint>
#include <vector>

class GaussianDataNode;

// ===========================================================================
// GaussianRenderManager  --  Singleton that merges all GaussianSplat instances
// into one unified preprocess -> sort -> render -> depth-pass pipeline.
//
// Flow per frame:
//   1. beginFrame()          -- called once at start of frame (clears instances)
//   2. registerInstance(...)  -- called from each DrawOverride::prepareForDraw
//   3. render(ctx, ...)      -- called from the FIRST DrawOverride::draw;
//                               subsequent draw() calls are no-ops
// ===========================================================================

struct RenderInstance {
    GaussianDataNode* dataNode;        // source data
    float             worldMat[16];    // per-instance world transform
    uint32_t          splatCount;      // dataNode->splatCount()
};

class GaussianRenderManager {
public:
    static GaussianRenderManager& instance();

    // --- Per-frame API (called from GaussianDrawOverride) ---
    void beginFrame(uint64_t frameStamp);
    void registerInstance(const RenderInstance& inst);

    // Store camera/viewport info (called once per frame from first prepareForDraw)
    void setFrameData(const float viewMat[16], const float projMat[16],
                      const float cameraPos[3], const float tanHalfFov[2],
                      float vpWidth, float vpHeight);

    bool canRender() const;
    uint32_t totalSplatCount() const { return m_totalSplats; }

    // Execute the merged pipeline. Returns true if rendering happened.
    // renderMode: 0=auto, 2=production, 3=diagnostic (fixed radius)
    bool render(ID3D11Device* device, ID3D11DeviceContext* ctx, int renderMode);

    // --- Marquee selection (invoked from Maya commands) -------------------
    // Project all splats of `node` through worldMat * viewProj into NDC,
    // test against [rectMinNDC, rectMaxNDC] and update node's selection mask
    // bit 0 according to `mode`:
    //   0 = replace, 1 = add, 2 = subtract, 3 = toggle
    // Deleted splats (bit 1) are never newly selected.
    //
    // Returns true on success. Automatically marks the Manager's merged
    // selection as dirty so the next draw picks up the change.
    bool runSelection(ID3D11Device* device, ID3D11DeviceContext* ctx,
                      GaussianDataNode* node,
                      const float worldMat[16],
                      const float viewProj[16],
                      float rectMinX, float rectMinY,
                      float rectMaxX, float rectMaxY,
                      int mode);

    // Explicitly mark the merged selection buffer as stale. Called by
    // commands that modify a data node's mask outside runSelection().
    void markSelectionDirty() { m_selectionDirty = true; }

    // Has the merged render already happened this frame?
    bool renderedThisFrame() const { return m_frameRendered; }
    uint64_t currentFrame() const { return m_frameStamp; }

    // Expose latest view/proj (row-major) captured by setFrameData(). Used by
    // selection commands to build viewProj for the marquee CS.
    const float* viewMatrix()     const { return m_viewMat; }
    const float* projMatrix()     const { return m_projMat; }
    float        viewportWidth()  const { return m_vpWidth; }
    float        viewportHeight() const { return m_vpHeight; }

    // Cleanup (call from uninitializePlugin)
    void releaseAll();

    // Access to merged compute outputs for depth pass (used by individual draw overrides)
    // These are valid only after render() returns true.
    ID3D11ShaderResourceView* srvPositionSS() const { return m_srvPositionSS; }
    ID3D11ShaderResourceView* srvRadius()     const { return m_srvRadius; }
    ID3D11ShaderResourceView* srvDepth()      const { return m_srvDepth; }
    ID3D11ShaderResourceView* srvColor()      const { return m_srvColor; }
    ID3D11ShaderResourceView* srvCov2D()      const { return m_srvCov2D; }
    ID3D11ShaderResourceView* sortedIndicesSRV() const { return m_sortValsA_SRV; }
    ID3D11ShaderResourceView* srvMergedSelection() const { return m_srvMergedSelection; }

    // Depth pass resources (shared across all instances)
    ID3D11ComputeShader*       depthClearCS()  const { return m_depthClearCS; }
    ID3D11ComputeShader*       depthPassCS()   const { return m_depthPassCS; }
    ID3D11Buffer*              depthCB()       const { return m_depthCB; }
    ID3D11Texture2D*           depthTex()      const { return m_depthTex; }
    ID3D11UnorderedAccessView* depthTex_UAV()  const { return m_depthTex_UAV; }
    ID3D11ShaderResourceView*  depthTex_SRV()  const { return m_depthTex_SRV; }
    uint32_t                   depthTexW()     const { return m_depthTexW; }
    uint32_t                   depthTexH()     const { return m_depthTexH; }
    ID3D11VertexShader*        depthCopyVS()   const { return m_depthCopyVS; }
    ID3D11PixelShader*         depthCopyPS()   const { return m_depthCopyPS; }
    ID3D11DepthStencilState*   depthWriteDS()  const { return m_depthWriteDS; }
    ID3D11BlendState*          depthCopyBlend() const { return m_depthCopyBlend; }
    bool                       depthPassReady() const { return m_depthPassReady; }

private:
    GaussianRenderManager() = default;
    ~GaussianRenderManager();
    GaussianRenderManager(const GaussianRenderManager&) = delete;
    GaussianRenderManager& operator=(const GaussianRenderManager&) = delete;

    // --- Frame state ---
    uint64_t                   m_frameStamp    = 0;
    bool                       m_frameRendered = false;
    std::vector<RenderInstance> m_instances;
    uint32_t                   m_totalSplats   = 0;

    // Camera / viewport (set once per frame)
    float m_viewMat[16]    = {};
    float m_projMat[16]    = {};
    float m_cameraPos[3]   = {};
    float m_tanHalfFov[2]  = {};
    float m_vpWidth        = 0.f;
    float m_vpHeight       = 0.f;

    // --- Pipeline ready flags ---
    bool m_pipelineReady   = false;
    bool m_sortReady       = false;
    bool m_depthPassReady  = false;

    // --- Merged input StructuredBuffers (concatenated from all instances) ---
    ID3D11Buffer*             m_mergedPositionWS  = nullptr;
    ID3D11ShaderResourceView* m_mergedSrvPosWS    = nullptr;
    ID3D11Buffer*             m_mergedScale       = nullptr;
    ID3D11ShaderResourceView* m_mergedSrvScale    = nullptr;
    ID3D11Buffer*             m_mergedRotation    = nullptr;
    ID3D11ShaderResourceView* m_mergedSrvRotation = nullptr;
    ID3D11Buffer*             m_mergedOpacity     = nullptr;
    ID3D11ShaderResourceView* m_mergedSrvOpacity  = nullptr;
    ID3D11Buffer*             m_mergedSHCoeffs    = nullptr;
    ID3D11ShaderResourceView* m_mergedSrvSH       = nullptr;

    // Per-splat instance ID (indexes into worldMats)
    ID3D11Buffer*             m_instanceIDBuf     = nullptr;
    ID3D11ShaderResourceView* m_instanceIDSrv     = nullptr;

    // Per-instance world matrices
    ID3D11Buffer*             m_worldMatsBuf      = nullptr;
    ID3D11ShaderResourceView* m_worldMatsSrv      = nullptr;

    // Merged per-splat selection mask (concatenated from all instances' masks).
    // Rebuilt when instance set or any instance's mask changes.
    ID3D11Buffer*             m_mergedSelection    = nullptr;
    ID3D11ShaderResourceView* m_srvMergedSelection = nullptr;
    bool                      m_selectionDirty     = true;
    std::vector<uint64_t>     m_instanceMaskVersions;  // last seen per-instance version

    uint32_t m_mergedAllocN = 0;   // currently allocated merged capacity
    uint32_t m_mergedAllocInstances = 0;

    // Cache signature: detect when the instance set changes so we only
    // rebuild the large concatenated input buffers on actual changes.
    // Signature = hash of (dataNode pointer, splatCount) per instance.
    size_t m_cachedSignature = 0;
    bool   m_inputsUploaded  = false;  // true once large buffers are valid

    // --- Compute outputs (written by preprocess, read by sort & render) ---
    ID3D11Buffer*              m_ubPositionSS  = nullptr;
    ID3D11UnorderedAccessView* m_uavPositionSS = nullptr;
    ID3D11ShaderResourceView*  m_srvPositionSS = nullptr;

    ID3D11Buffer*              m_ubDepth    = nullptr;
    ID3D11UnorderedAccessView* m_uavDepth   = nullptr;
    ID3D11ShaderResourceView*  m_srvDepth   = nullptr;

    ID3D11Buffer*              m_ubRadius   = nullptr;
    ID3D11UnorderedAccessView* m_uavRadius  = nullptr;
    ID3D11ShaderResourceView*  m_srvRadius  = nullptr;

    ID3D11Buffer*              m_ubColor    = nullptr;
    ID3D11UnorderedAccessView* m_uavColor   = nullptr;
    ID3D11ShaderResourceView*  m_srvColor   = nullptr;

    ID3D11Buffer*              m_ubCov2D    = nullptr;
    ID3D11UnorderedAccessView* m_uavCov2D   = nullptr;
    ID3D11ShaderResourceView*  m_srvCov2D   = nullptr;

    // --- Shaders ---
    ID3D11ComputeShader*  m_preprocessCS  = nullptr;
    ID3D11Buffer*         m_preprocessCB  = nullptr;
    ID3D11VertexShader*   m_prodVS        = nullptr;
    ID3D11PixelShader*    m_prodPS        = nullptr;
    ID3D11Buffer*         m_prodCB        = nullptr;

    // Render states
    ID3D11BlendState*        m_blendState = nullptr;
    ID3D11RasterizerState*   m_rsState    = nullptr;
    ID3D11DepthStencilState* m_dsState    = nullptr;

    // --- Selection CS (marquee) ---
    ID3D11ComputeShader*       m_selectCS       = nullptr;
    ID3D11Buffer*              m_selectCB       = nullptr;
    bool                       m_selectReady    = false;

    // --- Sort ---
    ID3D11ComputeShader*       m_sortCS_keygen  = nullptr;
    ID3D11ComputeShader*       m_sortCS_count   = nullptr;
    ID3D11ComputeShader*       m_sortCS_scan    = nullptr;
    ID3D11ComputeShader*       m_sortCS_scatter = nullptr;
    ID3D11Buffer*              m_sortCB         = nullptr;

    ID3D11Buffer*              m_sortKeysA      = nullptr;
    ID3D11UnorderedAccessView* m_sortKeysA_UAV  = nullptr;
    ID3D11ShaderResourceView*  m_sortKeysA_SRV  = nullptr;
    ID3D11Buffer*              m_sortKeysB      = nullptr;
    ID3D11UnorderedAccessView* m_sortKeysB_UAV  = nullptr;
    ID3D11ShaderResourceView*  m_sortKeysB_SRV  = nullptr;

    ID3D11Buffer*              m_sortValsA      = nullptr;
    ID3D11UnorderedAccessView* m_sortValsA_UAV  = nullptr;
    ID3D11ShaderResourceView*  m_sortValsA_SRV  = nullptr;
    ID3D11Buffer*              m_sortValsB      = nullptr;
    ID3D11UnorderedAccessView* m_sortValsB_UAV  = nullptr;
    ID3D11ShaderResourceView*  m_sortValsB_SRV  = nullptr;

    ID3D11Buffer*              m_sortBlockHist      = nullptr;
    ID3D11UnorderedAccessView* m_sortBlockHist_UAV  = nullptr;
    ID3D11ShaderResourceView*  m_sortBlockHist_SRV  = nullptr;

    // --- Depth pass ---
    ID3D11ComputeShader*       m_depthClearCS   = nullptr;
    ID3D11ComputeShader*       m_depthPassCS    = nullptr;
    ID3D11Buffer*              m_depthCB        = nullptr;
    ID3D11Texture2D*           m_depthTex       = nullptr;
    ID3D11UnorderedAccessView* m_depthTex_UAV   = nullptr;
    ID3D11ShaderResourceView*  m_depthTex_SRV   = nullptr;
    uint32_t                   m_depthTexW      = 0;
    uint32_t                   m_depthTexH      = 0;
    ID3D11VertexShader*        m_depthCopyVS    = nullptr;
    ID3D11PixelShader*         m_depthCopyPS    = nullptr;
    ID3D11DepthStencilState*   m_depthWriteDS   = nullptr;
    ID3D11BlendState*          m_depthCopyBlend = nullptr;

    // --- Init helpers ---
    bool initPipeline(ID3D11Device* device);
    bool initSortPipeline(ID3D11Device* device);
    bool initDepthPassPipeline(ID3D11Device* device);
    bool initSelectPipeline(ID3D11Device* device);

    // Build/refresh m_mergedSelection by concatenating per-instance masks.
    // Called from render(). Skips work if neither the instance set nor any
    // instance's mask version has changed since last call.
    bool updateMergedSelection(ID3D11Device* device, ID3D11DeviceContext* ctx);

    // --- Buffer management ---
    bool buildMergedInputs(ID3D11Device* device, ID3D11DeviceContext* ctx);
    bool createComputeOutputs(ID3D11Device* device, uint32_t N);
    bool createSortBuffers(ID3D11Device* device, uint32_t N);
    bool createDepthTexture(ID3D11Device* device, uint32_t w, uint32_t h);

    bool createUAVBuffer(ID3D11Device* device, const char* name,
                         uint32_t numElements, uint32_t stride,
                         ID3D11Buffer** outBuf,
                         ID3D11UnorderedAccessView** outUAV,
                         ID3D11ShaderResourceView** outSRV);

    bool createSRVBuffer(ID3D11Device* device, const char* name,
                         const void* initData, uint32_t numElements, uint32_t stride,
                         ID3D11Buffer** outBuf, ID3D11ShaderResourceView** outSRV);

    void releaseMergedInputs();
    void releaseComputeOutputs();
    void releaseSortBuffers();
    void releaseDepthPassResources();
    void releasePipeline();
};
