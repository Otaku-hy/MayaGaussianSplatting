#pragma once
#include <maya/MPxNode.h>
#include <maya/MString.h>
#include <maya/MTypeId.h>
#include <maya/MObject.h>
#include <maya/MBoundingBox.h>
#include <d3d11.h>
#include <cstdint>

#include "GaussianData.h"

// ---------------------------------------------------------------------------
// GaussianDataNode  --  MPxNode that owns a loaded .ply file's data.
//
// Multiple gaussianSplat render nodes can connect to one data node
// to share the same CPU data and GPU input buffers.
//
// Attributes:
//   filePath   (string, input)  --  path to the .ply file
//   dataReady  (bool,   output) --  set when PLY is loaded
//   outputData (message, output) -- connection point for render nodes
// ---------------------------------------------------------------------------
class GaussianDataNode : public MPxNode {
public:
    static void*   creator();
    static MStatus initialize();

    MStatus compute(const MPlug& plug, MDataBlock& dataBlock) override;

    // Static identifiers
    static MTypeId typeId;
    static MString typeName;

    // Maya attributes
    static MObject aFilePath;
    static MObject aDataReady;
    static MObject aOutputData;   // message attribute for downstream connections

    // Read-only access for render nodes
    const GaussianData& gaussianData() const { return m_data; }
    uint32_t            splatCount()   const { return (uint32_t)m_data.count(); }
    bool                hasData()      const { return !m_data.empty(); }
    MBoundingBox        boundingBox()  const;   // object-space bbox of loaded splats

    // GPU input buffer management (lazy upload, called from prepareForDraw)
    bool uploadInputBuffersIfNeeded(ID3D11Device* device);
    bool areInputsReady() const { return m_inputsReady; }

    // Non-owning SRV accessors (valid only after upload)
    ID3D11ShaderResourceView* srvPositionWS() const { return m_srvPositionWS; }
    ID3D11ShaderResourceView* srvScale()      const { return m_srvScale; }
    ID3D11ShaderResourceView* srvRotation()   const { return m_srvRotation; }
    ID3D11ShaderResourceView* srvOpacity()    const { return m_srvOpacity; }
    ID3D11ShaderResourceView* srvSHCoeffs()   const { return m_srvSHCoeffs; }

    // --- Selection mask (1 uint per splat; bit 0=selected, bit 1=deleted) ---
    ID3D11ShaderResourceView*  srvSelectionMask() const { return m_srvSelectionMask; }
    ID3D11UnorderedAccessView* uavSelectionMask() const { return m_uavSelectionMask; }
    ID3D11Buffer*              bufSelectionMask() const { return m_sbSelectionMask; }

    // Zero the entire mask (clears selection + deleted) on GPU and CPU shadow.
    // Bumps m_maskVersion so observers (RenderManager) can detect changes.
    void restoreAll(ID3D11DeviceContext* ctx);

    // Clear only bit 0 (selection) on GPU and CPU shadow.
    void clearSelection(ID3D11DeviceContext* ctx);

    // Promote current selection (bit 0) to deleted (bit 1); clears bit 0.
    // Runs on GPU then reads back to CPU shadow.
    void deleteSelected(ID3D11DeviceContext* ctx);

    // Copy GPU mask into m_maskShadow (for save / inspection).
    // Slow — only call on demand (e.g. Save PLY).
    bool readbackMask(ID3D11Device* device, ID3D11DeviceContext* ctx);

    const std::vector<uint32_t>& maskShadow()        const { return m_maskShadow; }
    std::vector<uint32_t>&       maskShadowMutable()       { return m_maskShadow; }

    // Monotonic counter bumped every time the mask changes (by any helper).
    // RenderManager compares this to its cached value to decide when to
    // re-concatenate the merged selection buffer.
    uint64_t maskVersion() const { return m_maskVersion; }

    // Bump the mask version (call after writing the GPU mask buffer
    // directly via compute shader, bypassing the helpers above).
    void markMaskChanged() { m_maskVersion++; }

    ~GaussianDataNode() override;

private:
    GaussianData m_data;
    MString      m_loadedPath;

    // GPU input StructuredBuffers (shared across all connected render nodes)
    ID3D11Buffer*             m_sbPositionWS  = nullptr;
    ID3D11ShaderResourceView* m_srvPositionWS = nullptr;
    ID3D11Buffer*             m_sbScale       = nullptr;
    ID3D11ShaderResourceView* m_srvScale      = nullptr;
    ID3D11Buffer*             m_sbRotation    = nullptr;
    ID3D11ShaderResourceView* m_srvRotation   = nullptr;
    ID3D11Buffer*             m_sbOpacity     = nullptr;
    ID3D11ShaderResourceView* m_srvOpacity    = nullptr;
    ID3D11Buffer*             m_sbSHCoeffs    = nullptr;
    ID3D11ShaderResourceView* m_srvSHCoeffs   = nullptr;

    // Selection mask (one uint per splat)
    ID3D11Buffer*              m_sbSelectionMask  = nullptr;
    ID3D11ShaderResourceView*  m_srvSelectionMask = nullptr;
    ID3D11UnorderedAccessView* m_uavSelectionMask = nullptr;
    std::vector<uint32_t>      m_maskShadow;    // updated by readbackMask() / helpers
    uint64_t                   m_maskVersion    = 0;

    bool m_inputsReady = false;
    bool m_inputsDirty = true;

    bool createSelectionMaskBuffer(ID3D11Device* device, uint32_t N);
    void releaseSelectionMaskBuffer();
    void releaseInputBuffers();

    static bool createSRVBuffer(ID3D11Device* device,
                                const char*   name,
                                const void*   initData,
                                uint32_t      numElements,
                                uint32_t      stride,
                                ID3D11Buffer**             outBuf,
                                ID3D11ShaderResourceView** outSRV);
};
