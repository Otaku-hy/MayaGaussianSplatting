#pragma once
#include <maya/MPxLocatorNode.h>
#include <maya/MString.h>
#include <maya/MTypeId.h>
#include <maya/MObject.h>
#include <maya/MBoundingBox.h>
#include <d3d11.h>
#include <cstdint>
#include <vector>
#include "GaussianData.h"

// ---------------------------------------------------------------------------
// GaussianNode  --  self-contained MPxLocatorNode.
//
// Owns a loaded .ply file's CPU data AND all DX11 GPU input buffers.
// Each gaussianSplat node is fully independent (no shared data node).
//
// Attributes:
//   filePath   (string, input)   -- path to the .ply file
//   dataReady  (bool,   output)  -- set true once PLY is loaded
//   pointSize  (float)           -- debug display point radius in pixels
//   renderMode (int, 0-3)        -- 0=auto, 1=debug, 2=prod, 3=diag
// ---------------------------------------------------------------------------
class GaussianNode : public MPxLocatorNode {
public:
    static void*   creator();
    static MStatus initialize();

    MStatus compute(const MPlug& plug, MDataBlock& dataBlock) override;

    bool         isBounded()   const override { return true; }
    MBoundingBox boundingBox() const override;

    ~GaussianNode() override;

    // Static identifiers
    static MTypeId typeId;
    static MString typeName;
    static MString drawDbClassification;
    static MString drawRegistrantId;

    // Maya attributes
    static MObject aFilePath;
    static MObject aDataReady;
    static MObject aPointSize;
    static MObject aRenderMode;

    // --- CPU data ---
    const GaussianData& gaussianData() const { return m_data; }
    bool     hasData()    const { return !m_data.empty(); }
    uint32_t splatCount() const { return (uint32_t)m_data.count(); }

    // --- GPU input buffers (lazy upload, called from prepareForDraw) ---
    bool uploadInputBuffersIfNeeded(ID3D11Device* device);
    bool areInputsReady() const { return m_inputsReady; }

    ID3D11ShaderResourceView* srvPositionWS() const { return m_srvPositionWS; }
    ID3D11ShaderResourceView* srvScale()      const { return m_srvScale; }
    ID3D11ShaderResourceView* srvRotation()   const { return m_srvRotation; }
    ID3D11ShaderResourceView* srvOpacity()    const { return m_srvOpacity; }
    ID3D11ShaderResourceView* srvSHCoeffs()   const { return m_srvSHCoeffs; }

    // --- Selection mask (one uint per splat; bit0=selected, bit1=deleted) ---
    ID3D11Buffer*              bufSelectionMask() const { return m_sbSelectionMask; }
    ID3D11ShaderResourceView*  srvSelectionMask() const { return m_srvSelectionMask; }
    ID3D11UnorderedAccessView* uavSelectionMask() const { return m_uavSelectionMask; }

    const std::vector<uint32_t>& maskShadow()       const { return m_maskShadow; }
    std::vector<uint32_t>&       maskShadowMutable()      { return m_maskShadow; }

    uint64_t maskVersion()  const { return m_maskVersion; }
    void markMaskChanged()        { m_maskVersion++; }

    void restoreAll    (ID3D11DeviceContext* ctx);
    void clearSelection(ID3D11DeviceContext* ctx);
    void deleteSelected(ID3D11DeviceContext* ctx);
    bool readbackMask  (ID3D11Device* device, ID3D11DeviceContext* ctx);

private:
    friend class GaussianDrawOverride;

    GaussianData m_data;
    MString      m_loadedPath;

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

    ID3D11Buffer*              m_sbSelectionMask  = nullptr;
    ID3D11ShaderResourceView*  m_srvSelectionMask = nullptr;
    ID3D11UnorderedAccessView* m_uavSelectionMask = nullptr;
    std::vector<uint32_t>      m_maskShadow;
    uint64_t                   m_maskVersion = 0;

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
