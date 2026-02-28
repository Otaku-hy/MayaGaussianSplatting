#pragma once
#include <maya/MPxGeometryOverride.h>
#include <maya/MString.h>
#include <maya/MObject.h>

class GaussianNode;

// ---------------------------------------------------------------------------
// GaussianGeometryOverride  —  VP2 geometry override for GaussianNode.
//
// Responsibilities:
//   updateDG()          – detect PLY path change, (re)load file
//   updateRenderItems() – register/update the debug point render item + shader
//   populateGeometry()  – upload position and colour vertex buffers to GPU
// ---------------------------------------------------------------------------
class GaussianGeometryOverride : public MHWRender::MPxGeometryOverride {
public:
    static MHWRender::MPxGeometryOverride* creator(const MObject& obj);

    explicit GaussianGeometryOverride(const MObject& obj);
    ~GaussianGeometryOverride() override;

    MHWRender::DrawAPI supportedDrawAPIs() const override;

    void updateDG() override;

    void updateRenderItems(const MDagPath&              path,
                           MHWRender::MRenderItemList&  list) override;

    void populateGeometry(const MHWRender::MGeometryRequirements& requirements,
                          const MHWRender::MRenderItemList&       renderItems,
                          MHWRender::MGeometry&                   data) override;

    void cleanUp() override;

    bool hasUIDrawables()        const override { return false; }
    bool requiresGeometryUpdate() const override { return m_geometryDirty; }

    // Called once from initializePlugin with the path derived from
    // MFnPlugin::loadPath(), e.g. "<pluginDir>/../shader/gaussianDebug.fx"
    static void setShaderPath(const MString& path) { s_shaderPath = path; }

private:
    void loadShader();

    GaussianNode*                m_node         = nullptr;
    MHWRender::MShaderInstance*  m_shader       = nullptr;
    mutable bool                 m_geometryDirty = true;

    static MString s_shaderPath;
    static const MString kRenderItemName;
};
