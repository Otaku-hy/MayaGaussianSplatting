#include "GaussianGeometryOverride.h"
#include "GaussianNode.h"
#include "PLYReader.h"

#include <maya/MDagPath.h>
#include <maya/MFnDependencyNode.h>
#include <maya/MPlug.h>
#include <maya/MGlobal.h>

#include <maya/MHWGeometry.h>
#include <maya/MShaderManager.h>
#include <maya/MViewport2Renderer.h>
#include <maya/MGeometryRequirements.h>

#include <cstring>

const MString GaussianGeometryOverride::kRenderItemName { "gaussianDebugPoints" };
MString       GaussianGeometryOverride::s_shaderPath    {};

// ---------------------------------------------------------------------------
MHWRender::MPxGeometryOverride*
GaussianGeometryOverride::creator(const MObject& obj) {
    return new GaussianGeometryOverride(obj);
}

GaussianGeometryOverride::GaussianGeometryOverride(const MObject& obj)
    : MHWRender::MPxGeometryOverride(obj)
{
    MFnDependencyNode fn(obj);
    m_node = dynamic_cast<GaussianNode*>(fn.userNode());
}

GaussianGeometryOverride::~GaussianGeometryOverride() {
    if (m_shader) {
        const MHWRender::MRenderer* r = MHWRender::MRenderer::theRenderer();
        if (r) r->getShaderManager()->releaseShader(m_shader);
        m_shader = nullptr;
    }
}

// ---------------------------------------------------------------------------
MHWRender::DrawAPI GaussianGeometryOverride::supportedDrawAPIs() const {
    // Prefer DirectX 11; OpenGL kept as fallback for non-Windows platforms.
    // NOTE: The GS-based point expansion in the .fx shader requires DX11.
    //       On OpenGL this render item will simply not draw until an OGSFX
    //       variant is added.
    return MHWRender::kDirectX11 | MHWRender::kOpenGLCoreProfile;
}

// ---------------------------------------------------------------------------
// updateDG  —  runs in the main thread; safe to access Maya DG and do I/O
// ---------------------------------------------------------------------------
void GaussianGeometryOverride::updateDG() {
    if (!m_node) return;

    MPlug   pathPlug(m_node->thisMObject(), GaussianNode::aFilePath);
    MString newPath = pathPlug.asString();

    // Only reload when the path actually changes
    if (newPath == m_node->m_loadedPath) return;
    m_node->m_loadedPath = newPath;

    m_node->m_data.clear();
    if (newPath.length() == 0) return;

    std::string errMsg;
    bool ok = PLYReader::read(newPath.asChar(), m_node->m_data, errMsg);
    if (ok) {
        MGlobal::displayInfo(
            MString("[GaussianSplat] Loaded ") +
            (unsigned int)m_node->m_data.count() +
            " splats from: " + newPath);
        m_geometryDirty = true;   // signal populateGeometry to re-upload
    } else {
        MGlobal::displayError(MString("[GaussianSplat] ") + errMsg.c_str());
    }
}

// ---------------------------------------------------------------------------
// loadShader  —  locate and compile gaussianDebug.fx
// ---------------------------------------------------------------------------
void GaussianGeometryOverride::loadShader() {
    const MHWRender::MRenderer*      renderer  = MHWRender::MRenderer::theRenderer();
    const MHWRender::MShaderManager* shaderMgr = renderer->getShaderManager();

    // s_shaderPath is set once in initializePlugin via setShaderPath().
    // It points to  <pluginDir>/../shader/gaussianDebug.fx
    if (s_shaderPath.length() == 0) {
        MGlobal::displayError("[GaussianSplat] Shader path not set – call "
                              "GaussianGeometryOverride::setShaderPath() in initializePlugin.");
        return;
    }

    m_shader = shaderMgr->getEffectsFileShader(s_shaderPath, "Main",
                                               /*macros=*/nullptr, 0,
                                               /*useEffectCache=*/true);
    if (!m_shader) {
        MGlobal::displayError(
            MString("[GaussianSplat] Failed to load shader: ") + s_shaderPath);
    }
}

// ---------------------------------------------------------------------------
// updateRenderItems
// ---------------------------------------------------------------------------
void GaussianGeometryOverride::updateRenderItems(
    const MDagPath&             /*path*/,
    MHWRender::MRenderItemList& list)
{
    // Lazy-load shader once
    if (!m_shader) loadShader();

    MHWRender::MRenderItem* item = nullptr;
    int idx = list.indexOf(kRenderItemName);
    if (idx < 0) {
        // Render item type = NonShadedItem so VP2 won't try to override
        // the shader with a scene material.
        item = MHWRender::MRenderItem::Create(
            kRenderItemName,
            MHWRender::MRenderItem::NonMaterialSceneItem,
            MHWRender::MGeometry::kPoints);
        item->setDrawMode(MHWRender::MGeometry::kAll);
        item->depthPriority(MHWRender::MRenderItem::sActiveWireDepthPriority);
        list.append(item);
    } else {
        item = list.itemAt(idx);
    }
    if (!item) return;

    if (m_shader) {
        // Push per-frame parameters
        MPlug psPlug(m_node->thisMObject(), GaussianNode::aPointSize);
        float pointSize = psPlug.asFloat();
        m_shader->setParameter("gPointSize", pointSize);

        // Viewport size – query from the renderer
        unsigned int vpW = 1280, vpH = 720;
        MHWRender::MRenderer::theRenderer()->outputTargetSize(vpW, vpH);
        float vps[2] = { (float)vpW, (float)vpH };
        m_shader->setParameter("gViewportSize", vps);

        item->setShader(m_shader);
    }

    item->enable(!m_node->m_data.empty());
}

// ---------------------------------------------------------------------------
// populateGeometry  —  upload vertex buffers to the GPU
// ---------------------------------------------------------------------------
void GaussianGeometryOverride::populateGeometry(
    const MHWRender::MGeometryRequirements& requirements,
    const MHWRender::MRenderItemList&       /*renderItems*/,
    MHWRender::MGeometry&                   data)
{
    if (!m_node || m_node->m_data.empty()) return;

    const GaussianData& gd    = m_node->m_data;
    const unsigned int  count = (unsigned int)gd.count();

    // Create buffers directly without relying on requirements enumeration.
    // In Maya 2026 VP2 with HLSL .fx shaders, vertexRequirements() may return
    // an empty list, leaving all vertex positions at [0,0,0].

    // Position buffer (float3, semantic POSITION)
    {
        MHWRender::MVertexBufferDescriptor desc(
            "", MHWRender::MGeometry::kPosition, MHWRender::MGeometry::kFloat, 3);
        MHWRender::MVertexBuffer* buf = data.createVertexBuffer(desc);
        if (buf) {
            float* dst = static_cast<float*>(buf->acquire(count, /*writeOnly=*/true));
            if (dst) {
                std::memcpy(dst, gd.positions.data(), count * 3 * sizeof(float));
                buf->commit(dst);
            }
        }
    }

    // Color buffer (float4, semantic COLOR0)
    {
        MHWRender::MVertexBufferDescriptor desc(
            "", MHWRender::MGeometry::kColor, MHWRender::MGeometry::kFloat, 4);
        MHWRender::MVertexBuffer* buf = data.createVertexBuffer(desc);
        if (buf) {
            float* dst = static_cast<float*>(buf->acquire(count, /*writeOnly=*/true));
            if (dst) {
                std::memcpy(dst, gd.colors.data(), count * 4 * sizeof(float));
                buf->commit(dst);
            }
        }
    }

    // No index buffer: for kPoints, Maya derives draw count directly from
    // the vertex buffer size. A manually created index buffer is not
    // correctly associated with the render item and causes DrawIndexed(1).

    m_geometryDirty = false;
}

// ---------------------------------------------------------------------------
void GaussianGeometryOverride::cleanUp() {}
