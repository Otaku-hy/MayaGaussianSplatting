#define NOMINMAX
#include "GaussianSelection.h"
#include "GaussianNode.h"
#include "GaussianRenderManager.h"
#include "GaussianData.h"

#include <maya/MGlobal.h>
#include <maya/MArgDatabase.h>
#include <maya/MSelectionList.h>
#include <maya/MItDependencyNodes.h>
#include <maya/MFnDependencyNode.h>
#include <maya/MFnDagNode.h>
#include <maya/MDagPath.h>
#include <maya/MMatrix.h>
#include <maya/MRenderUtil.h>
#include <maya/MViewport2Renderer.h>
#include <maya/M3dView.h>
#include <maya/MUiMessage.h>

#include <d3d11.h>
#include <algorithm>
#include <vector>
#include <cstdio>
#include <cstring>
#include <cstdint>

// ===========================================================================
// Helpers
// ===========================================================================
namespace {

ID3D11Device* getDX11Device() {
    MHWRender::MRenderer* r = MHWRender::MRenderer::theRenderer();
    if (!r) return nullptr;
    return (ID3D11Device*)r->GPUDeviceHandle();
}

ID3D11DeviceContext* getDX11Context(ID3D11Device* device) {
    ID3D11DeviceContext* ctx = nullptr;
    if (device) device->GetImmediateContext(&ctx);
    return ctx;
}

// ---------------------------------------------------------------------------
// Scene enumeration helpers
// ---------------------------------------------------------------------------

struct RenderPair {
    MDagPath      dagPath;
    GaussianNode* node = nullptr;
};

// Collect all gaussianSplat locator nodes in the scene.
void collectAllNodes(std::vector<RenderPair>& out) {
    MItDependencyNodes it(MFn::kPluginLocatorNode);
    for (; !it.isDone(); it.next()) {
        MObject obj = it.thisNode();
        MFnDependencyNode fn(obj);
        if (fn.typeName() != "gaussianSplat") continue;
        MFnDagNode dag(obj);
        if (dag.isDefaultNode()) continue;
        MDagPath path;
        if (dag.getPath(path) != MS::kSuccess) continue;
        GaussianNode* n = static_cast<GaussianNode*>(fn.userNode());
        if (n) { RenderPair p; p.dagPath = path; p.node = n; out.push_back(p); }
    }
}

// Collect gaussianSplat nodes:
//   - If any gaussianSplat (or its transform) is currently selected in Maya,
//     return only those (scoped operation).
//   - Otherwise fall back to all nodes in the scene.
void collectRenderPairs(std::vector<RenderPair>& out) {
    MSelectionList sel;
    MGlobal::getActiveSelectionList(sel);

    for (unsigned int i = 0; i < sel.length(); i++) {
        MDagPath path;
        if (sel.getDagPath(i, path) != MS::kSuccess) continue;

        // If transform, look for gaussianSplat shapes underneath
        if (path.node().hasFn(MFn::kTransform)) {
            unsigned int nShapes = 0;
            path.numberOfShapesDirectlyBelow(nShapes);
            for (unsigned int j = 0; j < nShapes; j++) {
                MDagPath shapePath = path;
                if (shapePath.extendToShapeDirectlyBelow(j) != MS::kSuccess) continue;
                MFnDependencyNode fn(shapePath.node());
                if (fn.typeName() != "gaussianSplat") continue;
                GaussianNode* n = static_cast<GaussianNode*>(fn.userNode());
                if (n) { RenderPair p; p.dagPath = shapePath; p.node = n; out.push_back(p); }
            }
        } else {
            MFnDependencyNode fn(path.node());
            if (fn.typeName() != "gaussianSplat") continue;
            GaussianNode* n = static_cast<GaussianNode*>(fn.userNode());
            if (n) { RenderPair p; p.dagPath = path; p.node = n; out.push_back(p); }
        }
    }

    if (out.empty()) collectAllNodes(out);  // fall back to all
}

// Collect all gaussianSplat nodes (for non-scoped commands like gsClearSelection).
void collectAllGaussianNodes(std::vector<GaussianNode*>& out) {
    MItDependencyNodes it(MFn::kPluginLocatorNode);
    for (; !it.isDone(); it.next()) {
        MObject obj = it.thisNode();
        MFnDependencyNode fn(obj);
        if (fn.typeName() != "gaussianSplat") continue;
        GaussianNode* n = static_cast<GaussianNode*>(fn.userNode());
        if (n) out.push_back(n);
    }
}

GaussianNode* findNodeByName(const MString& name) {
    MSelectionList sel;
    if (sel.add(name) != MS::kSuccess) return nullptr;
    MObject obj;
    sel.getDependNode(0, obj);
    MFnDependencyNode fn(obj);
    if (fn.typeName() != "gaussianSplat") return nullptr;
    return static_cast<GaussianNode*>(fn.userNode());
}

// Multiply two row-major 4x4 matrices: out = a * b
void matMul4x4(const float a[16], const float b[16], float out[16]) {
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) {
            float s = 0.f;
            for (int k = 0; k < 4; k++)
                s += a[r*4+k] * b[k*4+c];
            out[r*4+c] = s;
        }
}

// Convert MMatrix (row-major in Maya: m[row][col]) to float[16]
void mmatrixToFloat16(const MMatrix& m, float out[16]) {
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            out[r*4+c] = (float)m[r][c];
}

} // namespace

// ===========================================================================
// gsMarqueeSelect
// ===========================================================================
const MString GSMarqueeSelectCmd::commandName("gsMarqueeSelect");

MSyntax GSMarqueeSelectCmd::newSyntax() {
    MSyntax s;
    s.addFlag("-min", "-min", MSyntax::kDouble, MSyntax::kDouble);
    s.addFlag("-max", "-max", MSyntax::kDouble, MSyntax::kDouble);
    s.addFlag("-mo",  "-mode", MSyntax::kLong);
    return s;
}

MStatus GSMarqueeSelectCmd::doIt(const MArgList& args) {
    MStatus st;
    MArgDatabase db(syntax(), args, &st);
    if (!st) return st;

    double x0 = -1.0, y0 = -1.0, x1 = 1.0, y1 = 1.0;
    int mode = 0;
    if (db.isFlagSet("-min")) { db.getFlagArgument("-min", 0, x0); db.getFlagArgument("-min", 1, y0); }
    if (db.isFlagSet("-max")) { db.getFlagArgument("-max", 0, x1); db.getFlagArgument("-max", 1, y1); }
    if (db.isFlagSet("-mo"))    db.getFlagArgument("-mo", 0, mode);

    ID3D11Device* device = getDX11Device();
    if (!device) { displayError("No DX11 device (is Viewport 2.0 using DirectX 11?)"); return MS::kFailure; }
    ID3D11DeviceContext* ctx = getDX11Context(device);
    if (!ctx) { displayError("No DX11 context."); return MS::kFailure; }

    auto& mgr = GaussianRenderManager::instance();
    if (mgr.viewportWidth() <= 0.f || mgr.viewportHeight() <= 0.f) {
        displayError("Render manager has no viewport data yet. Did a frame render?");
        ctx->Release();
        return MS::kFailure;
    }

    // viewProj = view * proj (row-major)
    float viewProj[16];
    matMul4x4(mgr.viewMatrix(), mgr.projMatrix(), viewProj);

    std::vector<RenderPair> pairs;
    collectRenderPairs(pairs);

    int nSuccess = 0;
    for (const auto& p : pairs) {
        if (!p.node || !p.node->areInputsReady()) continue;
        float worldMat[16];
        mmatrixToFloat16(p.dagPath.inclusiveMatrix(), worldMat);
        if (mgr.runSelection(device, ctx, p.node,
                             worldMat, viewProj,
                             (float)x0, (float)y0, (float)x1, (float)y1,
                             mode))
            nSuccess++;
    }

    ctx->Release();

    // Force viewport refresh so the highlight appears immediately
    M3dView::active3dView().refresh(false, true);

    setResult(nSuccess);
    return MS::kSuccess;
}

// ===========================================================================
// gsClearSelection
// ===========================================================================
const MString GSClearSelectionCmd::commandName("gsClearSelection");

MStatus GSClearSelectionCmd::doIt(const MArgList&) {
    ID3D11Device* device = getDX11Device();
    if (!device) return MS::kFailure;
    ID3D11DeviceContext* ctx = getDX11Context(device);
    if (!ctx) return MS::kFailure;

    std::vector<GaussianNode*> nodes;
    collectAllGaussianNodes(nodes);
    for (GaussianNode* n : nodes)
        if (n->areInputsReady()) n->clearSelection(ctx);

    GaussianRenderManager::instance().markSelectionDirty();
    ctx->Release();
    M3dView::active3dView().refresh(false, true);
    return MS::kSuccess;
}

// ===========================================================================
// gsDeleteSelected
// ===========================================================================
const MString GSDeleteSelectedCmd::commandName("gsDeleteSelected");

MStatus GSDeleteSelectedCmd::doIt(const MArgList&) {
    ID3D11Device* device = getDX11Device();
    if (!device) return MS::kFailure;
    ID3D11DeviceContext* ctx = getDX11Context(device);
    if (!ctx) return MS::kFailure;

    std::vector<GaussianNode*> nodes;
    collectAllGaussianNodes(nodes);
    for (GaussianNode* n : nodes)
        if (n->areInputsReady()) n->deleteSelected(ctx);

    GaussianRenderManager::instance().markSelectionDirty();
    ctx->Release();
    M3dView::active3dView().refresh(false, true);
    return MS::kSuccess;
}

// ===========================================================================
// gsRestoreAll
// ===========================================================================
const MString GSRestoreAllCmd::commandName("gsRestoreAll");

MSyntax GSRestoreAllCmd::newSyntax() {
    MSyntax s;
    s.addFlag("-n", "-node", MSyntax::kString);
    return s;
}

MStatus GSRestoreAllCmd::doIt(const MArgList& args) {
    MStatus st;
    MArgDatabase db(syntax(), args, &st);
    if (!st) return st;

    ID3D11Device* device = getDX11Device();
    if (!device) return MS::kFailure;
    ID3D11DeviceContext* ctx = getDX11Context(device);
    if (!ctx) return MS::kFailure;

    if (db.isFlagSet("-n")) {
        MString name; db.getFlagArgument("-n", 0, name);
        GaussianNode* n = findNodeByName(name);
        if (!n) {
            displayError(MString("gsRestoreAll: no gaussianSplat named ") + name);
            ctx->Release();
            return MS::kFailure;
        }
        if (n->areInputsReady()) n->restoreAll(ctx);
    } else {
        std::vector<GaussianNode*> nodes;
        collectAllGaussianNodes(nodes);
        for (GaussianNode* n : nodes)
            if (n->areInputsReady()) n->restoreAll(ctx);
    }

    GaussianRenderManager::instance().markSelectionDirty();
    ctx->Release();
    M3dView::active3dView().refresh(false, true);
    return MS::kSuccess;
}

// ===========================================================================
// gsSavePLY  --  write mask-filtered data node to binary_little_endian PLY.
// ===========================================================================
const MString GSSavePLYCmd::commandName("gsSavePLY");

MSyntax GSSavePLYCmd::newSyntax() {
    MSyntax s;
    s.addFlag("-f", "-file", MSyntax::kString);
    s.addFlag("-n", "-node", MSyntax::kString);
    return s;
}

static bool writeBinaryPLY(const char* path, const GaussianData& data,
                            const std::vector<uint32_t>& mask,
                            size_t& keptOut)
{
    size_t N = data.splats.size();
    size_t kept = 0;
    for (size_t i = 0; i < N; i++)
        if (!(mask[i] & kMaskBitDeleted)) kept++;
    keptOut = kept;

    FILE* f = nullptr;
    if (fopen_s(&f, path, "wb") != 0 || !f) return false;

    // Header
    fprintf(f, "ply\n");
    fprintf(f, "format binary_little_endian 1.0\n");
    fprintf(f, "element vertex %zu\n", kept);
    fprintf(f, "property float x\nproperty float y\nproperty float z\n");
    fprintf(f, "property float nx\nproperty float ny\nproperty float nz\n");
    fprintf(f, "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n");
    for (int i = 0; i < 45; i++) fprintf(f, "property float f_rest_%d\n", i);
    fprintf(f, "property float opacity\n");
    fprintf(f, "property float scale_0\nproperty float scale_1\nproperty float scale_2\n");
    fprintf(f, "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n");
    fprintf(f, "end_header\n");

    const float zeroNormal[3] = { 0.f, 0.f, 0.f };
    for (size_t i = 0; i < N; i++) {
        if (mask[i] & kMaskBitDeleted) continue;
        const GaussianSplat& s = data.splats[i];
        fwrite(s.position,  sizeof(float),  3, f);
        fwrite(zeroNormal,  sizeof(float),  3, f);
        fwrite(s.f_dc,      sizeof(float),  3, f);
        fwrite(s.f_rest,    sizeof(float), 45, f);
        fwrite(&s.opacity,  sizeof(float),  1, f);
        fwrite(s.scale,     sizeof(float),  3, f);
        fwrite(s.rotation,  sizeof(float),  4, f);
    }
    fclose(f);
    return true;
}

MStatus GSSavePLYCmd::doIt(const MArgList& args) {
    MStatus st;
    MArgDatabase db(syntax(), args, &st);
    if (!st) return st;

    if (!db.isFlagSet("-f")) {
        displayError("gsSavePLY: -file is required.");
        return MS::kFailure;
    }
    MString path; db.getFlagArgument("-f", 0, path);

    GaussianNode* node = nullptr;
    if (db.isFlagSet("-n")) {
        MString name; db.getFlagArgument("-n", 0, name);
        node = findNodeByName(name);
        if (!node) {
            displayError(MString("gsSavePLY: no gaussianSplat named ") + name);
            return MS::kFailure;
        }
    } else {
        std::vector<GaussianNode*> nodes;
        collectAllGaussianNodes(nodes);
        for (GaussianNode* n : nodes) if (n->hasData()) { node = n; break; }
        if (!node) {
            displayError("gsSavePLY: no gaussianSplat with loaded data found.");
            return MS::kFailure;
        }
    }

    if (!node->areInputsReady() || !node->bufSelectionMask()) {
        displayError("gsSavePLY: data node has no GPU buffers yet (render once first).");
        return MS::kFailure;
    }

    // Refresh CPU mask shadow from GPU before save
    ID3D11Device* device = getDX11Device();
    if (!device) return MS::kFailure;
    ID3D11DeviceContext* ctx = getDX11Context(device);
    if (!ctx) return MS::kFailure;
    node->readbackMask(device, ctx);
    ctx->Release();

    const auto& mask = node->maskShadow();
    const auto& data = node->gaussianData();
    if (mask.size() != data.splats.size()) {
        displayError("gsSavePLY: mask / splat count mismatch.");
        return MS::kFailure;
    }

    size_t kept = 0;
    if (!writeBinaryPLY(path.asChar(), data, mask, kept)) {
        displayError(MString("gsSavePLY: failed to open for write: ") + path);
        return MS::kFailure;
    }

    displayInfo(MString("[gsSavePLY] Wrote ") + (unsigned)kept + " / " +
                (unsigned)data.splats.size() + " splats to " + path);
    setResult((int)kept);
    return MS::kSuccess;
}

// ===========================================================================
// GSMarqueeContext  —  VP2.0 (DX11) implementation
//
// In VP2.0 mode Maya ONLY calls the 3-arg overloads (doPress/doDrag/doRelease
// with MUIDrawManager).  The 1-arg legacy overloads are kept as fallback with
// diagnostic prints so we can detect if they are somehow called.
// drawFeedback() is called every repaint and draws the yellow marquee rect.
// ===========================================================================
const MString GSMarqueeContextCmd::commandName("gsMarqueeCtx");

MPxContext* GSMarqueeContextCmd::makeObj() {
    MGlobal::displayInfo("[GS CTX] makeObj() called — creating GSMarqueeContext");
    return new GSMarqueeContext;
}

GSMarqueeContext::GSMarqueeContext() {
    MGlobal::displayInfo("[GS CTX] Constructor called");
    setTitleString("Gaussian Marquee Select");
    setImage("aselect.png", MPxContext::kImage1);
}

void GSMarqueeContext::toolOnSetup(MEvent&) {
    MGlobal::displayInfo("[GS CTX] toolOnSetup — tool is now ACTIVE. Drag in viewport to select.");
    setHelpString("Drag to marquee-select Gaussian splats. Shift=add Ctrl=subtract Shift+Ctrl=toggle.");
    m_dragging = false;
}

void GSMarqueeContext::toolOffCleanup() {
    MGlobal::displayInfo("[GS CTX] toolOffCleanup — tool deactivated.");
    m_dragging = false;
}

// ---------------------------------------------------------------------------
// Helper: draw the yellow marquee rectangle via MUIDrawManager.
// Coordinates are viewport pixels (Y from bottom).
// ---------------------------------------------------------------------------
void GSMarqueeContext::drawRect(MHWRender::MUIDrawManager& dm) const {
    double cx = 0.5 * (m_x0 + m_x1);
    double cy = 0.5 * (m_y0 + m_y1);
    double hw = 0.5 * std::abs((double)(m_x1 - m_x0));
    double hh = 0.5 * std::abs((double)(m_y1 - m_y0));

    dm.beginDrawable();
    dm.setColor(MColor(1.f, 0.85f, 0.15f, 1.f));
    dm.setLineWidth(2.f);
    dm.rect2d(MPoint(cx, cy), MVector(0.0, 1.0, 0.0), hw, hh, false);
    dm.endDrawable();
}

// ---------------------------------------------------------------------------
// Helper: run the CPU selection pass and upload updated mask to GPU.
// vpW/vpH are viewport pixel dimensions.
// ---------------------------------------------------------------------------
void GSMarqueeContext::runSelectionFromRect(MEvent& event) {
    M3dView view = M3dView::active3dView();
    unsigned int vpW = view.portWidth();
    unsigned int vpH = view.portHeight();
    if (vpW == 0 || vpH == 0) {
        MGlobal::displayInfo("[GS CTX] runSelectionFromRect: viewport size is 0, aborting.");
        return;
    }

    int dx = std::abs((int)m_x1 - (int)m_x0);
    int dy = std::abs((int)m_y1 - (int)m_y0);

    MGlobal::displayInfo(MString("[GS CTX] doRelease: press=(") + m_x0 + "," + m_y0 +
                         ") release=(" + m_x1 + "," + m_y1 + ") delta=(" + dx + "," + dy +
                         ") viewport=" + (int)vpW + "x" + (int)vpH);

    if (dx < 2 && dy < 2) {
        MGlobal::displayInfo("[GS CTX] Plain click (< 2px drag) — clearing selection.");
        if (!(event.isModifierShift() || event.isModifierControl()))
            MGlobal::executeCommand("gsClearSelection");
        return;
    }

    // Screen pixels (Y=0 at bottom) -> NDC [-1, 1]
    float sxMin = (float)std::min(m_x0, m_x1);
    float sxMax = (float)std::max(m_x0, m_x1);
    float syMin = (float)std::min(m_y0, m_y1);
    float syMax = (float)std::max(m_y0, m_y1);

    float nxMin =  sxMin / (float)vpW * 2.f - 1.f;
    float nxMax =  sxMax / (float)vpW * 2.f - 1.f;
    float nyMin =  syMin / (float)vpH * 2.f - 1.f;
    float nyMax =  syMax / (float)vpH * 2.f - 1.f;

    int mode = 0;
    if      (event.isModifierShift() && event.isModifierControl()) mode = 3;
    else if (event.isModifierShift())                               mode = 1;
    else if (event.isModifierControl())                             mode = 2;

    MGlobal::displayInfo(MString("[GS CTX] NDC rect: [") + nxMin + "," + nyMin +
                         "] -> [" + nxMax + "," + nyMax + "]  mode=" + mode);

    ID3D11Device*        device = getDX11Device();
    ID3D11DeviceContext* ctx    = getDX11Context(device);
    if (!device || !ctx) {
        MGlobal::displayInfo("[GS CTX] ERROR: no DX11 device/context.");
        return;
    }

    auto& mgr = GaussianRenderManager::instance();
    MGlobal::displayInfo(MString("[GS CTX] RenderManager vpW=") + mgr.viewportWidth() +
                         " vpH=" + mgr.viewportHeight());

    if (mgr.viewportWidth() <= 0.f) {
        MGlobal::displayInfo("[GS CTX] WARNING: RenderManager has no viewport data yet — did a frame render?");
        if (ctx) ctx->Release();
        return;
    }

    float viewProj[16];
    matMul4x4(mgr.viewMatrix(), mgr.projMatrix(), viewProj);

    std::vector<RenderPair> pairs;
    collectRenderPairs(pairs);
    MGlobal::displayInfo(MString("[GS CTX] Found ") + (int)pairs.size() + " render pair(s) in scene.");

    int nSuccess = 0;
    for (const auto& p : pairs) {
        if (!p.node) { MGlobal::displayInfo("[GS CTX]   pair: null node, skip"); continue; }
        if (!p.node->areInputsReady()) {
            MGlobal::displayInfo("[GS CTX]   pair: inputs not ready, skip");
            continue;
        }
        MGlobal::displayInfo(MString("[GS CTX]   running selection on node with ") +
                             (int)p.node->splatCount() + " splats");
        float worldMat[16];
        mmatrixToFloat16(p.dagPath.inclusiveMatrix(), worldMat);
        if (mgr.runSelection(device, ctx, p.node, worldMat, viewProj,
                             nxMin, nyMin, nxMax, nyMax, mode))
            nSuccess++;
    }

    MGlobal::displayInfo(MString("[GS CTX] Selection done. Ran on ") + nSuccess + " node(s).");

    if (ctx) ctx->Release();
    M3dView::active3dView().refresh(false, false);
}

// ---------------------------------------------------------------------------
// VP2.0 doPress  —  called by Maya when mouse is pressed in VP2.0 viewport.
// ---------------------------------------------------------------------------
MStatus GSMarqueeContext::doPress(MEvent& event,
                                   MHWRender::MUIDrawManager& /*dm*/,
                                   const MHWRender::MFrameContext& /*fc*/) {
    event.getPosition(m_x0, m_y0);
    m_x1 = m_x0;
    m_y1 = m_y0;
    m_dragging = true;
    MGlobal::displayInfo(MString("[GS CTX] doPress (VP2.0) at pixel (") + m_x0 + "," + m_y0 + ")");
    return MS::kSuccess;
}

// ---------------------------------------------------------------------------
// VP2.0 doDrag  —  called on each mouse-move while button is held.
// Updates stored position; drawFeedback() will paint the rect.
// ---------------------------------------------------------------------------
MStatus GSMarqueeContext::doDrag(MEvent& event,
                                  MHWRender::MUIDrawManager& /*dm*/,
                                  const MHWRender::MFrameContext& /*fc*/) {
    event.getPosition(m_x1, m_y1);
    // No print here (called hundreds of times per second while dragging)
    return MS::kSuccess;
}

// ---------------------------------------------------------------------------
// VP2.0 doRelease  —  runs the actual selection logic.
// ---------------------------------------------------------------------------
MStatus GSMarqueeContext::doRelease(MEvent& event,
                                     MHWRender::MUIDrawManager& /*dm*/,
                                     const MHWRender::MFrameContext& /*fc*/) {
    event.getPosition(m_x1, m_y1);
    m_dragging = false;
    MGlobal::displayInfo(MString("[GS CTX] doRelease (VP2.0) at pixel (") + m_x1 + "," + m_y1 + ")");
    runSelectionFromRect(event);
    return MS::kSuccess;
}

// ---------------------------------------------------------------------------
// drawFeedback  —  called every repaint while tool is active; draws the rect.
// ---------------------------------------------------------------------------
MStatus GSMarqueeContext::drawFeedback(MHWRender::MUIDrawManager& dm,
                                        const MHWRender::MFrameContext& /*fc*/) {
    if (m_dragging) drawRect(dm);
    return MS::kSuccess;
}

// ---------------------------------------------------------------------------
// Legacy 1-arg overloads  —  kept as safety net with diagnostic prints.
// These should NOT be called in VP2.0/DX11 mode; if you see these messages
// it means Maya is using a legacy renderer.
// ---------------------------------------------------------------------------
MStatus GSMarqueeContext::doPress(MEvent& event) {
    event.getPosition(m_x0, m_y0);
    m_x1 = m_x0; m_y1 = m_y0;
    m_dragging = true;
    MGlobal::displayInfo(MString("[GS CTX] doPress (LEGACY 1-arg) at (") + m_x0 + "," + m_y0 + ")");
    return MS::kSuccess;
}

MStatus GSMarqueeContext::doDrag(MEvent& event) {
    event.getPosition(m_x1, m_y1);
    MGlobal::displayInfo("[GS CTX] doDrag (LEGACY 1-arg)");
    M3dView::active3dView().refresh(false, false);
    return MS::kSuccess;
}

MStatus GSMarqueeContext::doRelease(MEvent& event) {
    event.getPosition(m_x1, m_y1);
    m_dragging = false;
    MGlobal::displayInfo(MString("[GS CTX] doRelease (LEGACY 1-arg) at (") + m_x1 + "," + m_y1 + ")");
    runSelectionFromRect(event);
    return MS::kSuccess;
}
