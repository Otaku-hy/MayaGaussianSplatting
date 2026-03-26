#include "GaussianNode.h"
#include "GaussianDataNode.h"
#include <maya/MFnMessageAttribute.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnDependencyNode.h>
#include <maya/MPlugArray.h>
#include <maya/MGlobal.h>

// ---------------------------------------------------------------------------
// Static member definitions
// ---------------------------------------------------------------------------
MTypeId GaussianNode::typeId               { 0x00127A00 };
MString GaussianNode::typeName             { "gaussianSplat" };
MString GaussianNode::drawDbClassification { "drawdb/geometry/gaussianSplat" };
MString GaussianNode::drawRegistrantId     { "gaussianSplatPlugin" };

MObject GaussianNode::aInputData;
MObject GaussianNode::aPointSize;
MObject GaussianNode::aRenderMode;

// ---------------------------------------------------------------------------
void* GaussianNode::creator() { return new GaussianNode(); }

MStatus GaussianNode::initialize() {
    MFnMessageAttribute mAttr;
    MFnNumericAttribute nAttr;

    // inputData  --  message connection from GaussianDataNode
    aInputData = mAttr.create("inputData", "id");
    mAttr.setStorable(false);
    CHECK_MSTATUS_AND_RETURN_IT(addAttribute(aInputData));

    // pointSize
    aPointSize = nAttr.create("pointSize", "ps", MFnNumericData::kFloat, 4.0f);
    nAttr.setMin(0.5f);
    nAttr.setMax(64.0f);
    nAttr.setStorable(true);
    nAttr.setKeyable(true);
    CHECK_MSTATUS_AND_RETURN_IT(addAttribute(aPointSize));

    // renderMode  (0=auto, 1=debug, 2=production)
    aRenderMode = nAttr.create("renderMode", "rm", MFnNumericData::kInt, 0);
    nAttr.setMin(0);
    nAttr.setMax(3);
    nAttr.setStorable(true);
    nAttr.setKeyable(true);
    CHECK_MSTATUS_AND_RETURN_IT(addAttribute(aRenderMode));

    return MS::kSuccess;
}

// ---------------------------------------------------------------------------
// compute  --  nothing to compute; this node is a pure locator
// ---------------------------------------------------------------------------
MStatus GaussianNode::compute(const MPlug& /*plug*/, MDataBlock& /*dataBlock*/) {
    return MS::kUnknownParameter;
}

// ---------------------------------------------------------------------------
// findConnectedDataNode  --  walk the inputData plug to find the upstream node
// ---------------------------------------------------------------------------
MBoundingBox GaussianNode::boundingBox() const {
    GaussianDataNode* dn = findConnectedDataNode();
    if (!dn || !dn->hasData())
        return MBoundingBox(MPoint(-1, -1, -1), MPoint(1, 1, 1));
    return dn->boundingBox();
}

GaussianDataNode* GaussianNode::findConnectedDataNode() const {
    MPlug inputPlug(thisMObject(), aInputData);
    MPlugArray sources;
    inputPlug.connectedTo(sources, true /*asDst*/, false /*asSrc*/);

    if (sources.length() == 0) return nullptr;

    MFnDependencyNode fn(sources[0].node());
    return dynamic_cast<GaussianDataNode*>(fn.userNode());
}
