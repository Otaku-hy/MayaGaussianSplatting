#include "GaussianNode.h"
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnData.h>

// ---------------------------------------------------------------------------
// Static member definitions
// ---------------------------------------------------------------------------
MTypeId GaussianNode::typeId               { 0x00127A00 };
MString GaussianNode::typeName             { "gaussianSplat" };
MString GaussianNode::drawDbClassification { "drawdb/geometry/gaussianSplat" };
MString GaussianNode::drawRegistrantId     { "gaussianSplatPlugin" };

MObject GaussianNode::aFilePath;
MObject GaussianNode::aPointSize;

// ---------------------------------------------------------------------------
void* GaussianNode::creator() { return new GaussianNode(); }

MStatus GaussianNode::initialize() {
    MFnTypedAttribute   tAttr;
    MFnNumericAttribute nAttr;

    // filePath  ---------------------------------------------------------------
    aFilePath = tAttr.create("filePath", "fp", MFnData::kString);
    tAttr.setUsedAsFilename(true);
    tAttr.setStorable(true);
    CHECK_MSTATUS_AND_RETURN_IT(addAttribute(aFilePath));

    // pointSize  --------------------------------------------------------------
    aPointSize = nAttr.create("pointSize", "ps", MFnNumericData::kFloat, 4.0f);
    nAttr.setMin(0.5f);
    nAttr.setMax(64.0f);
    nAttr.setStorable(true);
    nAttr.setKeyable(true);
    CHECK_MSTATUS_AND_RETURN_IT(addAttribute(aPointSize));

    return MS::kSuccess;
}
