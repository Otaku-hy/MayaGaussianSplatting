#include "GaussianNode.h"
#include "PLYReader.h"
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnStringData.h>
#include <maya/MFnData.h>
#include <maya/MPlug.h>
#include <maya/MGlobal.h>

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

    // filePath — must supply MFnStringData default so it appears in AE
    MFnStringData stringDataFn;
    MObject defaultStr = stringDataFn.create("");
    aFilePath = tAttr.create("filePath", "fp", MFnData::kString, defaultStr);
    tAttr.setStorable(true);
    tAttr.setUsedAsFilename(true);
    CHECK_MSTATUS_AND_RETURN_IT(addAttribute(aFilePath));

    // pointSize
    aPointSize = nAttr.create("pointSize", "ps", MFnNumericData::kFloat, 10.0f);
    nAttr.setMin(1.0f);
    nAttr.setMax(200.0f);
    nAttr.setStorable(true);
    nAttr.setKeyable(true);
    CHECK_MSTATUS_AND_RETURN_IT(addAttribute(aPointSize));

    return MS::kSuccess;
}

// ---------------------------------------------------------------------------
// reloadIfNeeded  --  compare current filePath attr with m_loadedPath
// ---------------------------------------------------------------------------
void GaussianNode::reloadIfNeeded() {
    MPlug fpPlug(thisMObject(), aFilePath);
    MString currentPath = fpPlug.asString();

    if (currentPath == m_loadedPath) return;

    m_data.clear();
    m_loadedPath = currentPath;

    if (currentPath.length() == 0) return;

    std::string err;
    if (PLYReader::read(currentPath.asChar(), m_data, err)) {
        MGlobal::displayInfo(MString("[GaussianSplat] Loaded ") +
                             (unsigned int)m_data.count() + " splats from " + currentPath);
    } else {
        MGlobal::displayError(MString("[GaussianSplat] PLY load failed: ") + err.c_str());
        m_loadedPath = "";
    }
}
