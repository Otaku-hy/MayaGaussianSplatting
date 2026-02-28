#pragma once
#include <maya/MPxLocatorNode.h>
#include <maya/MString.h>
#include <maya/MTypeId.h>
#include <maya/MObject.h>
#include "GaussianData.h"

// ---------------------------------------------------------------------------
// GaussianNode  —  MPxLocatorNode that holds a loaded .ply file.
//
// Attributes exposed in Maya:
//   filePath  (string)  –  path to the .ply Gaussian splatting file
//   pointSize (float)   –  debug display point radius in pixels
//
// GaussianDrawOverride accesses m_data directly (declared friend).
// ---------------------------------------------------------------------------
class GaussianNode : public MPxLocatorNode {
public:
    static void*   creator();
    static MStatus initialize();

    // Locator does not have a meaningful bounding box for now
    bool isBounded() const override { return false; }

    // Static identifiers
    static MTypeId typeId;
    static MString typeName;
    static MString drawDbClassification;
    static MString drawRegistrantId;

    // Maya attributes
    static MObject aFilePath;
    static MObject aPointSize;

    // Read-only access for the geometry override
    const GaussianData& gaussianData()  const { return m_data; }
    const MString&      loadedPath()    const { return m_loadedPath; }

private:
    GaussianData m_data;
    MString      m_loadedPath;   // last successfully attempted path

    friend class GaussianDrawOverride;
};
