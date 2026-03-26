#pragma once
#include <maya/MPxLocatorNode.h>
#include <maya/MString.h>
#include <maya/MTypeId.h>
#include <maya/MObject.h>

class GaussianDataNode;

// ---------------------------------------------------------------------------
// GaussianNode  --  MPxLocatorNode that renders Gaussian splats.
//
// This node no longer loads PLY data itself. Instead, it connects to a
// GaussianDataNode via the inputData message attribute.
//
// Attributes:
//   inputData  (message) -- connect from GaussianDataNode.outputData
//   pointSize  (float)   -- debug display point radius in pixels
// ---------------------------------------------------------------------------
class GaussianNode : public MPxLocatorNode {
public:
    static void*   creator();
    static MStatus initialize();

    MStatus compute(const MPlug& plug, MDataBlock& dataBlock) override;

    bool isBounded() const override { return false; }

    // Static identifiers
    static MTypeId typeId;
    static MString typeName;
    static MString drawDbClassification;
    static MString drawRegistrantId;

    // Maya attributes
    static MObject aInputData;   // message attribute, connected from data node
    static MObject aPointSize;
    static MObject aRenderMode;  // 0=auto, 1=debug, 2=production

    // Find the upstream GaussianDataNode via the inputData connection
    GaussianDataNode* findConnectedDataNode() const;

private:
    friend class GaussianDrawOverride;
};
