#include <maya/MFnPlugin.h>
#include <maya/MGlobal.h>
#include <maya/MDrawRegistry.h>
#include "GaussianNode.h"
#include "GaussianDrawOverride.h"

#define EXPORT __declspec(dllexport)

// ---------------------------------------------------------------------------
EXPORT MStatus initializePlugin(MObject obj) {
    MFnPlugin plugin(obj, "CIS6600", "0.1", "Any");

    MStatus status = plugin.registerNode(
        GaussianNode::typeName,
        GaussianNode::typeId,
        GaussianNode::creator,
        GaussianNode::initialize,
        MPxNode::kLocatorNode,
        &GaussianNode::drawDbClassification);

    if (!status) {
        MGlobal::displayError("[GaussianSplat] registerNode failed: " + status.errorString());
        return status;
    }

    status = MHWRender::MDrawRegistry::registerDrawOverrideCreator(
        GaussianNode::drawDbClassification,
        GaussianNode::drawRegistrantId,
        GaussianDrawOverride::creator);

    if (!status) {
        MGlobal::displayError("[GaussianSplat] registerDrawOverrideCreator failed: " + status.errorString());
        plugin.deregisterNode(GaussianNode::typeId);
        return status;
    }

    MGlobal::displayInfo("[GaussianSplat] Plugin loaded.");
    return MS::kSuccess;
}

// ---------------------------------------------------------------------------
EXPORT MStatus uninitializePlugin(MObject obj) {
    MFnPlugin plugin(obj);

    MHWRender::MDrawRegistry::deregisterDrawOverrideCreator(
        GaussianNode::drawDbClassification,
        GaussianNode::drawRegistrantId);

    plugin.deregisterNode(GaussianNode::typeId);

    MGlobal::displayInfo("[GaussianSplat] Plugin unloaded.");
    return MS::kSuccess;
}
