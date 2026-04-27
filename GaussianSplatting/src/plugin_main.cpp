#include <maya/MFnPlugin.h>
#include <maya/MGlobal.h>
#include <maya/MDrawRegistry.h>
#include "GaussianNode.h"
#include "GaussianDataNode.h"
#include "GaussianDrawOverride.h"
#include "GaussianRenderManager.h"
#include "GaussianSelection.h"

#define EXPORT __declspec(dllexport)

// MEL command to build/remove the menu
static const char* kBuildMenuMel = R"MEL(
global proc gaussianSplat_buildMenu()
{
    global string $gMainWindow;
    if (`menu -exists gaussianSplatMenu`)
        deleteUI gaussianSplatMenu;

    menu -parent $gMainWindow -label "Gaussian Splatting" -tearOff true gaussianSplatMenu;

    menuItem -label "Load PLY File..."
             -annotation "Create a data node and render node from a PLY file"
             -command "gaussianSplat_loadPLY";

    menuItem -divider true;

    menuItem -label "Create Data Node"
             -annotation "Create an empty gaussianSplatData node"
             -command "gaussianSplat_createDataNode";

    menuItem -label "Create Render Node"
             -annotation "Create a gaussianSplat render node"
             -command "gaussianSplat_createRenderNode";

    menuItem -divider true;

    menuItem -label "Connect Selected (Data -> Render)"
             -annotation "Connect first selected data node to second selected render node"
             -command "gaussianSplat_connectSelected";

    menuItem -divider true -dividerLabel "Selection / Editing";

    menuItem -label "Marquee Select Tool"
             -annotation "Activate the Gaussian Splat marquee select tool (drag a rectangle in the viewport)"
             -command "gaussianSplat_activateMarquee";

    menuItem -label "Clear Selection"
             -annotation "Unselect all splats on every gaussianSplatData node"
             -command "gsClearSelection";

    menuItem -label "Delete Selected (soft)"
             -annotation "Soft-delete currently selected splats (hidden until restored or saved out)"
             -command "gsDeleteSelected";

    menuItem -label "Restore All"
             -annotation "Clear both selection and deleted bits on every data node"
             -command "gsRestoreAll";

    menuItem -divider true;

    menuItem -label "Save PLY As..."
             -annotation "Save the (mask-filtered) data node to a binary PLY file"
             -command "gaussianSplat_savePLY";
}

global proc gaussianSplat_activateMarquee()
{
    // Always destroy the old instance first so that after a plugin reload the
    // new DLL's makeObj() is called (stale context objects survive unloadPlugin).
    if (`contextInfo -exists gsMarqueeCtx1`)
        deleteUI gsMarqueeCtx1;
    gsMarqueeCtx gsMarqueeCtx1;
    setToolTo gsMarqueeCtx1;
    print "// Gaussian Marquee tool active. Drag in the viewport.\n";
}

// ---------------------------------------------------------------------------
// Attribute Editor template for gaussianSplatData — adds a Restore All button.
// ---------------------------------------------------------------------------
global proc AEgaussianSplatDataTemplate(string $nodeName)
{
    editorTemplate -beginScrollLayout;
        editorTemplate -beginLayout "Data" -collapse 0;
            editorTemplate -addControl "filePath";
            editorTemplate -addSeparator;
            editorTemplate -callCustom
                "AEgaussianSplatData_restoreNew"
                "AEgaussianSplatData_restoreReplace"
                "";
        editorTemplate -endLayout;
        editorTemplate -addExtraControls;
    editorTemplate -endScrollLayout;
}

global proc AEgaussianSplatData_restoreNew(string $dummy)
{
    rowLayout -nc 2 -cw2 160 220 -cl2 "left" "left" gsRestoreRow;
        text -label "Selection / Deletion:";
        button -label "Restore All (clear mask)"
               -annotation "Clear both selected and deleted bits; all splats visible again"
               -command "AEgaussianSplatData_restoreCB"
               gsRestoreBtn;
    setParent ..;
}

global proc AEgaussianSplatData_restoreReplace(string $dummy)
{
    // No per-node plug to refresh -- the button is stateless.
    if (!`button -exists gsRestoreBtn`) AEgaussianSplatData_restoreNew($dummy);
}

global proc AEgaussianSplatData_restoreCB()
{
    // Target the node that owns this AE -- use the current selection
    // restricted to gaussianSplatData.
    string $sel[] = `ls -selection -type gaussianSplatData`;
    if (size($sel) == 0) {
        // Fall back to all data nodes
        gsRestoreAll;
        return;
    }
    for ($n in $sel) gsRestoreAll -node $n;
}

global proc gaussianSplat_savePLY()
{
    string $nodes[] = `ls -type gaussianSplatData`;
    if (size($nodes) == 0) {
        warning "No gaussianSplatData in scene.";
        return;
    }
    string $node = $nodes[0];
    if (size($nodes) > 1) {
        string $sel[] = `ls -selection -type gaussianSplatData`;
        if (size($sel) > 0) $node = $sel[0];
    }

    string $files[] = `fileDialog2 -fileMode 0
                                   -caption "Save Gaussian Splatting PLY"
                                   -fileFilter "PLY Files (*.ply);;All Files (*.*)"`;
    if (size($files) == 0) return;
    gsSavePLY -node $node -file $files[0];
}

global proc gaussianSplat_removeMenu()
{
    if (`menu -exists gaussianSplatMenu`)
        deleteUI gaussianSplatMenu;
}

global proc gaussianSplat_loadPLY()
{
    string $files[] = `fileDialog2 -fileMode 1
                                   -caption "Load Gaussian Splatting PLY"
                                   -fileFilter "PLY Files (*.ply);;All Files (*.*)"`;
    if (size($files) == 0) return;

    string $dataNode = `createNode gaussianSplatData`;
    setAttr -type "string" ($dataNode + ".filePath") $files[0];

    string $transform = `createNode transform -name "gaussianSplat1"`;
    string $renderNode = `createNode gaussianSplat -parent $transform`;

    connectAttr ($dataNode + ".outputData") ($renderNode + ".inputData");

    select -r $transform;
    print ("// Created: " + $dataNode + " -> " + $renderNode + "\n");
}

global proc gaussianSplat_createDataNode()
{
    string $node = `createNode gaussianSplatData`;
    select -r $node;
    print ("// Created data node: " + $node + "\n");
}

global proc gaussianSplat_createRenderNode()
{
    string $transform = `createNode transform -name "gaussianSplat1"`;
    string $node = `createNode gaussianSplat -parent $transform`;
    select -r $transform;
    print ("// Created render node: " + $node + " (under " + $transform + ")\n");
}

global proc gaussianSplat_connectSelected()
{
    string $sel[] = `ls -selection`;
    if (size($sel) < 2) {
        warning "Select a gaussianSplatData node first, then a gaussianSplat node (or its transform).";
        return;
    }

    string $dataNode = $sel[0];
    string $renderTarget = $sel[1];

    // If the second selection is a transform, find the shape underneath
    string $renderNode = $renderTarget;
    if (`nodeType $renderTarget` == "transform") {
        string $shapes[] = `listRelatives -shapes -type "gaussianSplat" $renderTarget`;
        if (size($shapes) == 0) {
            warning ("No gaussianSplat shape found under " + $renderTarget);
            return;
        }
        $renderNode = $shapes[0];
    }

    if (`nodeType $dataNode` != "gaussianSplatData") {
        warning ($dataNode + " is not a gaussianSplatData node.");
        return;
    }
    if (`nodeType $renderNode` != "gaussianSplat") {
        warning ($renderNode + " is not a gaussianSplat node.");
        return;
    }

    connectAttr -force ($dataNode + ".outputData") ($renderNode + ".inputData");
    print ("// Connected: " + $dataNode + " -> " + $renderNode + "\n");
}
)MEL";

// ---------------------------------------------------------------------------
EXPORT MStatus initializePlugin(MObject obj) {
    MFnPlugin plugin(obj, "CIS6600 Team", "0.2", "Any");

    // Register the data node (non-locator DG node)
    MStatus status = plugin.registerNode(
        GaussianDataNode::typeName,
        GaussianDataNode::typeId,
        GaussianDataNode::creator,
        GaussianDataNode::initialize,
        MPxNode::kDependNode);

    if (!status) {
        MGlobal::displayError(
            "[GaussianSplat] registerNode(gaussianSplatData) failed: " + status.errorString());
        return status;
    }

    // Register the locator render node
    status = plugin.registerNode(
        GaussianNode::typeName,
        GaussianNode::typeId,
        GaussianNode::creator,
        GaussianNode::initialize,
        MPxNode::kLocatorNode,
        &GaussianNode::drawDbClassification);

    if (!status) {
        MGlobal::displayError(
            "[GaussianSplat] registerNode(gaussianSplat) failed: " + status.errorString());
        plugin.deregisterNode(GaussianDataNode::typeId);
        return status;
    }

    // Register the draw override
    status = MHWRender::MDrawRegistry::registerDrawOverrideCreator(
        GaussianNode::drawDbClassification,
        GaussianNode::drawRegistrantId,
        GaussianDrawOverride::creator);

    if (!status) {
        MGlobal::displayError(
            "[GaussianSplat] registerDrawOverrideCreator failed: " + status.errorString());
        plugin.deregisterNode(GaussianNode::typeId);
        plugin.deregisterNode(GaussianDataNode::typeId);
        return status;
    }

    // --- Selection / editing commands + marquee context ---
    plugin.registerCommand(GSMarqueeSelectCmd::commandName,
                           GSMarqueeSelectCmd::creator,
                           GSMarqueeSelectCmd::newSyntax);
    plugin.registerCommand(GSClearSelectionCmd::commandName,
                           GSClearSelectionCmd::creator);
    plugin.registerCommand(GSDeleteSelectedCmd::commandName,
                           GSDeleteSelectedCmd::creator);
    plugin.registerCommand(GSRestoreAllCmd::commandName,
                           GSRestoreAllCmd::creator,
                           GSRestoreAllCmd::newSyntax);
    plugin.registerCommand(GSSavePLYCmd::commandName,
                           GSSavePLYCmd::creator,
                           GSSavePLYCmd::newSyntax);
    plugin.registerContextCommand(GSMarqueeContextCmd::commandName,
                                   GSMarqueeContextCmd::creator);

    // Build menu via MEL
    MGlobal::executeCommand(kBuildMenuMel);
    MGlobal::executeCommand("gaussianSplat_buildMenu");

    MGlobal::displayInfo("[GaussianSplat] Plugin loaded (v0.2, data/render node split).");
    return MS::kSuccess;
}

// ---------------------------------------------------------------------------
EXPORT MStatus uninitializePlugin(MObject obj) {
    MFnPlugin plugin(obj);

    // Remove menu
    MGlobal::executeCommand("gaussianSplat_removeMenu");

    // Release merged render manager resources before deregistering nodes
    GaussianRenderManager::instance().releaseAll();

    plugin.deregisterContextCommand(GSMarqueeContextCmd::commandName);
    plugin.deregisterCommand(GSSavePLYCmd::commandName);
    plugin.deregisterCommand(GSRestoreAllCmd::commandName);
    plugin.deregisterCommand(GSDeleteSelectedCmd::commandName);
    plugin.deregisterCommand(GSClearSelectionCmd::commandName);
    plugin.deregisterCommand(GSMarqueeSelectCmd::commandName);

    MHWRender::MDrawRegistry::deregisterDrawOverrideCreator(
        GaussianNode::drawDbClassification,
        GaussianNode::drawRegistrantId);

    plugin.deregisterNode(GaussianNode::typeId);
    plugin.deregisterNode(GaussianDataNode::typeId);

    MGlobal::displayInfo("[GaussianSplat] Plugin unloaded.");
    return MS::kSuccess;
}
