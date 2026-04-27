#include <maya/MFnPlugin.h>
#include <maya/MGlobal.h>
#include <maya/MDrawRegistry.h>
#include "GaussianNode.h"
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
             -annotation "Create a gaussianSplat node and load a PLY file into it"
             -command "gaussianSplat_loadPLY";

    menuItem -label "Create Gaussian Splat Node"
             -annotation "Create an empty gaussianSplat node (set filePath in AE)"
             -command "gaussianSplat_createNode";

    menuItem -divider true -dividerLabel "Selection / Editing";

    menuItem -label "Marquee Select Tool"
             -annotation "Drag a rectangle in the viewport to select splats (scope: selected node or all)"
             -command "gaussianSplat_activateMarquee";

    menuItem -label "Clear Selection"
             -annotation "Unselect all splats on all nodes"
             -command "gsClearSelection";

    menuItem -label "Delete Selected (soft)"
             -annotation "Soft-delete selected splats (hidden, restorable)"
             -command "gsDeleteSelected";

    menuItem -label "Restore All"
             -annotation "Clear both selection and deleted bits on all nodes"
             -command "gsRestoreAll";

    menuItem -divider true;

    menuItem -label "Save PLY As..."
             -annotation "Save the (non-deleted) splats of the selected node to a PLY file"
             -command "gaussianSplat_savePLY";
}

global proc gaussianSplat_activateMarquee()
{
    // Always destroy the old instance so that after plugin reload the new
    // DLL's makeObj() is called (stale context objects survive unloadPlugin).
    if (`contextInfo -exists gsMarqueeCtx1`)
        deleteUI gsMarqueeCtx1;
    gsMarqueeCtx gsMarqueeCtx1;
    setToolTo gsMarqueeCtx1;
    print "// Gaussian Marquee tool active. Drag in the viewport to select splats.\n";
    print "// Tip: select a gaussianSplat node first to scope selection to that cloud only.\n";
}

// ---------------------------------------------------------------------------
// Attribute Editor template for gaussianSplat.
// Shows filePath + per-node editing buttons (Restore All, Delete Selected,
// Save PLY As).
// ---------------------------------------------------------------------------
global proc AEgaussianSplatTemplate(string $nodeName)
{
    editorTemplate -beginScrollLayout;

        editorTemplate -beginLayout "Point Cloud Data" -collapse 0;
            editorTemplate -addControl "filePath";
        editorTemplate -endLayout;

        editorTemplate -beginLayout "Display" -collapse 0;
            editorTemplate -addControl "pointSize";
            editorTemplate -addControl "renderMode";
        editorTemplate -endLayout;

        editorTemplate -beginLayout "Selection / Editing" -collapse 0;
            // Pass "" so Maya calls procs with "nodeName." — we parse nodeName ourselves.
            editorTemplate -callCustom
                "AEgaussianSplat_editNew"
                "AEgaussianSplat_editReplace"
                "";
        editorTemplate -endLayout;

        editorTemplate -addExtraControls;
    editorTemplate -endScrollLayout;

    // Suppress attributes already shown above
    editorTemplate -suppress "dataReady";
}

// callCustom passes "nodeName." (trailing dot when attribute is empty).
// Parse the node name from that string.
global proc string AEgaussianSplat_nodeName(string $nodeAttr)
{
    string $parts[] = stringToStringArray($nodeAttr, ".");
    return $parts[0];
}

global proc AEgaussianSplat_editNew(string $nodeAttr)
{
    string $n = AEgaussianSplat_nodeName($nodeAttr);
    columnLayout -adj true gsEditCol;
        button -label "Restore All  (clear selection + deleted)"
               -annotation "Reveal all hidden splats on this node"
               -command ("gsRestoreAll -node " + $n)
               gsRestoreBtn;
        button -label "Delete Selected  (soft-hide)"
               -annotation "Hide the currently selected splats (restorable)"
               -command "gsDeleteSelected"
               gsDeleteBtn;
        separator -height 8;
        button -label "Save PLY As..."
               -annotation "Export non-deleted splats to a PLY file"
               -command ("AEgaussianSplat_saveCB " + $n)
               gsSaveBtn;
    setParent ..;
}

global proc AEgaussianSplat_editReplace(string $nodeAttr)
{
    if (!`button -exists gsRestoreBtn`) { AEgaussianSplat_editNew($nodeAttr); return; }
    string $n = AEgaussianSplat_nodeName($nodeAttr);
    button -e -command ("gsRestoreAll -node " + $n)       gsRestoreBtn;
    button -e -command  "gsDeleteSelected"                 gsDeleteBtn;
    button -e -command ("AEgaussianSplat_saveCB " + $n)   gsSaveBtn;
}

global proc AEgaussianSplat_saveCB(string $nodeName)
{
    string $files[] = `fileDialog2 -fileMode 0
                                   -caption "Save Gaussian Splatting PLY"
                                   -fileFilter "PLY Files (*.ply);;All Files (*.*)"`;
    if (size($files) == 0) return;
    gsSavePLY -node $nodeName -file $files[0];
}

global proc gaussianSplat_savePLY()
{
    // Menu version: prefer selected gaussianSplat node, otherwise first in scene.
    string $shapes[] = `ls -selection -dag -type gaussianSplat`;
    string $node = "";
    if (size($shapes) > 0) {
        $node = $shapes[0];
    } else {
        string $all[] = `ls -type gaussianSplat`;
        if (size($all) == 0) { warning "No gaussianSplat in scene."; return; }
        $node = $all[0];
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

    // Each load creates an independent gaussianSplat node with its own data.
    string $transform = `createNode transform -name "gaussianSplat1"`;
    string $node      = `createNode gaussianSplat -parent $transform`;
    setAttr -type "string" ($node + ".filePath") $files[0];

    select -r $transform;
    print ("// Created: " + $node + " (filePath=" + $files[0] + ")\n");
}

global proc gaussianSplat_createNode()
{
    string $transform = `createNode transform -name "gaussianSplat1"`;
    string $node      = `createNode gaussianSplat -parent $transform`;
    select -r $transform;
    print ("// Created: " + $node + " -- set filePath in the Attribute Editor\n");
}
)MEL";

// ---------------------------------------------------------------------------
EXPORT MStatus initializePlugin(MObject obj) {
    MFnPlugin plugin(obj, "CIS6600 Team", "0.3", "Any");

    // Register the self-contained gaussianSplat locator node
    MStatus status = plugin.registerNode(
        GaussianNode::typeName,
        GaussianNode::typeId,
        GaussianNode::creator,
        GaussianNode::initialize,
        MPxNode::kLocatorNode,
        &GaussianNode::drawDbClassification);

    if (!status) {
        MGlobal::displayError(
            "[GaussianSplat] registerNode(gaussianSplat) failed: " + status.errorString());
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

    MGlobal::displayInfo("[GaussianSplat] Plugin unloaded.");
    return MS::kSuccess;
}
