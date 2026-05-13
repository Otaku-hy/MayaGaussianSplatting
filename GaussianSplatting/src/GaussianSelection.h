#pragma once

#include <maya/MPxCommand.h>
#include <maya/MPxContext.h>
#include <maya/MPxContextCommand.h>
#include <maya/MString.h>
#include <maya/MSyntax.h>
#include <maya/MEvent.h>
#include <maya/MUIDrawManager.h>
#include <maya/MFrameContext.h>

class GSMarqueeSelectCmd : public MPxCommand {
public:
    MStatus doIt(const MArgList& args) override;
    bool    isUndoable() const override { return false; }
    static void*    creator()   { return new GSMarqueeSelectCmd; }
    static MSyntax  newSyntax();
    static const MString commandName;
};

class GSClearSelectionCmd : public MPxCommand {
public:
    MStatus doIt(const MArgList& args) override;
    bool    isUndoable() const override { return false; }
    static void* creator() { return new GSClearSelectionCmd; }
    static const MString commandName;
};

class GSDeleteSelectedCmd : public MPxCommand {
public:
    MStatus doIt(const MArgList& args) override;
    bool    isUndoable() const override { return false; }
    static void* creator() { return new GSDeleteSelectedCmd; }
    static const MString commandName;
};

class GSRestoreAllCmd : public MPxCommand {
public:
    MStatus doIt(const MArgList& args) override;
    bool    isUndoable() const override { return false; }
    static void*    creator()   { return new GSRestoreAllCmd; }
    static MSyntax  newSyntax();
    static const MString commandName;
};

class GSSavePLYCmd : public MPxCommand {
public:
    MStatus doIt(const MArgList& args) override;
    bool    isUndoable() const override { return false; }
    static void*    creator()   { return new GSSavePLYCmd; }
    static MSyntax  newSyntax();
    static const MString commandName;
};

// ---------------------------------------------------------------------------
// Marquee context.
//
// In Maya VP2.0 (DX11), ONLY the 3-arg versions of doPress/doDrag/doRelease
// are called by Maya's event system.  The legacy 1-arg versions are never
// invoked when VP2.0 is active.  All input handling lives in the 3-arg
// overrides.  drawFeedback() redraws the yellow rect every viewport frame.
// ---------------------------------------------------------------------------
class GSMarqueeContext : public MPxContext {
public:
    GSMarqueeContext();

    void toolOnSetup   (MEvent& event) override;
    void toolOffCleanup()              override;

    // --- VP2.0 input+draw callbacks (the only ones called in DX11/VP2.0) ---
    MStatus doPress  (MEvent& event,
                      MHWRender::MUIDrawManager&       dm,
                      const MHWRender::MFrameContext&  fc) override;

    MStatus doDrag   (MEvent& event,
                      MHWRender::MUIDrawManager&       dm,
                      const MHWRender::MFrameContext&  fc) override;

    MStatus doRelease(MEvent& event,
                      MHWRender::MUIDrawManager&       dm,
                      const MHWRender::MFrameContext&  fc) override;

    // Called every repaint while tool is active — draws yellow rect overlay.
    MStatus drawFeedback(MHWRender::MUIDrawManager&       dm,
                         const MHWRender::MFrameContext&  fc) override;

    // --- Legacy 1-arg versions (only reached in non-VP2.0; kept for safety) ---
    MStatus doPress  (MEvent& event) override;
    MStatus doDrag   (MEvent& event) override;
    MStatus doRelease(MEvent& event) override;

private:
    void runSelectionFromRect(MEvent& event);
    void drawRect(MHWRender::MUIDrawManager& dm) const;

    short m_x0 = 0, m_y0 = 0;
    short m_x1 = 0, m_y1 = 0;
    bool  m_dragging = false;
};

class GSMarqueeContextCmd : public MPxContextCommand {
public:
    MPxContext* makeObj() override;
    static void* creator() { return new GSMarqueeContextCmd; }
    static const MString commandName;
};
