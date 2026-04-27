#include "GaussianDrawOverride.h"
#include "GaussianNode.h"
#include "GaussianDataNode.h"
#include "GaussianRenderManager.h"

#include <maya/MDagPath.h>
#include <maya/MFnDependencyNode.h>
#include <maya/MPlug.h>
#include <maya/MGlobal.h>
#include <maya/MDrawRegistry.h>
#include <maya/MFrameContext.h>
#include <maya/MDrawContext.h>
#include <maya/MViewport2Renderer.h>
#include <maya/MMatrix.h>

#include <d3d11.h>
#include <d3dcompiler.h>
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")

#include <algorithm>
#include <cstring>
#include <cmath>

// ===========================================================================
// Debug pipeline HLSL  (reads from shared StructuredBuffers, no sort)
// ===========================================================================
static const char* kDbgShaderSrc = R"HLSL(
StructuredBuffer<float3> gPositionWS : register(t0);
StructuredBuffer<float>  gOpacity    : register(t1);
StructuredBuffer<float3> gSHCoeffs   : register(t2);

cbuffer CBDebug : register(b0)
{
    row_major float4x4 gWVP;
    float  gPointSize;
    float  gVPWidth;
    float  gVPHeight;
    uint   gSHStride;    // = 16 (groups per splat), for indexing into gSHCoeffs
};

struct VS_OUT { float4 clip : SV_Position; float4 col : COLOR; };
struct GS_OUT { float4 clip : SV_Position; float4 col : COLOR; float2 uv : TEXCOORD0; };

VS_OUT VS(uint vid : SV_VertexID)
{
    float3 pos = gPositionWS[vid];
    // degree-0 SH color
    float3 col = gSHCoeffs[vid * gSHStride] * 0.282095f + 0.5f;
    col = max(col, 0.0f);
    // sigmoid opacity
    float alpha = 1.0f / (1.0f + exp(-gOpacity[vid]));

    VS_OUT o;
    o.clip = mul(float4(pos, 1.0f), gWVP);
    o.col  = float4(col, alpha);
    return o;
}

[maxvertexcount(4)]
void GS(point VS_OUT input[1], inout TriangleStream<GS_OUT> stream)
{
    float4 c = input[0].clip;
    float2 h = float2(gPointSize / gVPWidth, gPointSize / gVPHeight) * c.w;
    static const float2 corners[4] = {
        float2(-1.f,  1.f), float2( 1.f,  1.f),
        float2(-1.f, -1.f), float2( 1.f, -1.f),
    };
    GS_OUT o;
    o.col = input[0].col;
    [unroll]
    for (int k = 0; k < 4; k++) {
        o.clip = c + float4(corners[k] * h, 0.f, 0.f);
        o.uv   = corners[k];
        stream.Append(o);
    }
    stream.RestartStrip();
}

float4 PS(GS_OUT i) : SV_Target
{
    float r2 = dot(i.uv, i.uv);
    clip(1.0f - r2);
    float a = i.col.a * (1.0f - r2 * 0.4f);
    return float4(i.col.rgb, a);
}
)HLSL";

// ===========================================================================
// CB layout for debug path
// ===========================================================================
struct CBDebug {
    float wvp[16];
    float pointSize, vpWidth, vpHeight;
    uint32_t shStride;
};
static_assert(sizeof(CBDebug) % 16 == 0, "");

// ===========================================================================
// Utility: compile a shader stage
// ===========================================================================
static bool CompileStage(const char* src, size_t srcLen,
                         const char* entry, const char* target,
                         ID3DBlob** outBlob,
                         const D3D_SHADER_MACRO* defines = nullptr)
{
    UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;
#ifdef _DEBUG
    flags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif
    ID3DBlob* errBlob = nullptr;
    HRESULT hr = D3DCompile(src, srcLen, nullptr, defines, nullptr,
                            entry, target, flags, 0, outBlob, &errBlob);
    if (FAILED(hr)) {
        if (errBlob) {
            MGlobal::displayError(
                MString("[GaussianSplat] Shader compile error (") + entry + "): " +
                static_cast<const char*>(errBlob->GetBufferPointer()));
            errBlob->Release();
        }
        return false;
    }
    if (errBlob) errBlob->Release();
    return true;
}

// ===========================================================================
// GaussianDrawData
// ===========================================================================
GaussianDrawData::GaussianDrawData() : MUserData() {}
GaussianDrawData::~GaussianDrawData() { releaseAll(); }

#define SAFE_RELEASE(p) do { if (p) { (p)->Release(); (p) = nullptr; } } while(0)

void GaussianDrawData::releaseAll()
{
    releaseDebugResources();
    sharedSrvPositionWS = sharedSrvScale = sharedSrvRotation = nullptr;
    sharedSrvOpacity = sharedSrvSHCoeffs = nullptr;
    inputsReady = false;
}

void GaussianDrawData::releaseDebugResources()
{
    SAFE_RELEASE(dbgVS);
    SAFE_RELEASE(dbgGS);
    SAFE_RELEASE(dbgPS);
    SAFE_RELEASE(dbgCB);
    SAFE_RELEASE(blendState);
    SAFE_RELEASE(rsState);
    SAFE_RELEASE(dsState);
    dbgReady = false;
}

// ---------------------------------------------------------------------------
// initDebugPipeline
// ---------------------------------------------------------------------------
bool GaussianDrawData::initDebugPipeline(ID3D11Device* device)
{
    HRESULT hr;
    size_t srcLen = strlen(kDbgShaderSrc);

    ID3DBlob* vsBlob = nullptr, *gsBlob = nullptr, *psBlob = nullptr;
    if (!CompileStage(kDbgShaderSrc, srcLen, "VS", "vs_5_0", &vsBlob)) return false;
    if (!CompileStage(kDbgShaderSrc, srcLen, "GS", "gs_5_0", &gsBlob)) { vsBlob->Release(); return false; }
    if (!CompileStage(kDbgShaderSrc, srcLen, "PS", "ps_5_0", &psBlob)) { vsBlob->Release(); gsBlob->Release(); return false; }

    hr = device->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), nullptr, &dbgVS);
    if (FAILED(hr)) goto dbg_cleanup;
    hr = device->CreateGeometryShader(gsBlob->GetBufferPointer(), gsBlob->GetBufferSize(), nullptr, &dbgGS);
    if (FAILED(hr)) goto dbg_cleanup;
    hr = device->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(), nullptr, &dbgPS);
    if (FAILED(hr)) goto dbg_cleanup;

    {
        D3D11_BUFFER_DESC cbd = {};
        cbd.ByteWidth = sizeof(CBDebug); cbd.Usage = D3D11_USAGE_DYNAMIC;
        cbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER; cbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        hr = device->CreateBuffer(&cbd, nullptr, &dbgCB);
        if (FAILED(hr)) goto dbg_cleanup;
    }
    {
        D3D11_BLEND_DESC bd = {};
        bd.RenderTarget[0].BlendEnable           = TRUE;
        bd.RenderTarget[0].SrcBlend              = D3D11_BLEND_SRC_ALPHA;
        bd.RenderTarget[0].DestBlend             = D3D11_BLEND_INV_SRC_ALPHA;
        bd.RenderTarget[0].BlendOp               = D3D11_BLEND_OP_ADD;
        bd.RenderTarget[0].SrcBlendAlpha         = D3D11_BLEND_ONE;
        bd.RenderTarget[0].DestBlendAlpha        = D3D11_BLEND_ZERO;
        bd.RenderTarget[0].BlendOpAlpha          = D3D11_BLEND_OP_ADD;
        bd.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
        hr = device->CreateBlendState(&bd, &blendState);
        if (FAILED(hr)) goto dbg_cleanup;
    }
    {
        D3D11_RASTERIZER_DESC rd = {};
        rd.FillMode = D3D11_FILL_SOLID; rd.CullMode = D3D11_CULL_NONE; rd.DepthClipEnable = TRUE;
        hr = device->CreateRasterizerState(&rd, &rsState);
        if (FAILED(hr)) goto dbg_cleanup;
    }
    {
        D3D11_DEPTH_STENCIL_DESC dsd = {};
        dsd.DepthEnable    = TRUE;
        dsd.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
        dsd.DepthFunc      = D3D11_COMPARISON_LESS;
        dsd.StencilEnable  = FALSE;
        hr = device->CreateDepthStencilState(&dsd, &dsState);
        if (FAILED(hr)) goto dbg_cleanup;
    }

    dbgReady = true;
dbg_cleanup:
    vsBlob->Release(); gsBlob->Release(); psBlob->Release();
    if (!dbgReady) MGlobal::displayError("[GaussianSplat] Debug pipeline init failed.");
    return dbgReady;
}

// ===========================================================================
// GaussianDrawOverride
// ===========================================================================
MHWRender::MPxDrawOverride*
GaussianDrawOverride::creator(const MObject& obj) { return new GaussianDrawOverride(obj); }

GaussianDrawOverride::GaussianDrawOverride(const MObject& obj)
    : MHWRender::MPxDrawOverride(obj, GaussianDrawOverride::draw)
{
    MFnDependencyNode fn(obj);
    m_node = dynamic_cast<GaussianNode*>(fn.userNode());
}

MHWRender::DrawAPI GaussianDrawOverride::supportedDrawAPIs() const
{
    return MHWRender::kDirectX11;
}

bool GaussianDrawOverride::isBounded(const MDagPath& objPath, const MDagPath&) const
{
    if (!m_node) return false;
    GaussianDataNode* dn = m_node->findConnectedDataNode();
    return (dn && dn->hasData());
}

MBoundingBox GaussianDrawOverride::boundingBox(const MDagPath& objPath, const MDagPath&) const
{
    if (!m_node) return MBoundingBox(MPoint(-1,-1,-1), MPoint(1,1,1));
    GaussianDataNode* dn = m_node->findConnectedDataNode();
    if (!dn || !dn->hasData()) return MBoundingBox(MPoint(-1,-1,-1), MPoint(1,1,1));
    return dn->boundingBox();
}

// ---------------------------------------------------------------------------
// prepareForDraw
// ---------------------------------------------------------------------------
MUserData* GaussianDrawOverride::prepareForDraw(
    const MDagPath&                 objPath,
    const MDagPath&                 /*cameraPath*/,
    const MHWRender::MFrameContext& frameContext,
    MUserData*                      oldData)
{
    GaussianDrawData* data = dynamic_cast<GaussianDrawData*>(oldData);
    if (!data) data = new GaussianDrawData();
    if (!m_node) return data;

    MHWRender::MRenderer* renderer = MHWRender::MRenderer::theRenderer();
    ID3D11Device* device = static_cast<ID3D11Device*>(renderer->GPUDeviceHandle());

    // Init debug pipeline (lazy, once)
    if (!data->dbgReady && device) {
        if (data->initDebugPipeline(device))
            MGlobal::displayInfo("[GaussianSplat] Debug pipeline: OK");
        else
            MGlobal::displayError("[GaussianSplat] Debug pipeline: FAILED");
    }

    // Find connected data node
    GaussianDataNode* dataNode = m_node->findConnectedDataNode();

    // Clear shared SRV refs each frame
    data->sharedSrvPositionWS = nullptr;
    data->sharedSrvScale      = nullptr;
    data->sharedSrvRotation   = nullptr;
    data->sharedSrvOpacity    = nullptr;
    data->sharedSrvSHCoeffs   = nullptr;
    data->inputsReady         = false;
    data->vertexCount         = 0;
    data->registeredWithManager = false;

    if (!dataNode) return data;

    // Trigger data node compute (DG evaluation)
    MPlug dataReadyPlug(dataNode->thisMObject(), GaussianDataNode::aDataReady);
    dataReadyPlug.asBool();

    if (!dataNode->hasData()) return data;

    // Lazy GPU upload of shared input buffers (in data node)
    if (device) {
        dataNode->uploadInputBuffersIfNeeded(device);
    }

    if (!dataNode->areInputsReady()) return data;

    // Copy non-owning SRV pointers from data node (for debug path)
    data->sharedSrvPositionWS = dataNode->srvPositionWS();
    data->sharedSrvScale      = dataNode->srvScale();
    data->sharedSrvRotation   = dataNode->srvRotation();
    data->sharedSrvOpacity    = dataNode->srvOpacity();
    data->sharedSrvSHCoeffs   = dataNode->srvSHCoeffs();
    data->inputsReady         = true;

    uint32_t N = dataNode->splatCount();
    data->vertexCount = N;

    // Matrices
    MStatus status;
    MMatrix world    = objPath.inclusiveMatrix();
    MMatrix viewMat  = frameContext.getMatrix(MHWRender::MFrameContext::kViewMtx, &status);
    MMatrix projMat  = frameContext.getMatrix(MHWRender::MFrameContext::kProjectionMtx, &status);
    MMatrix viewProj = frameContext.getMatrix(MHWRender::MFrameContext::kViewProjMtx, &status);
    MMatrix wvp      = world * viewProj;

    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) {
            data->wvp[r*4+c]      = (float)wvp[r][c];
            data->worldMat[r*4+c] = (float)world[r][c];
            data->viewMat[r*4+c]  = (float)viewMat[r][c];
            data->projMat[r*4+c]  = (float)projMat[r][c];
        }

    // Camera position (world space)
    MMatrix invView = viewMat.inverse();
    data->cameraPos[0] = (float)invView[3][0];
    data->cameraPos[1] = (float)invView[3][1];
    data->cameraPos[2] = (float)invView[3][2];

    // Viewport
    int ox, oy, vpW, vpH;
    frameContext.getViewportDimensions(ox, oy, vpW, vpH);
    data->vpWidth  = (float)vpW;
    data->vpHeight = (float)vpH;

    // tanHalfFov
    data->tanHalfFov[0] = (float)(1.0 / projMat[0][0]);
    data->tanHalfFov[1] = (float)(1.0 / projMat[1][1]);

    // Point size
    MPlug psPlug(m_node->thisMObject(), GaussianNode::aPointSize);
    data->pointSize = psPlug.asFloat();

    // Render mode
    MPlug rmPlug(m_node->thisMObject(), GaussianNode::aRenderMode);
    data->renderMode = rmPlug.asInt();

    // -----------------------------------------------------------------------
    // Register with GaussianRenderManager for merged production rendering
    // -----------------------------------------------------------------------
    GaussianRenderManager& mgr = GaussianRenderManager::instance();

    // Detect new frame: if manager already rendered, bump the counter.
    // beginFrame() is idempotent for the same stamp, so multiple nodes
    // calling prepareForDraw in the same frame share the same stamp.
    static uint64_t s_frameCounter = 0;
    if (mgr.renderedThisFrame() || mgr.currentFrame() == 0) {
        ++s_frameCounter;
    }
    mgr.beginFrame(s_frameCounter);

    // Set camera/viewport data (idempotent, same for all instances in a frame)
    mgr.setFrameData(data->viewMat, data->projMat, data->cameraPos,
                     data->tanHalfFov, data->vpWidth, data->vpHeight);

    // Register this instance. Deduplication (same dataNode already present)
    // is handled inside registerInstance to prevent count multiplication when
    // Maya calls prepareForDraw multiple times per logical frame.
    RenderInstance inst;
    inst.dataNode   = dataNode;
    inst.splatCount = N;
    std::memcpy(inst.worldMat, data->worldMat, 64);
    mgr.registerInstance(inst);
    data->registeredWithManager = true;

    return data;
}

// ---------------------------------------------------------------------------
// draw
// ---------------------------------------------------------------------------
void GaussianDrawOverride::draw(const MHWRender::MDrawContext& context,
                                const MUserData*               userData)
{
    const GaussianDrawData* data = static_cast<const GaussianDrawData*>(userData);
    if (!data || data->vertexCount == 0) return;

    MHWRender::MRenderer* renderer = MHWRender::MRenderer::theRenderer();
    ID3D11Device* device = static_cast<ID3D11Device*>(renderer->GPUDeviceHandle());
    if (!device) return;

    ID3D11DeviceContext* ctx = nullptr;
    device->GetImmediateContext(&ctx);
    if (!ctx) return;

    // Save Maya DX11 state
    ID3D11BlendState*       prevBlend  = nullptr; float prevBF[4]; UINT prevSM;
    ID3D11RasterizerState*  prevRS     = nullptr;
    ID3D11DepthStencilState* prevDS    = nullptr; UINT prevDSRef;
    ID3D11InputLayout*      prevLayout = nullptr;
    ID3D11VertexShader*     prevVS     = nullptr;
    ID3D11GeometryShader*   prevGS     = nullptr;
    ID3D11PixelShader*      prevPS     = nullptr;
    ctx->OMGetBlendState(&prevBlend, prevBF, &prevSM);
    ctx->RSGetState(&prevRS);
    ctx->OMGetDepthStencilState(&prevDS, &prevDSRef);
    ctx->IAGetInputLayout(&prevLayout);
    ctx->VSGetShader(&prevVS, nullptr, nullptr);
    ctx->GSGetShader(&prevGS, nullptr, nullptr);
    ctx->PSGetShader(&prevPS, nullptr, nullptr);

    // -----------------------------------------------------------------------
    // Determine render path
    // -----------------------------------------------------------------------
    GaussianRenderManager& mgr = GaussianRenderManager::instance();

    bool useProd = false;
    if (data->renderMode == 1) {
        useProd = false;  // force debug
    } else if (data->registeredWithManager) {
        useProd = true;   // production via manager
    }

    // -----------------------------------------------------------------------
    // Production path: delegate to RenderManager (only executes once per frame)
    // -----------------------------------------------------------------------
    if (useProd && !mgr.renderedThisFrame())
    {
        mgr.render(device, ctx, data->renderMode);
    }
    // If manager already rendered this frame, this draw() is a no-op for production.
    // The merged pipeline already drew ALL instances in one pass.

    // -----------------------------------------------------------------------
    // Debug fallback path  (circles from shared StructuredBuffers, per-node)
    // -----------------------------------------------------------------------
    if (!useProd && data->dbgReady && data->inputsReady)
    {
        float blendFactor[] = { 1.f, 1.f, 1.f, 1.f };
        ctx->OMSetBlendState(data->blendState, blendFactor, 0xFFFFFFFF);
        ctx->RSSetState(data->rsState);
        ctx->OMSetDepthStencilState(data->dsState, 0);

        // Update debug CB
        {
            D3D11_MAPPED_SUBRESOURCE mapped;
            if (SUCCEEDED(ctx->Map(data->dbgCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
                CBDebug* cb = static_cast<CBDebug*>(mapped.pData);
                std::memcpy(cb->wvp, data->wvp, sizeof(cb->wvp));
                cb->pointSize = data->pointSize;
                cb->vpWidth   = data->vpWidth;
                cb->vpHeight  = data->vpHeight;
                cb->shStride  = (uint32_t)kSHCoeffsPerSplat;
                ctx->Unmap(data->dbgCB, 0);
            }
        }

        // Bind shared StructuredBuffers as VS SRVs
        ID3D11ShaderResourceView* vsSRVs[] = {
            data->sharedSrvPositionWS, data->sharedSrvOpacity, data->sharedSrvSHCoeffs
        };
        ctx->IASetInputLayout(nullptr);
        ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);
        ctx->VSSetShader(data->dbgVS, nullptr, 0);
        ctx->VSSetConstantBuffers(0, 1, &data->dbgCB);
        ctx->VSSetShaderResources(0, 3, vsSRVs);
        ctx->GSSetShader(data->dbgGS, nullptr, 0);
        ctx->GSSetConstantBuffers(0, 1, &data->dbgCB);
        ctx->PSSetShader(data->dbgPS, nullptr, 0);
        ctx->Draw(data->vertexCount, 0);

        // Unbind
        ID3D11ShaderResourceView* nullSRVs[3] = {};
        ctx->VSSetShaderResources(0, 3, nullSRVs);
    }

    // Restore Maya state
    ctx->IASetInputLayout(prevLayout);
    ctx->VSSetShader(prevVS, nullptr, 0);
    ctx->GSSetShader(prevGS, nullptr, 0);
    ctx->PSSetShader(prevPS, nullptr, 0);
    ctx->OMSetBlendState(prevBlend, prevBF, prevSM);
    ctx->RSSetState(prevRS);
    ctx->OMSetDepthStencilState(prevDS, prevDSRef);

    auto safeRelease = [](IUnknown* p) { if (p) p->Release(); };
    safeRelease(prevBlend); safeRelease(prevRS); safeRelease(prevDS);
    safeRelease(prevLayout); safeRelease(prevVS); safeRelease(prevGS); safeRelease(prevPS);
    ctx->Release();
}

#undef SAFE_RELEASE
