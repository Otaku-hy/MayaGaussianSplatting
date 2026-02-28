#include "GaussianDrawOverride.h"
#include "GaussianNode.h"
#include "PLYReader.h"

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

#include <cstring>

// ===========================================================================
// Inline HLSL source  (compiled at runtime via D3DCompile)
// ===========================================================================
static const char* kShaderSrc = R"HLSL(
cbuffer CBPerObject : register(b0)
{
    row_major float4x4 gWVP;
    float  gPointSize;
    float  gVPWidth;
    float  gVPHeight;
    float  gPad;
};

// ---- structs ---------------------------------------------------------------
struct VS_IN  { float3 pos : POSITION;  float4 col : COLOR; };
struct VS_OUT { float4 clip : SV_Position; float4 col : COLOR; };
struct GS_OUT { float4 clip : SV_Position; float4 col : COLOR; float2 uv : TEXCOORD0; };

// ---- vertex shader ---------------------------------------------------------
VS_OUT VS(VS_IN i)
{
    VS_OUT o;
    o.clip = mul(float4(i.pos, 1.0f), gWVP);
    o.col  = i.col;
    return o;
}

// ---- geometry shader  (point -> screen-aligned quad) ----------------------
[maxvertexcount(4)]
void GS(point VS_OUT input[1], inout TriangleStream<GS_OUT> stream)
{
    float4 c = input[0].clip;

    // half-size in clip space (constant pixel radius regardless of depth)
    float2 h = float2(gPointSize / gVPWidth, gPointSize / gVPHeight) * c.w;

    static const float2 corners[4] = {
        float2(-1.f,  1.f),   // TL
        float2( 1.f,  1.f),   // TR
        float2(-1.f, -1.f),   // BL
        float2( 1.f, -1.f),   // BR
    };

    GS_OUT o;
    o.col = input[0].col;
    [unroll]
    for (int k = 0; k < 4; k++)
    {
        o.clip = c + float4(corners[k] * h, 0.f, 0.f);
        o.uv   = corners[k];
        stream.Append(o);
    }
    stream.RestartStrip();
}

// ---- pixel shader ----------------------------------------------------------
float4 PS(GS_OUT i) : SV_Target
{
    float r2 = dot(i.uv, i.uv);
    clip(1.0f - r2);                          // circular mask
    float a = i.col.a * (1.0f - r2 * 0.4f);  // soft edge
    return float4(i.col.rgb, a);
}
)HLSL";

// ===========================================================================
// CBPerObject layout (must match HLSL)
// ===========================================================================
struct CBPerObject {
    float wvp[16];
    float pointSize;
    float vpWidth;
    float vpHeight;
    float pad;
};
static_assert(sizeof(CBPerObject) % 16 == 0, "CB must be 16-byte aligned");

// ===========================================================================
// GaussianDrawData
// ===========================================================================
GaussianDrawData::GaussianDrawData()
    : MUserData()
{}

GaussianDrawData::~GaussianDrawData() { releaseAll(); }

void GaussianDrawData::releaseAll()
{
#define SAFE_RELEASE(p) do { if (p) { (p)->Release(); (p) = nullptr; } } while(0)
    SAFE_RELEASE(vs);
    SAFE_RELEASE(gs);
    SAFE_RELEASE(ps);
    SAFE_RELEASE(inputLayout);
    SAFE_RELEASE(posBuf);
    SAFE_RELEASE(colBuf);
    SAFE_RELEASE(constBuf);
    SAFE_RELEASE(blendState);
    SAFE_RELEASE(rsState);
#undef SAFE_RELEASE
    shadersReady = false;
    vertexCount  = 0;
}

// ---------------------------------------------------------------------------
bool GaussianDrawData::initShaders(ID3D11Device* device)
{
    HRESULT hr;
    ID3DBlob* vsBlob = nullptr;
    ID3DBlob* gsBlob = nullptr;
    ID3DBlob* psBlob = nullptr;
    ID3DBlob* errBlob = nullptr;

    UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;
#ifdef _DEBUG
    flags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif

    auto compileStage = [&](const char* entry, const char* target, ID3DBlob** out) -> bool {
        HRESULT r = D3DCompile(kShaderSrc, strlen(kShaderSrc), nullptr, nullptr, nullptr,
                               entry, target, flags, 0, out, &errBlob);
        if (FAILED(r)) {
            if (errBlob) {
                MGlobal::displayError(
                    MString("[GaussianSplat] Shader compile error (") + entry + "): " +
                    static_cast<const char*>(errBlob->GetBufferPointer()));
                errBlob->Release();
            }
            return false;
        }
        if (errBlob) { errBlob->Release(); errBlob = nullptr; }
        return true;
    };

    if (!compileStage("VS", "vs_5_0", &vsBlob)) return false;
    if (!compileStage("GS", "gs_5_0", &gsBlob)) { vsBlob->Release(); return false; }
    if (!compileStage("PS", "ps_5_0", &psBlob)) { vsBlob->Release(); gsBlob->Release(); return false; }

    hr = device->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), nullptr, &vs);
    if (FAILED(hr)) goto cleanup;

    hr = device->CreateGeometryShader(gsBlob->GetBufferPointer(), gsBlob->GetBufferSize(), nullptr, &gs);
    if (FAILED(hr)) goto cleanup;

    hr = device->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(), nullptr, &ps);
    if (FAILED(hr)) goto cleanup;

    {   // Input layout: POSITION (float3, slot 0) + COLOR (float4, slot 1)
        D3D11_INPUT_ELEMENT_DESC layout[] = {
            { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT,   0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
            { "COLOR",    0, DXGI_FORMAT_R32G32B32A32_FLOAT, 1, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        };
        hr = device->CreateInputLayout(layout, 2,
                                       vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(),
                                       &inputLayout);
        if (FAILED(hr)) goto cleanup;
    }

    {   // Constant buffer (dynamic, written every frame)
        D3D11_BUFFER_DESC cbd = {};
        cbd.ByteWidth      = sizeof(CBPerObject);
        cbd.Usage          = D3D11_USAGE_DYNAMIC;
        cbd.BindFlags      = D3D11_BIND_CONSTANT_BUFFER;
        cbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        hr = device->CreateBuffer(&cbd, nullptr, &constBuf);
        if (FAILED(hr)) goto cleanup;
    }

    {   // Alpha-blend state
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
        if (FAILED(hr)) goto cleanup;
    }

    {   // Rasterizer: no back-face culling (quads may be any winding)
        D3D11_RASTERIZER_DESC rd = {};
        rd.FillMode        = D3D11_FILL_SOLID;
        rd.CullMode        = D3D11_CULL_NONE;
        rd.DepthClipEnable = TRUE;
        hr = device->CreateRasterizerState(&rd, &rsState);
        if (FAILED(hr)) goto cleanup;
    }

    shadersReady = true;

cleanup:
    vsBlob->Release();
    gsBlob->Release();
    psBlob->Release();
    if (FAILED(hr)) {
        MGlobal::displayError("[GaussianSplat] DX11 resource creation failed.");
        releaseAll();
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
bool GaussianDrawData::uploadVertices(ID3D11Device* device,
                                      const float*  positions,
                                      const float*  colors,
                                      unsigned int  count)
{
    // Release old buffers
    if (posBuf) { posBuf->Release(); posBuf = nullptr; }
    if (colBuf) { colBuf->Release(); colBuf = nullptr; }
    vertexCount = 0;

    if (!device || count == 0) return false;

    HRESULT hr;

    {   // Position buffer  (N * float3)
        D3D11_BUFFER_DESC bd = {};
        bd.ByteWidth  = count * 3 * sizeof(float);
        bd.Usage      = D3D11_USAGE_IMMUTABLE;
        bd.BindFlags  = D3D11_BIND_VERTEX_BUFFER;
        D3D11_SUBRESOURCE_DATA init = { positions, 0, 0 };
        hr = device->CreateBuffer(&bd, &init, &posBuf);
        if (FAILED(hr)) {
            MGlobal::displayError("[GaussianSplat] Failed to create position vertex buffer.");
            return false;
        }
    }

    {   // Colour buffer  (N * float4)
        D3D11_BUFFER_DESC bd = {};
        bd.ByteWidth  = count * 4 * sizeof(float);
        bd.Usage      = D3D11_USAGE_IMMUTABLE;
        bd.BindFlags  = D3D11_BIND_VERTEX_BUFFER;
        D3D11_SUBRESOURCE_DATA init = { colors, 0, 0 };
        hr = device->CreateBuffer(&bd, &init, &colBuf);
        if (FAILED(hr)) {
            MGlobal::displayError("[GaussianSplat] Failed to create colour vertex buffer.");
            posBuf->Release(); posBuf = nullptr;
            return false;
        }
    }

    vertexCount  = count;
    vertexDirty  = false;
    return true;
}

// ===========================================================================
// GaussianDrawOverride
// ===========================================================================
MHWRender::MPxDrawOverride*
GaussianDrawOverride::creator(const MObject& obj)
{
    return new GaussianDrawOverride(obj);
}

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

    // ---- Get DX11 device ----
    MHWRender::MRenderer* renderer = MHWRender::MRenderer::theRenderer();
    ID3D11Device* device = static_cast<ID3D11Device*>(renderer->GPUDeviceHandle());

    // ---- Init shaders once ----
    if (!data->shadersReady && device)
        data->initShaders(device);

    // ---- Reload PLY if filePath attribute changed ----
    MPlug pathPlug(m_node->thisMObject(), GaussianNode::aFilePath);
    MString newPath = pathPlug.asString();

    if (newPath != m_loadedPath) {
        m_loadedPath = newPath;
        m_node->m_data.clear();

        if (newPath.length() > 0) {
            std::string err;
            if (PLYReader::read(newPath.asChar(), m_node->m_data, err)) {
                MGlobal::displayInfo(
                    MString("[GaussianSplat] Loaded ") +
                    (unsigned int)m_node->m_data.count() + " splats.");
            } else {
                MGlobal::displayError(MString("[GaussianSplat] ") + err.c_str());
            }
        }
        data->vertexDirty = true;
    }

    // ---- Upload vertex data if dirty ----
    if (data->vertexDirty && device && !m_node->m_data.empty()) {
        const GaussianData& gd = m_node->m_data;
        data->uploadVertices(device,
                             gd.positions.data(),
                             gd.colors.data(),
                             (unsigned int)gd.count());
    }

    // ---- World-View-Projection matrix ----
    {
        MStatus  status;
        MMatrix  world    = objPath.inclusiveMatrix();
        MMatrix  viewProj = frameContext.getMatrix(
                                MHWRender::MFrameContext::kViewProjMtx, &status);
        MMatrix  wvp      = world * viewProj;

        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 4; c++)
                data->wvp[r * 4 + c] = static_cast<float>(wvp[r][c]);
    }

    // ---- Viewport size ----
    {
        int ox, oy, vpW, vpH;
        frameContext.getViewportDimensions(ox, oy, vpW, vpH);
        data->vpWidth  = static_cast<float>(vpW);
        data->vpHeight = static_cast<float>(vpH);
    }

    // ---- Point size from node attribute ----
    {
        MPlug psPlug(m_node->thisMObject(), GaussianNode::aPointSize);
        data->pointSize = psPlug.asFloat();
    }

    return data;
}

// ---------------------------------------------------------------------------
void GaussianDrawOverride::draw(const MHWRender::MDrawContext& context,
                                const MUserData*               userData)
{
    const GaussianDrawData* data = static_cast<const GaussianDrawData*>(userData);
    if (!data || !data->shadersReady || data->vertexCount == 0) return;

    // ---- Get DX11 device + immediate context ----
    MHWRender::MRenderer* renderer = MHWRender::MRenderer::theRenderer();
    ID3D11Device* device = static_cast<ID3D11Device*>(renderer->GPUDeviceHandle());
    if (!device) return;

    ID3D11DeviceContext* ctx = nullptr;
    device->GetImmediateContext(&ctx);
    if (!ctx) return;

    // ---- Update constant buffer ----
    {
        D3D11_MAPPED_SUBRESOURCE mapped;
        if (SUCCEEDED(ctx->Map(data->constBuf, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
            CBPerObject* cb = static_cast<CBPerObject*>(mapped.pData);
            std::memcpy(cb->wvp, data->wvp, sizeof(cb->wvp));
            cb->pointSize = data->pointSize;
            cb->vpWidth   = data->vpWidth;
            cb->vpHeight  = data->vpHeight;
            cb->pad       = 0.f;
            ctx->Unmap(data->constBuf, 0);
        }
    }

    // ---- Save minimal Maya state ----
    ID3D11BlendState*      prevBlend   = nullptr;  float prevBlendFactor[4]; UINT prevSampleMask;
    ID3D11RasterizerState* prevRS      = nullptr;
    ID3D11InputLayout*     prevLayout  = nullptr;
    ID3D11VertexShader*    prevVS      = nullptr;
    ID3D11GeometryShader*  prevGS      = nullptr;
    ID3D11PixelShader*     prevPS      = nullptr;

    ctx->OMGetBlendState(&prevBlend, prevBlendFactor, &prevSampleMask);
    ctx->RSGetState(&prevRS);
    ctx->IAGetInputLayout(&prevLayout);
    ctx->VSGetShader(&prevVS, nullptr, nullptr);
    ctx->GSGetShader(&prevGS, nullptr, nullptr);
    ctx->PSGetShader(&prevPS, nullptr, nullptr);

    // ---- Set our pipeline ----
    ctx->IASetInputLayout(data->inputLayout);
    ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);

    ID3D11Buffer* vbs[]     = { data->posBuf, data->colBuf };
    UINT          strides[] = { 3 * sizeof(float), 4 * sizeof(float) };
    UINT          offsets[] = { 0, 0 };
    ctx->IASetVertexBuffers(0, 2, vbs, strides, offsets);

    ctx->VSSetShader(data->vs, nullptr, 0);
    ctx->VSSetConstantBuffers(0, 1, &data->constBuf);

    ctx->GSSetShader(data->gs, nullptr, 0);
    ctx->GSSetConstantBuffers(0, 1, &data->constBuf);

    ctx->PSSetShader(data->ps, nullptr, 0);

    float blendFactor[] = { 1.f, 1.f, 1.f, 1.f };
    ctx->OMSetBlendState(data->blendState, blendFactor, 0xFFFFFFFF);
    ctx->RSSetState(data->rsState);

    // ---- Draw all splats as points (GS expands each to a quad) ----
    ctx->Draw(data->vertexCount, 0);

    // ---- Restore Maya state ----
    ctx->IASetInputLayout(prevLayout);
    ctx->VSSetShader(prevVS, nullptr, 0);
    ctx->GSSetShader(prevGS, nullptr, 0);
    ctx->PSSetShader(prevPS, nullptr, 0);
    ctx->OMSetBlendState(prevBlend, prevBlendFactor, prevSampleMask);
    ctx->RSSetState(prevRS);

    // Release refs added by Get* calls
    auto safeRelease = [](IUnknown* p) { if (p) p->Release(); };
    safeRelease(prevBlend);
    safeRelease(prevRS);
    safeRelease(prevLayout);
    safeRelease(prevVS);
    safeRelease(prevGS);
    safeRelease(prevPS);

    ctx->Release();
}
