#include "PresetState.hpp"

#include "MilkdropStaticShaders.hpp"
#include "PresetFileParser.hpp"
#include "PerFrameContext.hpp"
#include <cmath>

#include <Renderer/ShaderCache.hpp>

#include <glm/gtc/matrix_transform.hpp>

#include <random>

namespace libprojectM {
namespace MilkdropPreset {

const glm::mat4 PresetState::orthogonalProjection = glm::ortho(-1.0f, 1.0f, 1.0f, -1.0f, -40.0f, 40.0f);
const glm::mat4 PresetState::orthogonalProjectionFlipped = glm::ortho(-1.0f, 1.0f, -1.0f, 1.0f, -40.0f, 40.0f);

PresetState::PresetState()
    : globalMemory(projectm_eval_memory_buffer_create())
{
    std::random_device randomDevice;
    std::mt19937 randomGenerator(randomDevice());
    std::uniform_int_distribution<> distrib(0, std::numeric_limits<int>::max());

    hueRandomOffsets[0] = static_cast<float>(distrib(randomGenerator) % 64841L) * 0.01f;
    hueRandomOffsets[1] = static_cast<float>(distrib(randomGenerator) % 53751L) * 0.01f;
    hueRandomOffsets[2] = static_cast<float>(distrib(randomGenerator) % 42661L) * 0.01f;
    hueRandomOffsets[3] = static_cast<float>(distrib(randomGenerator) % 31571L) * 0.01f;
}

PresetState::~PresetState()
{
    projectm_eval_memory_buffer_destroy(globalMemory);
}

void PresetState::Initialize(PresetFileParser& parsedFile)
{

    // General:
    decay = parsedFile.GetFloat("fDecay", decay);
    gammaAdj = parsedFile.GetFloat("fGammaAdj", gammaAdj);
    videoEchoZoom = parsedFile.GetFloat("fVideoEchoZoom", videoEchoZoom);
    videoEchoAlpha = parsedFile.GetFloat("fVideoEchoAlpha", videoEchoAlpha);
    videoEchoOrientation = parsedFile.GetInt("nVideoEchoOrientation", videoEchoOrientation);
    redBlueStereo = parsedFile.GetBool("bRedBlueStereo", redBlueStereo);
    brighten = parsedFile.GetBool("bBrighten", brighten);
    darken = parsedFile.GetBool("bDarken", darken);
    solarize = parsedFile.GetBool("bSolarize", solarize);
    invert = parsedFile.GetBool("bInvert", invert);
    shader = parsedFile.GetFloat("fShader", shader);
    blur1Min = parsedFile.GetFloat("b1n", blur1Min);
    blur2Min = parsedFile.GetFloat("b2n", blur2Min);
    blur3Min = parsedFile.GetFloat("b3n", blur3Min);
    blur1Max = parsedFile.GetFloat("b1x", blur1Max);
    blur2Max = parsedFile.GetFloat("b2x", blur2Max);
    blur3Max = parsedFile.GetFloat("b3x", blur3Max);
    blur1EdgeDarken = parsedFile.GetFloat("b1ed", blur1EdgeDarken);

    // Wave:
    waveMode = parsedFile.GetInt("nWaveMode", waveMode);
    additiveWaves = parsedFile.GetBool("bAdditiveWaves", additiveWaves);
    waveDots = parsedFile.GetBool("bWaveDots", waveDots);
    waveThick = parsedFile.GetBool("bWaveThick", waveThick);
    modWaveAlphaByvolume = parsedFile.GetBool("bModWaveAlphaByVolume", modWaveAlphaByvolume);
    maximizeWaveColor = parsedFile.GetBool("bMaximizeWaveColor", maximizeWaveColor);
    waveAlpha = parsedFile.GetFloat("fWaveAlpha", waveAlpha);
    waveScale = parsedFile.GetFloat("fWaveScale", waveScale);
    waveSmoothing = parsedFile.GetFloat("fWaveSmoothing", waveSmoothing);
    waveParam = parsedFile.GetFloat("fWaveParam", waveParam);
    modWaveAlphaStart = parsedFile.GetFloat("fModWaveAlphaStart", modWaveAlphaStart);
    modWaveAlphaEnd = parsedFile.GetFloat("fModWaveAlphaEnd", modWaveAlphaEnd);
    waveR = parsedFile.GetFloat("wave_r", waveR);
    waveG = parsedFile.GetFloat("wave_g", waveG);
    waveB = parsedFile.GetFloat("wave_b", waveB);
    waveX = parsedFile.GetFloat("wave_x", waveX);
    waveY = parsedFile.GetFloat("wave_y", waveY);
    mvX = parsedFile.GetFloat("nMotionVectorsX", mvX);
    mvY = parsedFile.GetFloat("nMotionVectorsY", mvY);
    mvDX = parsedFile.GetFloat("mv_dx", mvDX);
    mvDY = parsedFile.GetFloat("mv_dy", mvDY);
    mvL = parsedFile.GetFloat("mv_l", mvL);
    mvR = parsedFile.GetFloat("mv_r", mvR);
    mvG = parsedFile.GetFloat("mv_g", mvG);
    mvB = parsedFile.GetFloat("mv_b", mvB);
    mvA = parsedFile.GetBool("bMotionVectorsOn", false) ? 1.0f : 0.0f; // for backwards compatibility
    mvA = parsedFile.GetFloat("mv_a", mvA);

    // Motion:
    zoom = parsedFile.GetFloat("zoom", zoom);
    rot = parsedFile.GetFloat("rot", rot);
    rotCX = parsedFile.GetFloat("cx", rotCX);
    rotCY = parsedFile.GetFloat("cy", rotCY);
    xPush = parsedFile.GetFloat("dx", xPush);
    yPush = parsedFile.GetFloat("dy", yPush);
    warpAmount = parsedFile.GetFloat("warp", warpAmount);
    stretchX = parsedFile.GetFloat("sx", stretchX);
    stretchY = parsedFile.GetFloat("sy", stretchY);
    texWrap = parsedFile.GetBool("bTexWrap", texWrap);
    darkenCenter = parsedFile.GetBool("bDarkenCenter", darkenCenter);
    warpAnimSpeed = parsedFile.GetFloat("fWarpAnimSpeed", warpAnimSpeed);
    warpScale = parsedFile.GetFloat("fWarpScale", warpScale);
    zoomExponent = parsedFile.GetFloat("fZoomExponent", zoomExponent);

    // Borders:
    outerBorderSize = parsedFile.GetFloat("ob_size", outerBorderSize);
    outerBorderR = parsedFile.GetFloat("ob_r", outerBorderR);
    outerBorderG = parsedFile.GetFloat("ob_g", outerBorderG);
    outerBorderB = parsedFile.GetFloat("ob_b", outerBorderB);
    outerBorderA = parsedFile.GetFloat("ob_a", outerBorderA);
    innerBorderSize = parsedFile.GetFloat("ib_size", innerBorderSize);
    innerBorderR = parsedFile.GetFloat("ib_r", innerBorderR);
    innerBorderG = parsedFile.GetFloat("ib_g", innerBorderG);
    innerBorderB = parsedFile.GetFloat("ib_b", innerBorderB);
    innerBorderA = parsedFile.GetFloat("ib_a", innerBorderA);

    // Versions:
    presetVersion = parsedFile.GetInt("MILKDROP_PRESET_VERSION", presetVersion);
    if (presetVersion < 200)
    {
        // Milkdrop 1.x did not use shaders.
        warpShaderVersion = 0;
        compositeShaderVersion = 0;
    }
    else if (presetVersion == 200)
    {
        // Milkdrop 2.0 only supported a single shader language level variable.
        warpShaderVersion = parsedFile.GetInt("PSVERSION", warpShaderVersion);
        compositeShaderVersion = parsedFile.GetInt("PSVERSION", compositeShaderVersion);
    }
    else
    {
        warpShaderVersion = parsedFile.GetInt("PSVERSION_WARP", warpShaderVersion);
        compositeShaderVersion = parsedFile.GetInt("PSVERSION_COMP", compositeShaderVersion);
    }

    // Code:
    perFrameInitCode = parsedFile.GetCode("per_frame_init_");
    perFrameCode = parsedFile.GetCode("per_frame_");
    perPixelCode = parsedFile.GetCode("per_pixel_");

    // Custom waveform code:
    for (int i = 0; i < CustomWaveformCount; i++)
    {
        std::string const wavePrefix = "wave_" + std::to_string(i) + "_";
        customWaveInitCode[i] = parsedFile.GetCode(wavePrefix + "init");
        customWavePerFrameCode[i] = parsedFile.GetCode(wavePrefix + "per_frame");
        customWavePerPointCode[i] = parsedFile.GetCode(wavePrefix + "per_point");
    }

    // Custom shape code:
    for (int i = 0; i < CustomShapeCount; i++)
    {
        std::string const shapePrefix = "shape_" + std::to_string(i) + "_";
        customShapeInitCode[i] = parsedFile.GetCode(shapePrefix + "init");
        customShapePerFrameCode[i] = parsedFile.GetCode(shapePrefix + "per_frame");
    }

    // Shader code:
    warpShader = parsedFile.GetCode("warp_");
    compositeShader = parsedFile.GetCode("comp_");
}

void PresetState::LoadShaders()
{
    auto staticShaders = libprojectM::MilkdropPreset::MilkdropStaticShaders::Get();

    auto untexturedShaderShared = renderContext.shaderCache->Get("milkdrop_generic_untextured");
    if (!untexturedShaderShared)
    {
        untexturedShaderShared = std::make_shared<Renderer::Shader>();
        untexturedShaderShared->CompileProgram(staticShaders->GetUntexturedDrawVertexShader(),
                                        staticShaders->GetUntexturedDrawFragmentShader());
        renderContext.shaderCache->Insert("milkdrop_generic_untextured", untexturedShaderShared);
    }
    untexturedShader = untexturedShaderShared;

    auto texturedShaderShared = renderContext.shaderCache->Get("milkdrop_generic_textured");
    if (!texturedShaderShared)
    {
        texturedShaderShared = std::make_shared<Renderer::Shader>();
        texturedShaderShared->CompileProgram(staticShaders->GetTexturedDrawVertexShader(),
                                      staticShaders->GetTexturedDrawFragmentShader());
        renderContext.shaderCache->Insert("milkdrop_generic_textured", texturedShaderShared);
    }
    texturedShader = texturedShaderShared;
}

void PresetState::UpdateUniformConstants(const PerFrameContext& perFrameContext)
{
    // 1. Viewport-dependent static constants: only recompute when screen size changes
    int const curW = renderContext.viewportSizeX;
    int const curH = renderContext.viewportSizeY;
    if (curW != uniformCache.lastViewportW || curH != uniformCache.lastViewportH)
    {
        uniformCache.lastViewportW = curW;
        uniformCache.lastViewportH = curH;

        float const fW = static_cast<float>(curW);
        float const fH = static_cast<float>(curH);
        float const invW = (curW > 0) ? (1.0f / fW) : 1.0f;
        float const invH = (curH > 0) ? (1.0f / fH) : 1.0f;

        float const aspX = renderContext.aspectX;
        float const aspY = renderContext.aspectY;
        float const invAspX = (aspX != 0.0f) ? (1.0f / aspX) : 1.0f;
        float const invAspY = (aspY != 0.0f) ? (1.0f / aspY) : 1.0f;

        // _c0: aspect
        uniformCache.c0[0] = aspX;
        uniformCache.c0[1] = aspY;
        uniformCache.c0[2] = invAspX;
        uniformCache.c0[3] = invAspY;

        // _c7: viewport
        uniformCache.c7[0] = fW;
        uniformCache.c7[1] = fH;
        uniformCache.c7[2] = invW;
        uniformCache.c7[3] = invH;

        // _c12: mip info (log2)
        float const mipX = (fW > 0.0f) ? log2f(fW) : 0.0f;
        float const mipY = (fH > 0.0f) ? log2f(fH) : 0.0f;
        float const mipAvg = 0.5f * (mipX + mipY);
        uniformCache.c12[0] = mipX;
        uniformCache.c12[1] = mipY;
        uniformCache.c12[2] = mipAvg;
        uniformCache.c12[3] = 0.0f;
    }

    // 2. Avoid recomputing per-frame constants if already updated for this exact frame
    if (uniformCache.lastFrameNumber == renderContext.frame)
    {
        return;
    }
    uniformCache.lastFrameNumber = renderContext.frame;

    // Time calculations
    auto floatTime = static_cast<float>(renderContext.time);
    auto timeSincePresetStartWrapped = floatTime - static_cast<int>(floatTime / 10000.0) * 10000;

    // _c2: timing
    uniformCache.c2[0] = timeSincePresetStartWrapped;
    uniformCache.c2[1] = renderContext.fps;
    uniformCache.c2[2] = static_cast<float>(renderContext.frame);
    uniformCache.c2[3] = renderContext.progress;

    // _c3, _c4: audio data
    uniformCache.c3[0] = audioData.bass;
    uniformCache.c3[1] = audioData.mid;
    uniformCache.c3[2] = audioData.treb;
    uniformCache.c3[3] = audioData.vol;

    uniformCache.c4[0] = audioData.bassAtt;
    uniformCache.c4[1] = audioData.midAtt;
    uniformCache.c4[2] = audioData.trebAtt;
    uniformCache.c4[3] = audioData.volAtt;

    // Blur values
    BlurTexture::Values blurMin;
    BlurTexture::Values blurMax;
    BlurTexture::GetSafeBlurMinMaxValues(perFrameContext, blurMin, blurMax);

    uniformCache.c5[0] = blurMax[0] - blurMin[0];
    uniformCache.c5[1] = blurMin[0];
    uniformCache.c5[2] = blurMax[1] - blurMin[1];
    uniformCache.c5[3] = blurMin[1];

    uniformCache.c6[0] = blurMax[2] - blurMin[2];
    uniformCache.c6[1] = blurMin[2];
    uniformCache.c6[2] = blurMin[0];
    uniformCache.c6[3] = blurMax[0];

    uniformCache.c13[0] = blurMin[1];
    uniformCache.c13[1] = blurMax[1];
    uniformCache.c13[2] = blurMin[2];
    uniformCache.c13[3] = blurMax[2];

    // Fast trig: _c8 (cos), _c9 (sin)
    uniformCache.c8[0] = 0.5f + 0.5f * cosf(floatTime * 0.329f + 1.2f);
    uniformCache.c8[1] = 0.5f + 0.5f * cosf(floatTime * 1.293f + 3.9f);
    uniformCache.c8[2] = 0.5f + 0.5f * cosf(floatTime * 5.070f + 2.5f);
    uniformCache.c8[3] = 0.5f + 0.5f * cosf(floatTime * 20.051f + 5.4f);

    uniformCache.c9[0] = 0.5f + 0.5f * sinf(floatTime * 0.329f + 1.2f);
    uniformCache.c9[1] = 0.5f + 0.5f * sinf(floatTime * 1.293f + 3.9f);
    uniformCache.c9[2] = 0.5f + 0.5f * sinf(floatTime * 5.070f + 2.5f);
    uniformCache.c9[3] = 0.5f + 0.5f * sinf(floatTime * 20.051f + 5.4f);

    // Slow trig: _c10 (cos), _c11 (sin)
    uniformCache.c10[0] = 0.5f + 0.5f * cosf(floatTime * 0.0050f + 2.7f);
    uniformCache.c10[1] = 0.5f + 0.5f * cosf(floatTime * 0.0085f + 5.3f);
    uniformCache.c10[2] = 0.5f + 0.5f * cosf(floatTime * 0.0133f + 4.5f);
    uniformCache.c10[3] = 0.5f + 0.5f * cosf(floatTime * 0.0217f + 3.8f);

    uniformCache.c11[0] = 0.5f + 0.5f * sinf(floatTime * 0.0050f + 2.7f);
    uniformCache.c11[1] = 0.5f + 0.5f * sinf(floatTime * 0.0085f + 5.3f);
    uniformCache.c11[2] = 0.5f + 0.5f * sinf(floatTime * 0.0133f + 4.5f);
    uniformCache.c11[3] = 0.5f + 0.5f * sinf(floatTime * 0.0217f + 3.8f);

    // Q-variables (q1..q32):
    for (size_t i = 0; i < 8; ++i)
    {
        size_t const qIdx = i * 4;
        uniformCache.qValues[i][0] = static_cast<float>(frameQVariables[qIdx]);
        uniformCache.qValues[i][1] = static_cast<float>(frameQVariables[qIdx + 1]);
        uniformCache.qValues[i][2] = static_cast<float>(frameQVariables[qIdx + 2]);
        uniformCache.qValues[i][3] = static_cast<float>(frameQVariables[qIdx + 3]);
    }
}

} // namespace MilkdropPreset
} // namespace libprojectM
