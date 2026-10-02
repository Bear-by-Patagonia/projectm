/**
 * @file PresetState.hpp
 *
 * Declares a class that hold the variable states for a single preset.
 */
#pragma once

#include "Constants.hpp"

#include "BlurTexture.hpp"

#include <Audio/FrameAudioData.hpp>

#include <Renderer/RenderContext.hpp>
#include <Renderer/Shader.hpp>
#include <Renderer/TextureSamplerDescriptor.hpp>

#include <projectm-eval.h>

#include <string>

namespace libprojectM {
namespace MilkdropPreset {

class PresetFileParser;
class PerFrameContext;

using BlendableFloat = float; //!< Currently a placeholder to mark blendable values.

/**
 * @brief Hold the current preset state and initial values.
 *
 * This is the base state class which is filled on preset load and updated between frames
 * to reflect the current state of rendering.
 */
class PresetState
{
public:
    PresetState();

    ~PresetState();

    /**
     * @brief Loads the initial values and code from the preset file.
     * @param parsedFile The file parser with the preset data.
     */
    void Initialize(PresetFileParser& parsedFile);

    /**
     * @brief Loads or compiles the generic shaders.
     * Call after setting renderContext.
     */
    void LoadShaders();

    BlendableFloat gammaAdj{2.0f};
    BlendableFloat videoEchoZoom{2.0f};
    BlendableFloat videoEchoAlpha{0.0f};
    BlendableFloat videoEchoAlphaOld{0.0f};
    int videoEchoOrientation{0};
    int videoEchoOrientationOld{0};

    BlendableFloat decay{0.98f};

    int waveMode{0};
    int oldWaveMode{-1};
    bool additiveWaves{false};
    BlendableFloat waveAlpha{0.8f};
    BlendableFloat waveScale{1.0f};
    BlendableFloat waveSmoothing{0.75f};
    bool waveDots{false};
    bool waveThick{false};
    BlendableFloat waveParam{0.0f};
    bool modWaveAlphaByvolume{false};
    BlendableFloat modWaveAlphaStart{0.75f};
    BlendableFloat modWaveAlphaEnd{0.95f};
    float warpAnimSpeed{1.0f};
    BlendableFloat warpScale{1.0f};
    BlendableFloat zoomExponent{1.0f};
    BlendableFloat shader{0.0f};
    bool maximizeWaveColor{true};
    bool texWrap{true};
    bool darkenCenter{false};
    bool redBlueStereo{false};
    bool brighten{false};
    bool darken{false};
    bool solarize{false};
    bool invert{false};

    BlendableFloat zoom{1.0f};
    BlendableFloat rot{0.0f};
    BlendableFloat rotCX{0.5f};
    BlendableFloat rotCY{0.5f};
    BlendableFloat xPush{0.0f};
    BlendableFloat yPush{0.0f};
    BlendableFloat warpAmount{1.0f};
    BlendableFloat stretchX{1.0f};
    BlendableFloat stretchY{1.0f};
    BlendableFloat waveR{1.0f};
    BlendableFloat waveG{1.0f};
    BlendableFloat waveB{1.0f};
    BlendableFloat waveX{0.5f};
    BlendableFloat waveY{0.5f};
    BlendableFloat outerBorderSize{0.01f};
    BlendableFloat outerBorderR{0.0f};
    BlendableFloat outerBorderG{0.0f};
    BlendableFloat outerBorderB{0.0f};
    BlendableFloat outerBorderA{0.0f};
    BlendableFloat innerBorderSize{0.01f};
    BlendableFloat innerBorderR{0.25f};
    BlendableFloat innerBorderG{0.25f};
    BlendableFloat innerBorderB{0.25f};
    BlendableFloat innerBorderA{0.0f};
    BlendableFloat mvX{12.0f};
    BlendableFloat mvY{9.0f};
    BlendableFloat mvDX{0.0f};
    BlendableFloat mvDY{0.0f};
    BlendableFloat mvL{0.9f};
    BlendableFloat mvR{1.0f};
    BlendableFloat mvG{1.0f};
    BlendableFloat mvB{1.0f};
    BlendableFloat mvA{0.0f};
    BlendableFloat blur1Min{0.0f};
    BlendableFloat blur2Min{0.0f};
    BlendableFloat blur3Min{0.0f};
    BlendableFloat blur1Max{1.0f};
    BlendableFloat blur2Max{1.0f};
    BlendableFloat blur3Max{1.0f};
    BlendableFloat blur1EdgeDarken{0.25f};

    int presetVersion{100};        //!< Value of MILKDROP_PRESET_VERSION in preset files.
    int warpShaderVersion{2};      //!< PSVERSION or PSVERSION_WARP.
    int compositeShaderVersion{2}; //!< PSVERSION or PSVERSION_COMP.

    std::array<float, 4> hueRandomOffsets; //!< Per-preset constant offsets for the hue animation

    projectm_eval_mem_buffer globalMemory{nullptr};  //!< gmegabuf data. Using per-frame buffers in projectM to reduce interference.
    double globalRegisters[100]{};                   //!< Global reg00-reg99 variables.
    std::array<double, QVarCount> frameQVariables{}; //!< Q variables after per-frame code evaluation.

    libprojectM::Audio::FrameAudioData audioData; //!< Holds audio/spectrum data and values for beat detection.
    Renderer::RenderContext renderContext;        //!< Current renderer state data like viewport size and generic shaders.

    std::string perFrameInitCode; //!< Preset init code, run once on load.
    std::string perFrameCode;     //!< Preset per-frame code, run once at the start of each frame.
    std::string perPixelCode;     //!< Preset per-pixel/per-vertex code, run once per warp mesh vertex.

    std::array<std::string, CustomWaveformCount> customWaveInitCode;     //!< Custom wave init code, run once on load.
    std::array<std::string, CustomWaveformCount> customWavePerFrameCode; //!< Custom wave per-frame code, run once after the per-frame code.
    std::array<std::string, CustomWaveformCount> customWavePerPointCode; //!< Custom wave per-point code, run once per waveform vertex.

    std::array<std::string, CustomShapeCount> customShapeInitCode;     //!< Custom shape init code, run once on load.
    std::array<std::string, CustomShapeCount> customShapePerFrameCode; //!< Custom shape per-frame code, run once per shape instance.

    std::string warpShader;      //!< Warp shader code.
    std::string compositeShader; //!< Composite shader code.

    std::weak_ptr<Renderer::Shader> untexturedShader; //!< Shader used to draw untextured primitives, e.g. waveforms.
    std::weak_ptr<Renderer::Shader> texturedShader;   //!< Shader used to draw textured primitives, e.g. textured shapes and the warp mesh.

    std::weak_ptr<Renderer::Texture> mainTexture; //!< A weak reference to the main texture in the preset framebuffer.
    BlurTexture blurTexture;                      //!< The blur textures used in this preset. Contents depend on the shader code using GetBlurX().

    std::map<int, Renderer::TextureSamplerDescriptor> randomTextureDescriptors; //!< Descriptors for random texture IDs. Should be the same across both warp and comp shaders.

    static const glm::mat4 orthogonalProjection;        //!< Projection matrix that transforms DirectX screen-space coordinates into the OpenGL coordinate frame.
    static const glm::mat4 orthogonalProjectionFlipped; //!< Projection matrix that transforms DirectX screen-space coordinates into the OpenGL coordinate frame.

    struct UniformConstantsCache {
        int64_t lastFrameNumber{-1};
        int lastViewportW{0};
        int lastViewportH{0};

        // Static viewport parameters (only recomputed on size change)
        float c0[4]{};  // aspectX, aspectY, 1/aspectX, 1/aspectY
        float c7[4]{};  // viewportSizeX, viewportSizeY, 1/w, 1/h
        float c12[4]{}; // log2(w), log2(h), avg, 0.0f

        // Per-frame parameters
        float c2[4]{};  // timeSincePresetStartWrapped, fps, frame, progress
        float c3[4]{};  // audioData: bass, mid, treb, vol
        float c4[4]{};  // audioData: bassAtt, midAtt, trebAtt, volAtt
        float c5[4]{};  // blur scale/bias 1 & 2
        float c6[4]{};  // blur scale/bias 3 & min/max 1
        float c8[4]{};  // fast trig cos
        float c9[4]{};  // fast trig sin
        float c10[4]{}; // slow trig cos
        float c11[4]{}; // slow trig sin
        float c13[4]{}; // blur min/max 2 & 3

        float qValues[8][4]{}; // q1..q32 cast to float
    } uniformCache;

    void UpdateUniformConstants(const PerFrameContext& perFrameContext);
};

} // namespace MilkdropPreset
} // namespace libprojectM
