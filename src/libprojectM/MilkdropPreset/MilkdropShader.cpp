#include "MilkdropShader.hpp"

#include "PresetState.hpp"
#include "Utils.hpp"

#include <MilkdropStaticShaders.hpp>

#include <GLSLGenerator.h>
#include <HLSLParser.h>
#include <Logging.hpp>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/mat4x4.hpp>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <regex>
#include <set>

namespace libprojectM {
namespace MilkdropPreset {

using libprojectM::MilkdropPreset::MilkdropStaticShaders;

static bool s_hdrPeakModeEnabled = true;
static bool s_fp16PrecisionEnabled = false;
static bool s_curlNoiseEnabled = true;
static bool s_catmullRomEnabled = true;

void MilkdropShader::SetHdrPeakModeEnabled(bool enabled)
{
    s_hdrPeakModeEnabled = enabled;
}

bool MilkdropShader::IsHdrPeakModeEnabled()
{
    return s_hdrPeakModeEnabled;
}

void MilkdropShader::SetFp16PrecisionEnabled(bool enabled)
{
    s_fp16PrecisionEnabled = enabled;
}

bool MilkdropShader::IsFp16PrecisionEnabled()
{
    return s_fp16PrecisionEnabled;
}

void MilkdropShader::SetCurlNoiseEnabled(bool enabled)
{
    s_curlNoiseEnabled = enabled;
}

bool MilkdropShader::IsCurlNoiseEnabled()
{
    return s_curlNoiseEnabled;
}

void MilkdropShader::SetCatmullRomEnabled(bool enabled)
{
    s_catmullRomEnabled = enabled;
}

bool MilkdropShader::IsCatmullRomEnabled()
{
    return s_catmullRomEnabled;
}

auto MilkdropShader::UsesDiscard() const -> bool
{
    return m_usesDiscard;
}

auto MilkdropShader::UsesPolarCoords() const -> bool
{
    return m_usesPolarCoords;
}

auto MilkdropShader::UsesHueShader() const -> bool
{
    return m_usesHueShader;
}

auto MilkdropShader::UsesPerFrameUBO() const -> bool
{
    return m_usesPerFrameUbo;
}

auto MilkdropShader::TranspileToGLSLSource(const std::string& presetShaderCode) -> std::string
{
    LoadCode(presetShaderCode);

    std::set<std::string> samplerDeclarations;
    std::set<std::string> texSizeDeclarations;
    std::vector<std::string> mainSamplerNames;

    for (const auto& name : m_samplerNames)
    {
        std::string baseName = name;
        if (name.length() > 3 && name.at(2) == '_')
        {
            baseName = name.substr(3);
        }
        std::string const lowerBase = Utils::ToLower(baseName);
        if (lowerBase == "main")
        {
            std::string effectiveName = name;
            if (name == "FC_main" || name == "PC_main" || name == "FW_main" || name == "PW_main")
            {
                effectiveName = Utils::ToLower(name);
            }
            samplerDeclarations.insert("uniform sampler2D sampler_" + effectiveName + ";\n");
            texSizeDeclarations.insert("uniform float4 texsize_main;\n");
            mainSamplerNames.push_back("sampler_" + effectiveName);
        }
        else if (lowerBase == "blur1" || lowerBase == "blur2" || lowerBase == "blur3")
        {
            samplerDeclarations.insert("uniform sampler2D sampler_" + lowerBase + ";\n");
        }
        else if (lowerBase.find("noisevol") != std::string::npos)
        {
            samplerDeclarations.insert("uniform sampler3D sampler_" + name + ";\n");
            texSizeDeclarations.insert("uniform float4 texsize_" + baseName + ";\n");
        }
        else
        {
            samplerDeclarations.insert("uniform sampler2D sampler_" + name + ";\n");
            texSizeDeclarations.insert("uniform float4 texsize_" + baseName + ";\n");
        }
    }

    std::string glslResult = GenerateGLSLSource(samplerDeclarations, texSizeDeclarations, mainSamplerNames, m_preprocessedCode);
    UpdateShaderUsageFlags(glslResult);
    return glslResult;
}

static auto floatRand = []() { return static_cast<float>(rand() % 7381) / 7380.0f; };

MilkdropShader::MilkdropShader(ShaderType type)
    : m_type(type)
    , m_randValues({floatRand(), floatRand(), floatRand(), floatRand()})
{
    unsigned int index = 0;
    do
    {
        for (int i = 0; i < 4; i++)
        {
            float const m_randTranslationMult = 1;
            float const rotMult = 0.9f * powf(index / 8.0f, 3.2f);
            m_randTranslation[index].x = (floatRand() * 2 - 1) * m_randTranslationMult;
            m_randTranslation[index].y = (floatRand() * 2 - 1) * m_randTranslationMult;
            m_randTranslation[index].z = (floatRand() * 2 - 1) * m_randTranslationMult;
            m_randRotationCenters[index].x = floatRand() * 6.28f;
            m_randRotationCenters[index].y = floatRand() * 6.28f;
            m_randRotationCenters[index].z = floatRand() * 6.28f;
            m_randRotationSpeeds[index].x = (floatRand() * 2 - 1) * rotMult;
            m_randRotationSpeeds[index].y = (floatRand() * 2 - 1) * rotMult;
            m_randRotationSpeeds[index].z = (floatRand() * 2 - 1) * rotMult;
            index++;
        }
    } while (index < sizeof(m_randTranslation) / sizeof(m_randTranslation[0]));

    m_locC.fill(-1);
    m_locQ.fill(-1);
    m_locRot.fill(-1);
}

MilkdropShader::~MilkdropShader()
{
    if (m_perFrameUbo != 0)
    {
        glDeleteBuffers(1, &m_perFrameUbo);
        m_perFrameUbo = 0;
    }
}

void MilkdropShader::LoadCode(const std::string& presetShaderCode)
{
    m_fragmentShaderCode = presetShaderCode;
    m_preprocessedCode = m_fragmentShaderCode;

    // Apply preprocessor macro expansion first so sampler alias #defines
    // (e.g. #define sampler_pic sampler_cells or #define sampler_pic sampler_rand00)
    // are resolved before sampler discovery scans for active textures.
    M4::Allocator allocator;
    M4::HLSLTree tree(&allocator);
    M4::HLSLParser parser(&allocator, &tree);
    std::string expanded;
    if (parser.ApplyPreprocessor("", m_preprocessedCode.c_str(), m_preprocessedCode.size(), expanded))
    {
        m_preprocessedCode = std::move(expanded);
    }

    GetReferencedSamplers(m_preprocessedCode);
    PreprocessPresetShader(m_preprocessedCode);
}

void MilkdropShader::LoadTexturesAndCompile(PresetState& presetState)
{
    std::locale loc;

    m_mainTextureDescriptors.clear();
    m_textureSamplerDescriptors.clear();

    // Now request the textures and descriptors from the texture manager.
    for (const auto& name : m_samplerNames)
    {
        std::string baseName = name;
        if (name.length() > 3 && name.at(2) == '_')
        {
            baseName = name.substr(3);
        }

        std::string lowerCaseName = Utils::ToLower(baseName);

        // The "main" and "blurX" textures are preset-specific and are not managed by TextureManager.
        if (lowerCaseName == "main")
        {
            std::string effectiveName = name;
            if (name == "FC_main" || name == "PC_main" || name == "FW_main" || name == "PW_main")
            {
                effectiveName = Utils::ToLower(name);
            }
            Renderer::TextureSamplerDescriptor desc(presetState.mainTexture.lock(),
                                                    presetState.renderContext.textureManager->GetSampler(effectiveName),
                                                    effectiveName,
                                                    "main");
            m_mainTextureDescriptors.push_back(std::move(desc));
            continue;
        }

        // A few presets directly use the (undocumented) sampler name.
        if (lowerCaseName == "blur1")
        {
            UpdateMaxBlurLevel(BlurTexture::BlurLevel::Blur1);
            continue;
        }
        if (lowerCaseName == "blur2")
        {
            UpdateMaxBlurLevel(BlurTexture::BlurLevel::Blur2);
            continue;
        }
        if (lowerCaseName == "blur3")
        {
            UpdateMaxBlurLevel(BlurTexture::BlurLevel::Blur3);
            continue;
        }

        // Random textures need special treatment.
        if (lowerCaseName.length() >= 6 &&
            lowerCaseName.substr(0, 4) == "rand" && std::isdigit(lowerCaseName.at(4), loc) && std::isdigit(lowerCaseName.at(5), loc))
        {
            // First look up the random texture index in the preset state so the texture matches between warp and composite shaders
            int randomSlot = -1;
            try
            {
                randomSlot = std::stoi(lowerCaseName.substr(4, 2));
            }
            catch (...) // Ignore any conversion errors.
            {
            }

            if (randomSlot >= 0 && randomSlot <= 15)
            {
                if (presetState.randomTextureDescriptors.find(randomSlot) != presetState.randomTextureDescriptors.end())
                {
                    // Use existing texture descriptor.
                    m_textureSamplerDescriptors.push_back(presetState.randomTextureDescriptors.at(randomSlot));
                    continue;
                }

                // Slot empty, request a new random texture.
                auto desc = presetState.renderContext.textureManager->GetRandomTexture(name);

                // Also store a copy in preset state!
                presetState.randomTextureDescriptors.insert({randomSlot, desc});

                m_textureSamplerDescriptors.push_back(std::move(desc));
                continue;
            }

            // Fall through if slot number is out of range and treat as normal texture.
        }

        auto desc = presetState.renderContext.textureManager->GetTexture(name);
        m_textureSamplerDescriptors.push_back(std::move(desc));
    }

    // Now that we have the textures, transpile the code.
    TranspileHLSLShader(presetState, m_preprocessedCode);

    // Update blur texture level if shader was compiled successfully.
    presetState.blurTexture.SetRequiredBlurLevel(m_maxBlurLevelRequired);
}

void MilkdropShader::LoadVariables(const PresetState& presetState, const PerFrameContext& perFrameContext)
{
    // Ensure uniform constants cache is updated for the current frame
    const_cast<PresetState&>(presetState).UpdateUniformConstants(perFrameContext);

    m_shader.Bind();

    m_shader.SetUniformMat4x4(m_locVertexTransformation, PresetState::orthogonalProjection);

    if (m_usesPerFrameUbo && m_perFrameUbo != 0)
    {
        PerFrameUBOData ubo{};
        if (m_hasRandFrameUniform)
        {
            ubo.rand_frame[0] = floatRand();
            ubo.rand_frame[1] = floatRand();
            ubo.rand_frame[2] = floatRand();
            ubo.rand_frame[3] = floatRand();
        }
        std::memcpy(ubo.rand_preset, m_randValues.data(), sizeof(ubo.rand_preset));
        std::memcpy(ubo.c[0], presetState.uniformCache.c0, sizeof(float) * 4);
        ubo.c[1][0] = (m_type == ShaderType::CompositeShader && s_hdrPeakModeEnabled) ? 1.0f : 0.0f;
        ubo.c[1][1] = 0.0f;
        ubo.c[1][2] = 0.0f;
        ubo.c[1][3] = 0.0f;
        std::memcpy(ubo.c[2], presetState.uniformCache.c2, sizeof(float) * 4);
        std::memcpy(ubo.c[3], presetState.uniformCache.c3, sizeof(float) * 4);
        std::memcpy(ubo.c[4], presetState.uniformCache.c4, sizeof(float) * 4);
        std::memcpy(ubo.c[5], presetState.uniformCache.c5, sizeof(float) * 4);
        std::memcpy(ubo.c[6], presetState.uniformCache.c6, sizeof(float) * 4);
        std::memcpy(ubo.c[7], presetState.uniformCache.c7, sizeof(float) * 4);
        if (m_hasFastTrigUniforms)
        {
            std::memcpy(ubo.c[8], presetState.uniformCache.c8, sizeof(float) * 4);
            std::memcpy(ubo.c[9], presetState.uniformCache.c9, sizeof(float) * 4);
        }
        if (m_hasSlowTrigUniforms)
        {
            std::memcpy(ubo.c[10], presetState.uniformCache.c10, sizeof(float) * 4);
            std::memcpy(ubo.c[11], presetState.uniformCache.c11, sizeof(float) * 4);
        }
        std::memcpy(ubo.c[12], presetState.uniformCache.c12, sizeof(float) * 4);
        std::memcpy(ubo.c[13], presetState.uniformCache.c13, sizeof(float) * 4);
        std::memcpy(ubo.q, presetState.uniformCache.qValues, sizeof(ubo.q));

        glBindBuffer(GL_UNIFORM_BUFFER, m_perFrameUbo);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(PerFrameUBOData), &ubo);
        glBindBufferBase(GL_UNIFORM_BUFFER, 0, m_perFrameUbo);
        glBindBuffer(GL_UNIFORM_BUFFER, 0);
    }
    else
    {
        if (m_locRandFrame >= 0)
        {
            float const randFrame[4] = {floatRand(), floatRand(), floatRand(), floatRand()};
            m_shader.SetUniformFloat4(m_locRandFrame, randFrame);
        }
        if (m_locRandPreset >= 0)
        {
            m_shader.SetUniformFloat4(m_locRandPreset, m_randValues.data());
        }

        if (m_locC[0] >= 0) m_shader.SetUniformFloat4(m_locC[0], presetState.uniformCache.c0);
        if (m_locC[1] >= 0)
        {
            float const c1[4] = {(m_type == ShaderType::CompositeShader && s_hdrPeakModeEnabled) ? 1.0f : 0.0f,
                                 0.0f,
                                 0.0f,
                                 0.0f};
            m_shader.SetUniformFloat4(m_locC[1], c1);
        }
        if (m_locC[2] >= 0) m_shader.SetUniformFloat4(m_locC[2], presetState.uniformCache.c2);
        if (m_locC[3] >= 0) m_shader.SetUniformFloat4(m_locC[3], presetState.uniformCache.c3);
        if (m_locC[4] >= 0) m_shader.SetUniformFloat4(m_locC[4], presetState.uniformCache.c4);
        if (m_locC[5] >= 0) m_shader.SetUniformFloat4(m_locC[5], presetState.uniformCache.c5);
        if (m_locC[6] >= 0) m_shader.SetUniformFloat4(m_locC[6], presetState.uniformCache.c6);
        if (m_locC[7] >= 0) m_shader.SetUniformFloat4(m_locC[7], presetState.uniformCache.c7);

        if (m_hasFastTrigUniforms)
        {
            if (m_locC[8] >= 0) m_shader.SetUniformFloat4(m_locC[8], presetState.uniformCache.c8);
            if (m_locC[9] >= 0) m_shader.SetUniformFloat4(m_locC[9], presetState.uniformCache.c9);
        }

        if (m_hasSlowTrigUniforms)
        {
            if (m_locC[10] >= 0) m_shader.SetUniformFloat4(m_locC[10], presetState.uniformCache.c10);
            if (m_locC[11] >= 0) m_shader.SetUniformFloat4(m_locC[11], presetState.uniformCache.c11);
        }

        if (m_locC[12] >= 0) m_shader.SetUniformFloat4(m_locC[12], presetState.uniformCache.c12);
        if (m_locC[13] >= 0) m_shader.SetUniformFloat4(m_locC[13], presetState.uniformCache.c13);

        for (size_t i = 0; i < 8; ++i)
        {
            if (m_locQ[i] >= 0)
            {
                m_shader.SetUniformFloat4(m_locQ[i], presetState.uniformCache.qValues[i]);
            }
        }
    }

    // Lazy 3D Rotations: ONLY compute if preset shader actually references any rot_* uniform
    if (m_has3DRotationUniforms)
    {
        auto floatTime = static_cast<float>(presetState.renderContext.time);
        std::array<glm::mat4, 24> tempMatrices{};

        // write matrices
        for (int i = 0; i < 20; i++)
        {
            glm::mat4 const rotationX = glm::rotate(glm::mat4(1.0f), m_randRotationCenters[i].x + m_randRotationSpeeds[i].x * floatTime, glm::vec3(1.0f, 0.0f, 0.0f));
            glm::mat4 const rotationY = glm::rotate(glm::mat4(1.0f), m_randRotationCenters[i].y + m_randRotationSpeeds[i].y * floatTime, glm::vec3(0.0f, 1.0f, 0.0f));
            glm::mat4 const rotationZ = glm::rotate(glm::mat4(1.0f), m_randRotationCenters[i].z + m_randRotationSpeeds[i].z * floatTime, glm::vec3(0.0f, 0.0f, 1.0f));

            glm::mat4 const randomTranslation = glm::translate(glm::mat4(1.0f), glm::vec3(m_randTranslation[i].x, m_randTranslation[i].y, m_randTranslation[i].z));

            tempMatrices[i] = randomTranslation * rotationX;
            tempMatrices[i] = rotationZ * tempMatrices[i];
            tempMatrices[i] = rotationY * tempMatrices[i];
        }

        // the last 4 are totally random, each frame
        for (int i = 20; i < 24; i++)
        {
            glm::mat4 const rotationX = glm::rotate(glm::mat4(1.0f), floatRand() * 6.28f, glm::vec3(1.0f, 0.0f, 0.0f));
            glm::mat4 const rotationY = glm::rotate(glm::mat4(1.0f), floatRand() * 6.28f, glm::vec3(0.0f, 1.0f, 0.0f));
            glm::mat4 const rotationZ = glm::rotate(glm::mat4(1.0f), floatRand() * 6.28f, glm::vec3(0.0f, 0.0f, 1.0f));

            glm::mat4 const randomTranslation = glm::translate(glm::mat4(1.0f), glm::vec3(floatRand(), floatRand(), floatRand()));

            tempMatrices[i] = randomTranslation * rotationX;
            tempMatrices[i] = rotationZ * tempMatrices[i];
            tempMatrices[i] = rotationY * tempMatrices[i];
        }

        for (size_t i = 0; i < 24; ++i)
        {
            if (m_locRot[i] >= 0)
            {
                m_shader.SetUniformMat3x4(m_locRot[i], tempMatrices[i]);
            }
        }
    }

    m_shader.SetUniformInt(m_locCurlNoise, s_curlNoiseEnabled ? 1 : 0);
    m_shader.SetUniformInt(m_locCatmullRom, s_catmullRomEnabled ? 1 : 0);

    // Bind all texture and sampler descriptors. This includes the main and blur textures.
    GLint textureUnit{0};
    for (auto& desc : m_mainTextureDescriptors)
    {
        // Update main texture, swaps every frame.
        desc.Texture(presetState.mainTexture);
        desc.Bind(textureUnit, m_shader);
        textureUnit++;
    }
    presetState.blurTexture.Bind(textureUnit, m_shader);
    for (auto& desc : m_textureSamplerDescriptors)
    {
        if (desc.Empty())
        {
            desc.TryUpdate(*presetState.renderContext.textureManager);
        }
        desc.Bind(textureUnit, m_shader);
        textureUnit++;
    }
}

auto MilkdropShader::Shader() -> Renderer::Shader&
{
    return m_shader;
}

void MilkdropShader::PreprocessPresetShader(std::string& program)
{
    std::string shaderTypeString = "composite";
    if (m_type == ShaderType::WarpShader)
    {
        shaderTypeString = "warp";
    }

    if (program.length() <= 0)
    {
        throw Renderer::ShaderException("[MilkdropShader] Preset " + shaderTypeString + " shader is declared, but empty.");
    }

    size_t found;

    // Find "sampler_state" overrides and blank them out in-place (with spaces),
    // preserving exact character positions between 'stripped' and 'program'
    // so any functions or code defined before 'shader_body' remain synchronized and uncorrupted.
    std::string stripped = Utils::StripComments(program);
    size_t samplerPos = 0;
    while ((samplerPos = stripped.find("sampler_state", samplerPos)) != std::string::npos)
    {
        // Search backwards for declaration start ("sampler", "sampler2D", etc.)
        // or up to previous semicolon or brace or start of string
        size_t declStart = stripped.rfind(';', samplerPos);
        if (declStart == std::string::npos)
        {
            declStart = 0;
        }
        else
        {
            declStart++; // skip ';'
        }
        size_t braceStart = stripped.find_last_of("{}", samplerPos);
        if (braceStart != std::string::npos && braceStart >= declStart)
        {
            declStart = braceStart + 1;
        }

        // Now search forward from samplerPos for '{' and matching '}' followed by optional ';'
        size_t endPos = samplerPos + 13; // length of "sampler_state"
        size_t openBrace = stripped.find('{', samplerPos);
        size_t semiBeforeBrace = stripped.find(';', samplerPos);
        if (openBrace != std::string::npos && (semiBeforeBrace == std::string::npos || openBrace < semiBeforeBrace))
        {
            int depth = 1;
            size_t k = openBrace + 1;
            while (k < stripped.length() && depth > 0)
            {
                if (stripped[k] == '{') depth++;
                else if (stripped[k] == '}') depth--;
                k++;
            }
            endPos = k;
            while (endPos < stripped.length() && (stripped[endPos] == ';' || stripped[endPos] == ' ' || stripped[endPos] == '\t' || stripped[endPos] == '\r'))
            {
                endPos++;
            }
        }
        else if (semiBeforeBrace != std::string::npos)
        {
            endPos = semiBeforeBrace + 1;
        }

        // Blank out in BOTH program and stripped with spaces, preserving newlines
        for (size_t i = declStart; i < endPos && i < program.length(); ++i)
        {
            if (program[i] != '\n') program[i] = ' ';
            if (stripped[i] != '\n') stripped[i] = ' ';
        }

        samplerPos = endPos;
    }

    // replace shader_body with entry point function
    // Use the stripped copy so a commented-out shader_body is not matched.
    found = stripped.find("shader_body");
    if (found != std::string::npos)
    {
        if (m_type == ShaderType::WarpShader)
        {
            program.replace(int(found), 11, R"(
void PS(float4 _vDiffuse : COLOR,
        float4 _uv : TEXCOORD0,
        float2 _rad_ang : TEXCOORD1,
        out float4 _return_value : COLOR0,
        out float4 _mv_tex_coords : COLOR1)
)");
        }
        else
        {
            program.replace(int(found), 11, R"(
void PS(float4 _vDiffuse : COLOR,
        float2 _uv : TEXCOORD0,
        float2 _rad_ang : TEXCOORD1,
        out float4 _return_value : COLOR)
)");
        }
    }
    else
    {
        LOG_DEBUG("[MilkdropShader] Failed " + shaderTypeString + " shader code:\n" + program);
        throw Renderer::ShaderException("[MilkdropShader] Preset " + shaderTypeString + " shader is missing \"shader_body\" entry point.");
    }

    // replace the "{" immediately following shader_body with some variable declarations
    found = program.find('{', found);
    if (found != std::string::npos)
    {
        std::string progMain = "{\nfloat3 ret = 0;\n";
        if (m_type == ShaderType::WarpShader)
        {
            progMain.append("_mv_tex_coords.xy = _uv.xy;\n");
        }
        program.replace(int(found), 1, progMain);
    }
    else
    {
        LOG_DEBUG("[MilkdropShader] Failed " + shaderTypeString + " shader code:\n" + program);
        throw Renderer::ShaderException("[MilkdropShader] Preset " + shaderTypeString + " shader has no opening braces.");
    }

    // Find matching closing brace and cut off excess text after shader's main function
    int bracesOpen = 1;
    size_t pos = found + 1;
    for (; pos < program.length() && bracesOpen > 0; ++pos)
    {
        switch (program.at(pos))
        {
            case '/':
                // Skip line comments until EoL to prevent false counting
                if (pos < program.length() - 1 && program.at(pos + 1) == '/')
                {
                    for (; pos < program.length(); ++pos)
                    {
                        if (program.at(pos) == '\n')
                        {
                            break;
                        }
                    }
                }
                // Skip block comments to prevent false counting
                else if (pos < program.length() - 1 && program.at(pos + 1) == '*')
                {
                    pos += 2;
                    for (; pos < program.length() - 1; ++pos)
                    {
                        if (program.at(pos) == '*' && program.at(pos + 1) == '/')
                        {
                            ++pos; // skip past '/'
                            break;
                        }
                    }
                }
                continue;

            case '{':
                bracesOpen++;
                continue;

            case '}':
                bracesOpen--;
        }
    }

    std::string returnCode = "_return_value = float4(saturate(ret.xyz), 1.0);\n}\n";

    if (bracesOpen == 0)
    {
        program.replace(pos - 1, 1, returnCode);
        if (pos - 1 + returnCode.length() < program.length())
        {
            program.resize(pos - 1 + returnCode.length());
        }
    }
    else
    {
        found = program.rfind('}');
        if (found != std::string::npos)
        {
            program.replace(int(found), 1, returnCode);
            if (found + returnCode.length() < program.length())
            {
                program.resize(found + returnCode.length());
            }
        }
        else
        {
            LOG_DEBUG("[MilkdropShader] Failed " + shaderTypeString + " shader code:\n" + program);
            throw Renderer::ShaderException("[MilkdropShader] Preset " + shaderTypeString + " shader has no closing brace.");
        }
    }

    std::string fullSource; //!< Full shader source before translation, includes all uniforms etc.

    // First copy the generic "header" into the shader. Includes uniforms and some defines
    // to unwrap the packed 4-element uniforms into single values.
    fullSource.append(MilkdropStaticShaders::Get()->GetPresetShaderHeader());

    if (m_type == ShaderType::WarpShader)
    {
        fullSource.append("#define rad _rad_ang.x\n"
                          "#define ang _rad_ang.y\n"
                          "#define uv _uv.xy\n"
                          "#define uv_orig _uv.zw\n");
    }
    else
    {
        fullSource.append("#define rad _rad_ang.x\n"
                          "#define ang _rad_ang.y\n"
                          "#define uv _uv.xy\n"
                          "#define uv_orig _uv.xy\n"
                          "#define hue_shader _vDiffuse.xyz\n");
    }

    fullSource.append(program);

    program = fullSource;
}

void MilkdropShader::GetReferencedSamplers(const std::string& program)
{
    // Look up samplers referenced in the shader program
    m_samplerNames.clear();

    // "main" should always be present.
    m_samplerNames.insert("main");

    // Strip comments so that commented-out sampler/texsize declarations are not matched.
    std::string const stripped = Utils::StripComments(program);

    // Search for sampler usage
    auto found = stripped.find("sampler_", 0);
    while (found != std::string::npos)
    {
        found += 8;
        size_t const end = stripped.find_first_of(" ;,\n\r)", found);

        if (end != std::string::npos)
        {
            std::string sampler = stripped.substr(static_cast<int>(found), static_cast<int>(end - found));
            while (sampler.rfind("sampler_", 0) == 0)
            {
                sampler = sampler.substr(8);
            }
            // Skip "sampler_state", as it's a reserved word and not a sampler.
            if (!sampler.empty() && sampler != "state")
            {
                m_samplerNames.insert(sampler);
            }
        }

        found = stripped.find("sampler_", found);
    }

    // Also search for texsize usage, some presets don't reference the sampler.
    found = stripped.find("texsize_", 0);
    while (found != std::string::npos)
    {
        found += 8;
        size_t const end = stripped.find_first_of(" ;,.\n\r)", found);

        if (end != std::string::npos)
        {
            std::string const sampler = stripped.substr(static_cast<int>(found), static_cast<int>(end - found));
            m_samplerNames.insert(sampler);
        }

        found = stripped.find("texsize_", found);
    }

    {
        // Remove duplicate mentions or "randXX" names, keeping the long forms only (first one will determine the actual texture loaded).
        auto samplerName = m_samplerNames.begin();
        std::locale loc;
        while (samplerName != m_samplerNames.end())
        {
            std::string lowerCaseName = Utils::ToLower(*samplerName);
            if (lowerCaseName.length() == 6 &&
                lowerCaseName.substr(0, 4) == "rand" && std::isdigit(lowerCaseName.at(4), loc) && std::isdigit(lowerCaseName.at(5), loc))
            {
                auto additionalName = samplerName;
                additionalName++;
                if (additionalName != m_samplerNames.end())
                {
                    std::string addLowerCaseName = Utils::ToLower(*additionalName);
                    if (addLowerCaseName.length() > 7 &&
                        addLowerCaseName.substr(0, 6) == lowerCaseName &&
                        addLowerCaseName[6] == '_')
                    {
                        samplerName = m_samplerNames.erase(samplerName);
                    }
                }
            }
            samplerName++;
        }
    }

    if (stripped.find("GetBlur3") != std::string::npos)
    {
        UpdateMaxBlurLevel(BlurTexture::BlurLevel::Blur3);
    }
    else if (stripped.find("GetBlur2") != std::string::npos)
    {
        UpdateMaxBlurLevel(BlurTexture::BlurLevel::Blur2);
    }
    else if (stripped.find("GetBlur1") != std::string::npos)
    {
        UpdateMaxBlurLevel(BlurTexture::BlurLevel::Blur1);
    }
    else
    {
        m_maxBlurLevelRequired = BlurTexture::BlurLevel::None;
    }
}

auto MilkdropShader::GenerateGLSLSource(const std::set<std::string>& samplerDeclarations,
                                        const std::set<std::string>& texSizeDeclarations,
                                        const std::vector<std::string>& mainSamplerNames,
                                        std::string program) const -> std::string
{
    std::string shaderTypeString = "composite";
    if (m_type == ShaderType::WarpShader)
    {
        shaderTypeString = "warp";
    }

    M4::GLSLGenerator generator;
    M4::Allocator allocator;

    M4::HLSLTree tree(&allocator);
    M4::HLSLParser parser(&allocator, &tree);

    // Preprocess define macros
    std::string sourcePreprocessed;
    if (!parser.ApplyPreprocessor("", program.c_str(), program.size(), sourcePreprocessed))
    {
        LOG_DEBUG("[MilkdropShader] Failed " + shaderTypeString + " shader code:\n" + program);
        throw Renderer::ShaderException("Error translating HLSL " + shaderTypeString + " shader: Preprocessing failed.");
    }

    // Remove previous shader and sampler declarations using static precompiled regexes
    static const std::regex s_reSamplerDecl(
        R"((?:uniform\s+)?(?:sampler(?:1D|2D|3D|CUBE)?|Texture(?:1D|2D|3D))\s+[\w\d_]+(?:\s*:\s*register\s*\([^\)]*\))?(?:\s*=\s*sampler_state\s*\{[\s\S]*?\}\s*;?|\s*=\s*sampler_state\s*;?|\s*;?))");
    static const std::regex s_reSamplerState(
        R"((?:SamplerState\s+[\w\d_]+(?:\s*:\s*register\s*\([^\)]*\))?\s*;?|sampler_state\s*\{[\s\S]*?\}\s*;?))");
    static const std::regex s_reSamplerCall(
        R"(sampler(?:1D|2D|3D|CUBE|)(\s+|\().*)");
    static const std::regex s_reTexSizeDecl(
        R"((?:uniform\s+)?(?:float[234]|vec[234])\s+texsize_[^;\n]*;?)");

    std::smatch matches;
    while (std::regex_search(sourcePreprocessed, matches, s_reSamplerDecl))
    {
        sourcePreprocessed.replace(matches.position(), matches.length(), "");
    }
    while (std::regex_search(sourcePreprocessed, matches, s_reSamplerState))
    {
        sourcePreprocessed.replace(matches.position(), matches.length(), "");
    }
    while (std::regex_search(sourcePreprocessed, matches, s_reSamplerCall))
    {
        sourcePreprocessed.replace(matches.position(), matches.length(), "");
    }
    while (std::regex_search(sourcePreprocessed, matches, s_reTexSizeDecl))
    {
        sourcePreprocessed.replace(matches.position(), matches.length(), "");
    }

    // Now insert them on top.
    for (const auto& texSizeDeclaration : texSizeDeclarations)
    {
        sourcePreprocessed.insert(0, texSizeDeclaration);
    }
    for (const auto& samplerDeclaration : samplerDeclarations)
    {
        sourcePreprocessed.insert(0, samplerDeclaration);
    }

    // Transpile from HLSL (aka preset shader aka DirectX shader) to GLSL (aka OpenGL shader lang)
    // First, parse HLSL into a tree
    if (!parser.Parse("", sourcePreprocessed.c_str(), sourcePreprocessed.size()))
    {
        LOG_DEBUG("[MilkdropShader] Failed " + shaderTypeString + " shader code:\n" + program);
        LOG_DEBUG("[MilkdropShader] Failed preprocessed " + shaderTypeString + " shader code:\n" + sourcePreprocessed);
        throw Renderer::ShaderException("[MilkdropShader] Error translating HLSL " + shaderTypeString + " shader: HLSL parsing failed.");
    }

    // Then generate GLSL from the resulting parser tree
    if (!generator.Generate(&tree, M4::GLSLGenerator::Target_FragmentShader,
                            MilkdropStaticShaders::Get()->GetGlslGeneratorVersion(),
                            "PS", M4::GLSLGenerator::Options(M4::GLSLGenerator::Flag_AlternateNanPropagation)))
    {
        LOG_DEBUG("[MilkdropShader] Failed " + shaderTypeString + " shader code:\n" + program);
        LOG_DEBUG("[MilkdropShader] Failed preprocessed " + shaderTypeString + " shader code:\n" + sourcePreprocessed);
        throw Renderer::ShaderException("[MilkdropShader] Error translating HLSL " + shaderTypeString + " shader: GLSL generating failed.\nSource:\n" + sourcePreprocessed);
    }

    std::string glslResult = generator.GetResult();
    if (s_fp16PrecisionEnabled)
    {
        const std::string highpFloat = "precision highp float;";
        const std::string medpFloat = "precision mediump float;";
        size_t pos = glslResult.find(highpFloat);
        if (pos != std::string::npos)
        {
            glslResult.replace(pos, highpFloat.length(), medpFloat);
        }

        const std::string highpSampler = "precision highp sampler3D;";
        const std::string medpSampler = "precision mediump sampler3D;";
        pos = glslResult.find(highpSampler);
        if (pos != std::string::npos)
        {
            glslResult.replace(pos, highpSampler.length(), medpSampler);
        }
        LOG_INFO("[MilkdropShader] Dynamic FP16 (mediump) precision engaged for " + shaderTypeString + " shader");
    }

    std::set<std::string> const uniqueMainSamplers(mainSamplerNames.begin(), mainSamplerNames.end());

    auto replaceAllOccurrences = [](std::string& text, const std::string& from, const std::string& to) -> size_t {
        size_t count = 0;
        size_t pos = 0;
        while ((pos = text.find(from, pos)) != std::string::npos)
        {
            text.replace(pos, from.length(), to);
            pos += to.length();
            ++count;
        }
        return count;
    };

    // GLSLGenerator::ChooseUniqueName appends a numeric suffix (e.g. tex2Dlod0, tex2Dbias0, tex2Dgrad0).
    auto replaceIntrinsicCall = [](std::string& text, const std::string& baseFunc, const std::string& samplerArg, const std::string& replacementFunc) -> size_t {
        size_t count = 0;
        size_t pos = 0;
        const std::string argSuffix = "(" + samplerArg + ",";
        while ((pos = text.find(baseFunc, pos)) != std::string::npos)
        {
            // Ensure word boundary before baseFunc
            if (pos > 0 && (std::isalnum(static_cast<unsigned char>(text[pos - 1])) || text[pos - 1] == '_'))
            {
                pos += baseFunc.length();
                continue;
            }
            size_t digitEnd = pos + baseFunc.length();
            while (digitEnd < text.length() && std::isdigit(static_cast<unsigned char>(text[digitEnd])))
            {
                ++digitEnd;
            }
            if (text.compare(digitEnd, argSuffix.length(), argSuffix) == 0)
            {
                const std::string fullTo = replacementFunc + "(";
                const size_t fullLen = (digitEnd - pos) + argSuffix.length();
                text.replace(pos, fullLen, fullTo);
                pos += fullTo.length();
                ++count;
            }
            else
            {
                pos = digitEnd;
            }
        }
        return count;
    };

    // Determine whether 1..2 tap Catmull-Rom 4-tap bicubic advection applies to sampler_main in WarpShader.
    size_t mainTapCount = 0;
    if (s_catmullRomEnabled && m_type == ShaderType::WarpShader)
    {
        const std::string target = "texture(sampler_main,";
        size_t scanPos = 0;
        while ((scanPos = glslResult.find(target, scanPos)) != std::string::npos)
        {
            ++mainTapCount;
            scanPos += target.length();
        }
    }
    const bool injectBicubicMain = (s_catmullRomEnabled && m_type == ShaderType::WarpShader && mainTapCount >= 1 && mainTapCount <= 2);

    auto hasIdentifier = [](const std::string& text, const char* ident) -> bool {
        const size_t identLen = std::strlen(ident);
        size_t pos = 0;
        while ((pos = text.find(ident, pos)) != std::string::npos)
        {
            const bool leftOk = (pos == 0) ||
                                (!std::isalnum(static_cast<unsigned char>(text[pos - 1])) && text[pos - 1] != '_');
            const size_t after = pos + identLen;
            const bool rightOk = (after >= text.size()) ||
                                 (!std::isalnum(static_cast<unsigned char>(text[after])) && text[after] != '_');
            if (leftOk && rightOk)
            {
                return true;
            }
            pos += identLen;
        }
        return false;
    };

    // Strip unreferenced 3D rotation matrix declarations emitted unconditionally from PresetShaderHeader
    static constexpr const char* kRotUniformNames[24] = {
        "rot_s1", "rot_s2", "rot_s3", "rot_s4",
        "rot_d1", "rot_d2", "rot_d3", "rot_d4",
        "rot_f1", "rot_f2", "rot_f3", "rot_f4",
        "rot_vf1", "rot_vf2", "rot_vf3", "rot_vf4",
        "rot_uf1", "rot_uf2", "rot_uf3", "rot_uf4",
        "rot_rand1", "rot_rand2", "rot_rand3", "rot_rand4"
    };
    for (const char* rotName : kRotUniformNames)
    {
        const std::string declLine = std::string("uniform mat3x4 ") + rotName + ";\n";
        size_t const pos = glslResult.find(declLine);
        if (pos != std::string::npos)
        {
            glslResult.erase(pos, declLine.length());
            if (hasIdentifier(glslResult, rotName))
            {
                glslResult.insert(pos, declLine);
            }
        }
    }

    // Pack referenced per-frame vec4 uniforms into a compact 384-byte std140 PerFrameBlock UBO
    // while omitting the block completely if the shader uses zero per-frame vec4 constants.
    const auto glslVersion = MilkdropStaticShaders::Get()->GetGlslGeneratorVersion();
    const bool supportsUbo = (glslVersion == M4::GLSLGenerator::Version_140 ||
                              glslVersion == M4::GLSLGenerator::Version_150 ||
                              glslVersion == M4::GLSLGenerator::Version_330 ||
                              glslVersion == M4::GLSLGenerator::Version_300_ES);

    bool emitPerFrameUbo = false;
    if (supportsUbo)
    {
        static constexpr const char* kPerFrameVec4Names[24] = {
            "rand_frame", "rand_preset",
            "_c0", "_c1", "_c2", "_c3", "_c4", "_c5", "_c6",
            "_c7", "_c8", "_c9", "_c10", "_c11", "_c12", "_c13",
            "_qa", "_qb", "_qc", "_qd", "_qe", "_qf", "_qg", "_qh"
        };

        for (const char* varName : kPerFrameVec4Names)
        {
            const std::string declWithNewline = std::string("uniform vec4 ") + varName + ";\n";
            size_t pos = glslResult.find(declWithNewline);
            if (pos != std::string::npos)
            {
                glslResult.erase(pos, declWithNewline.length());
                continue;
            }
            const std::string declPlain = std::string("uniform vec4 ") + varName + ";";
            pos = glslResult.find(declPlain);
            if (pos != std::string::npos)
            {
                glslResult.erase(pos, declPlain.length());
            }
        }

        bool usesAnyPerFrameVec4 = injectBicubicMain;
        if (!usesAnyPerFrameVec4)
        {
            for (const char* varName : kPerFrameVec4Names)
            {
                if (hasIdentifier(glslResult, varName))
                {
                    usesAnyPerFrameVec4 = true;
                    break;
                }
            }
        }

        if (usesAnyPerFrameVec4)
        {
            emitPerFrameUbo = true;
            static const std::string s_perFrameBlockDecl = R"(layout(std140) uniform PerFrameBlock {
    vec4 rand_frame;
    vec4 rand_preset;
    vec4 _c0;
    vec4 _c1;
    vec4 _c2;
    vec4 _c3;
    vec4 _c4;
    vec4 _c5;
    vec4 _c6;
    vec4 _c7;
    vec4 _c8;
    vec4 _c9;
    vec4 _c10;
    vec4 _c11;
    vec4 _c12;
    vec4 _c13;
    vec4 _qa;
    vec4 _qb;
    vec4 _qc;
    vec4 _qd;
    vec4 _qe;
    vec4 _qf;
    vec4 _qg;
    vec4 _qh;
};
)";
            size_t blockInsertPos = std::string::npos;
            size_t const fragOutPos = glslResult.find("out vec4 rast_FragData[");
            if (fragOutPos != std::string::npos)
            {
                size_t const lineEnd = glslResult.find('\n', fragOutPos);
                if (lineEnd != std::string::npos)
                {
                    blockInsertPos = lineEnd + 1;
                }
            }
            if (blockInsertPos == std::string::npos)
            {
                blockInsertPos = glslResult.find("void PS(");
            }
            if (blockInsertPos != std::string::npos)
            {
                glslResult.insert(blockInsertPos, s_perFrameBlockDecl);
            }
        }
    }

    // Zero-Flip UV Coordinate Scheme:
    // Both m_previousFrameBuffer and m_currentFrameBuffer are stored right-side up in OpenGL FBO orientation (O_GL).
    // Replacing main-texture lookups sample(S, uv) with sample(S, vec2(uv.x, 1.0 - uv.y)) satisfies the exact
    // texel-center identity with CopyTexture(..., flipVertical=true) while eliminating 2..3 fullscreen FBO copies per frame.
    std::string zeroFlipHelpers;
    if (injectBicubicMain)
    {
        if (!emitPerFrameUbo && glslResult.find("uniform vec4 _c7;") == std::string::npos && glslResult.find("_c7") == std::string::npos)
        {
            zeroFlipHelpers += "uniform vec4 _c7;\n";
        }
        zeroFlipHelpers += R"(
vec4 sampleCatmullRom4Tap(sampler2D tex, vec2 uv) {
    vec2 ts = _c7.xy;
    vec2 its = _c7.zw;
    vec2 sp = uv * ts;
    vec2 tc = floor(sp - 0.5) + 0.5;
    vec2 f = sp - tc;
    vec2 f2 = f * f;
    vec2 f3 = f2 * f;
    vec2 w0 = f2 - 0.5 * (f3 + f);
    vec2 w1 = 1.5 * f3 - 2.5 * f2 + 1.0;
    vec2 w3 = 0.5 * (f3 - f2);
    vec2 w2 = 1.0 - w0 - w1 - w3;
    vec2 s0 = w0 + w1;
    vec2 s1 = w2 + w3;
    vec2 f0 = w1 / max(s0, vec2(1e-5));
    vec2 f1 = w3 / max(s1, vec2(1e-5));
    vec2 t0 = (tc - 1.0 + f0) * its;
    vec2 t1 = (tc + 1.0 + f1) * its;
    return (texture(tex, vec2(t0.x, t0.y)) * (s0.x * s0.y) +
            texture(tex, vec2(t1.x, t0.y)) * (s1.x * s0.y) +
            texture(tex, vec2(t0.x, t1.y)) * (s0.x * s1.y) +
            texture(tex, vec2(t1.x, t1.y)) * (s1.x * s1.y));
}
)";
        LOG_INFO("[MilkdropShader] Catmull-Rom 4-Tap bicubic advection injected for warp shader");
    }

    for (const auto& samp : uniqueMainSamplers)
    {
        if (glslResult.find("uniform sampler2D " + samp + ";") == std::string::npos)
        {
            continue;
        }

        const size_t texReplaced = replaceAllOccurrences(glslResult, "texture(" + samp + ",", "_pm_texMain_" + samp + "(");
        const size_t lodReplaced = replaceIntrinsicCall(glslResult, "tex2Dlod", samp, "_pm_texMainLod_" + samp);
        const size_t biasReplaced = replaceIntrinsicCall(glslResult, "tex2Dbias", samp, "_pm_texMainBias_" + samp);
        const size_t gradReplaced = replaceIntrinsicCall(glslResult, "tex2Dgrad", samp, "_pm_texMainGrad_" + samp);
        const size_t projReplaced = replaceAllOccurrences(glslResult, "textureProj(" + samp + ",", "_pm_texMainProj_" + samp + "(");

        if (texReplaced > 0)
        {
            if (injectBicubicMain && samp == "sampler_main")
            {
                zeroFlipHelpers += "vec4 _pm_texMain_" + samp + "(vec2 _u) { return sampleCatmullRom4Tap(" + samp + ", vec2(_u.x, 1.0 - _u.y)); }\n";
            }
            else
            {
                zeroFlipHelpers += "vec4 _pm_texMain_" + samp + "(vec2 _u) { return texture(" + samp + ", vec2(_u.x, 1.0 - _u.y)); }\n";
            }
            zeroFlipHelpers += "vec4 _pm_texMain_" + samp + "(vec2 _u, float _b) { return texture(" + samp + ", vec2(_u.x, 1.0 - _u.y), _b); }\n";
        }
        if (lodReplaced > 0)
        {
            zeroFlipHelpers += "vec4 _pm_texMainLod_" + samp + "(vec4 _t) { return textureLod(" + samp + ", vec2(_t.x, 1.0 - _t.y), _t.w); }\n";
            zeroFlipHelpers += "vec4 _pm_texMainLod_" + samp + "(vec3 _t) { return textureLod(" + samp + ", vec2(_t.x, 1.0 - _t.y), _t.z); }\n";
            zeroFlipHelpers += "vec4 _pm_texMainLod_" + samp + "(vec2 _t, float _l) { return textureLod(" + samp + ", vec2(_t.x, 1.0 - _t.y), _l); }\n";
        }
        if (biasReplaced > 0)
        {
            zeroFlipHelpers += "vec4 _pm_texMainBias_" + samp + "(vec4 _t) { return texture(" + samp + ", vec2(_t.x, 1.0 - _t.y), _t.w); }\n";
        }
        if (gradReplaced > 0)
        {
            zeroFlipHelpers += "vec4 _pm_texMainGrad_" + samp + "(vec2 _t, vec2 _dx, vec2 _dy) { return textureGrad(" + samp + ", vec2(_t.x, 1.0 - _t.y), vec2(_dx.x, -_dx.y), vec2(_dy.x, -_dy.y)); }\n";
        }
        if (projReplaced > 0)
        {
            zeroFlipHelpers += "vec4 _pm_texMainProj_" + samp + "(vec3 _p) { return textureProj(" + samp + ", vec3(_p.x, _p.z - _p.y, _p.z)); }\n";
            zeroFlipHelpers += "vec4 _pm_texMainProj_" + samp + "(vec4 _p) { return textureProj(" + samp + ", vec4(_p.x, _p.w - _p.y, _p.z, _p.w)); }\n";
        }
    }

    if (!zeroFlipHelpers.empty())
    {
        // Insert right after the last global uniform declaration so any user helper functions
        // declared before PS() as well as PS() and main() have full visibility of the wrappers.
        size_t insertPos = std::string::npos;
        size_t searchPos = 0;
        while ((searchPos = glslResult.find("\nuniform ", searchPos)) != std::string::npos)
        {
            size_t const lineEnd = glslResult.find('\n', searchPos + 1);
            if (lineEnd != std::string::npos)
            {
                insertPos = lineEnd + 1;
                searchPos = lineEnd;
            }
            else
            {
                break;
            }
        }
        if (insertPos == std::string::npos)
        {
            insertPos = glslResult.find("void PS(");
        }
        if (insertPos == std::string::npos)
        {
            insertPos = glslResult.find("void main()");
        }
        if (insertPos != std::string::npos)
        {
            glslResult.insert(insertPos, zeroFlipHelpers);
        }
    }

    LOG_TRACE("[MilkdropShader] Transpiled GLSL " + shaderTypeString + " shader code:\n" + glslResult);
    return glslResult;
}

void MilkdropShader::TranspileHLSLShader(const PresetState& presetState, std::string& program)
{
    // Collect unique samplers and texsize uniforms
    std::set<std::string> samplerDeclarations;
    std::set<std::string> texSizeDeclarations;
    std::vector<std::string> mainSamplerNames;
    for (const auto& desc : m_mainTextureDescriptors)
    {
        samplerDeclarations.insert(desc.SamplerDeclaration());
        texSizeDeclarations.insert(desc.TexSizeDeclaration());
        mainSamplerNames.push_back("sampler_" + desc.SamplerName());
    }
    for (const auto& desc : presetState.blurTexture.GetDescriptorsForBlurLevel(m_maxBlurLevelRequired))
    {
        samplerDeclarations.insert(desc.SamplerDeclaration());
    }
    for (const auto& desc : m_textureSamplerDescriptors)
    {
        samplerDeclarations.insert(desc.SamplerDeclaration());
        texSizeDeclarations.insert(desc.TexSizeDeclaration());
    }

    std::string glslResult = GenerateGLSLSource(samplerDeclarations, texSizeDeclarations, mainSamplerNames, program);
    UpdateShaderUsageFlags(glslResult);

    // Compile the preset shader fragment shader with the standard vertex shader.
    if (m_type == ShaderType::WarpShader)
    {
        m_shader.CompileProgram(MilkdropStaticShaders::Get()->GetPresetWarpVertexShader(), glslResult.c_str());
    }
    else
    {
        m_shader.CompileProgram(MilkdropStaticShaders::Get()->GetPresetCompVertexShader(), glslResult.c_str());
    }

    CacheUniformLocations();
}

void MilkdropShader::UpdateShaderUsageFlags(const std::string& glslSource)
{
    auto containsIdentifier = [](const std::string& text, const char* ident, size_t startOffset = 0, size_t endOffset = std::string::npos) -> bool {
        const size_t identLen = std::strlen(ident);
        const size_t limit = std::min(endOffset, text.size());
        size_t pos = startOffset;
        while (pos < limit && (pos = text.find(ident, pos)) != std::string::npos && pos < limit)
        {
            const bool leftOk = (pos == 0) ||
                                (!std::isalnum(static_cast<unsigned char>(text[pos - 1])) && text[pos - 1] != '_');
            const size_t after = pos + identLen;
            const bool rightOk = (after >= text.size()) ||
                                 (!std::isalnum(static_cast<unsigned char>(text[after])) && text[after] != '_');
            if (leftOk && rightOk)
            {
                return true;
            }
            pos += identLen;
        }
        return false;
    };

    m_usesDiscard = containsIdentifier(glslSource, "discard");
    m_usesPerFrameUbo = (glslSource.find("uniform PerFrameBlock") != std::string::npos);

    size_t codeAfterUbo = 0;
    if (m_usesPerFrameUbo)
    {
        size_t const blockPos = glslSource.find("uniform PerFrameBlock");
        size_t const blockClose = glslSource.find("};", blockPos);
        if (blockClose != std::string::npos)
        {
            codeAfterUbo = blockClose + 2;
        }
    }

    m_hasRandFrameUniform = containsIdentifier(glslSource, "rand_frame", codeAfterUbo);
    m_hasFastTrigUniforms = containsIdentifier(glslSource, "_c8", codeAfterUbo) ||
                            containsIdentifier(glslSource, "_c9", codeAfterUbo);
    m_hasSlowTrigUniforms = containsIdentifier(glslSource, "_c10", codeAfterUbo) ||
                            containsIdentifier(glslSource, "_c11", codeAfterUbo);

    m_usesPolarCoords = false;
    m_usesHueShader = false;
    size_t const psSig = glslSource.find("void PS(");
    if (psSig != std::string::npos)
    {
        size_t const psOpen = glslSource.find('{', psSig);
        size_t const mainSig = glslSource.find("void main()", psSig);
        if (psOpen != std::string::npos)
        {
            size_t const psEnd = (mainSig != std::string::npos) ? mainSig : glslSource.size();
            m_usesPolarCoords = containsIdentifier(glslSource, "_rad_ang", psOpen + 1, psEnd);
            m_usesHueShader = (m_type == ShaderType::CompositeShader) &&
                              containsIdentifier(glslSource, "_vDiffuse", psOpen + 1, psEnd);
        }
    }
}

void MilkdropShader::CacheUniformLocations()
{
    m_locVertexTransformation = m_shader.GetUniformLocation("vertex_transformation");
    m_locCurlNoise = m_shader.GetUniformLocation("u_curlNoise");
    m_locCatmullRom = m_shader.GetUniformLocation("u_catmullRom");

    if (m_usesPerFrameUbo && m_shader.Program() != 0)
    {
        GLuint const blockIndex = glGetUniformBlockIndex(m_shader.Program(), "PerFrameBlock");
        if (blockIndex != GL_INVALID_INDEX)
        {
            glUniformBlockBinding(m_shader.Program(), blockIndex, 0);
            if (m_perFrameUbo == 0)
            {
                glGenBuffers(1, &m_perFrameUbo);
                glBindBuffer(GL_UNIFORM_BUFFER, m_perFrameUbo);
                glBufferData(GL_UNIFORM_BUFFER, sizeof(PerFrameUBOData), nullptr, GL_DYNAMIC_DRAW);
                glBindBuffer(GL_UNIFORM_BUFFER, 0);
            }
            m_locRandFrame = -1;
            m_locRandPreset = -1;
            m_locC.fill(-1);
            m_locQ.fill(-1);
        }
        else
        {
            m_usesPerFrameUbo = false;
        }
    }

    if (!m_usesPerFrameUbo)
    {
        m_locRandFrame = m_shader.GetUniformLocation("rand_frame");
        m_locRandPreset = m_shader.GetUniformLocation("rand_preset");

        static const char* const s_cVarNames[14] = {
            "_c0", "_c1", "_c2", "_c3", "_c4", "_c5", "_c6",
            "_c7", "_c8", "_c9", "_c10", "_c11", "_c12", "_c13"
        };
        for (size_t i = 0; i < 14; ++i)
        {
            m_locC[i] = m_shader.GetUniformLocation(s_cVarNames[i]);
        }
        m_hasFastTrigUniforms = (m_locC[8] >= 0 || m_locC[9] >= 0);
        m_hasSlowTrigUniforms = (m_locC[10] >= 0 || m_locC[11] >= 0);

        static const char* const s_qVarNames[8] = {
            "_qa", "_qb", "_qc", "_qd", "_qe", "_qf", "_qg", "_qh"
        };
        for (size_t i = 0; i < 8; ++i)
        {
            m_locQ[i] = m_shader.GetUniformLocation(s_qVarNames[i]);
        }
    }

    static const char* const s_rotVarNames[24] = {
        "rot_s1", "rot_s2", "rot_s3", "rot_s4",
        "rot_d1", "rot_d2", "rot_d3", "rot_d4",
        "rot_f1", "rot_f2", "rot_f3", "rot_f4",
        "rot_vf1", "rot_vf2", "rot_vf3", "rot_vf4",
        "rot_uf1", "rot_uf2", "rot_uf3", "rot_uf4",
        "rot_rand1", "rot_rand2", "rot_rand3", "rot_rand4"
    };
    m_has3DRotationUniforms = false;
    for (size_t i = 0; i < 24; ++i)
    {
        m_locRot[i] = m_shader.GetUniformLocation(s_rotVarNames[i]);
        if (m_locRot[i] >= 0)
        {
            m_has3DRotationUniforms = true;
        }
    }
}

void MilkdropShader::UpdateMaxBlurLevel(BlurTexture::BlurLevel requestedLevel)
{
    if (m_maxBlurLevelRequired >= requestedLevel)
    {
        return;
    }

    m_maxBlurLevelRequired = requestedLevel;

    if (m_maxBlurLevelRequired == BlurTexture::BlurLevel::Blur3)
    {
        m_samplerNames.insert("blur1");
        m_samplerNames.insert("blur2");
        m_samplerNames.insert("blur3");
    }
    else if (m_maxBlurLevelRequired == BlurTexture::BlurLevel::Blur2)
    {
        m_samplerNames.insert("blur1");
        m_samplerNames.insert("blur2");
    }
    else
    {
        m_samplerNames.insert("blur1");
    }
}

} // namespace MilkdropPreset
} // namespace libprojectM
