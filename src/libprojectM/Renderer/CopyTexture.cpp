#include "Renderer/CopyTexture.hpp"

namespace libprojectM {
namespace Renderer {

#ifdef USE_GLES
static constexpr char ShaderVersion[] = "#version 300 es\n\n";
#else
static constexpr char ShaderVersion[] = "#version 330\n\n";
#endif

static constexpr char CopyTextureVertexShader[] = R"(
precision mediump float;

layout(location = 0) in vec2 position;
layout(location = 2) in vec2 tex_coord;

out vec2 fragment_tex_coord;

uniform mat4 vertex_transformation;

void main() {
    gl_Position = vec4(position, 0.0, 1.0) * vertex_transformation;
    fragment_tex_coord = tex_coord;
}
)";

#include <atomic>

static std::atomic<bool> s_wideGamutEnabled{true};
static std::atomic<int>  s_colorProfile{1}; // Default: 1 = Vivid (Vibrante 4K), 0 = Natural (Cinematográfico Natural)

void CopyTexture::SetWideGamutEnabled(bool enabled)
{
    s_wideGamutEnabled.store(enabled, std::memory_order_relaxed);
}

bool CopyTexture::IsWideGamutEnabled()
{
    return s_wideGamutEnabled.load(std::memory_order_relaxed);
}

void CopyTexture::SetColorProfile(int profile)
{
    s_colorProfile.store(profile, std::memory_order_relaxed);
}

int CopyTexture::GetColorProfile()
{
    return s_colorProfile.load(std::memory_order_relaxed);
}

void CopyTexture::SetIsScreenPresentation(bool enable)
{
    m_isScreenPresentation = enable;
}

bool CopyTexture::IsScreenPresentation() const
{
    return m_isScreenPresentation;
}

extern "C" __attribute__((visibility("default"))) void projectm_set_wide_gamut_mode(bool enabled)
{
    CopyTexture::SetWideGamutEnabled(enabled);
}

extern "C" __attribute__((visibility("default"))) void projectm_set_color_profile(int profile)
{
    CopyTexture::SetColorProfile(profile);
}

// Fast 1:1 Bit-Exact Passthrough Fragment Shader for Internal Feedback/Flip Passes
static constexpr char CopyTextureFragmentShader[] = R"(
precision highp float;

in vec2 fragment_tex_coord;
uniform sampler2D texture_sampler;
out vec4 color;

void main() {
    color = texture(texture_sampler, fragment_tex_coord);
}
)";

// Highly Optimized OKLab Display P3 Presentation Fragment Shader
static constexpr char PresentationTextureFragmentShader[] = R"(
precision highp float;

in vec2 fragment_tex_coord;

uniform sampler2D texture_sampler;
uniform int u_wide_gamut_mode;
uniform int u_color_profile; // 0 = Natural, 1 = Vivid

out vec4 color;

// Fast sRGB to Linear conversion using exp2(y * log2(x)) for single-cycle SFU execution on Mali/Adreno
vec3 sRGBToLinear(vec3 c) {
    vec3 linearLow = c / 12.92;
    vec3 linearHigh = exp2(vec3(2.4) * log2(max((c + vec3(0.055)) / 1.055, vec3(1e-6))));
    return mix(linearHigh, linearLow, step(c, vec3(0.04045)));
}

// Fast Linear to sRGB conversion using exp2(y * log2(x))
vec3 linearToSRGB(vec3 c) {
    vec3 sRGBLow = c * 12.92;
    vec3 sRGBHigh = 1.055 * exp2(vec3(1.0 / 2.4) * log2(max(c, vec3(1e-6)))) - vec3(0.055);
    return mix(sRGBHigh, sRGBLow, step(c, vec3(0.0031308)));
}

// --- Perceptual Color Space Constants (OKLab / OKLCH) ---
const mat3 kLinRGBToLMS = mat3(
    0.4122214708, 0.2119034982, 0.0883024619,
    0.5363325363, 0.6806995451, 0.2817188376,
    0.0514459929, 0.1073969566, 0.6299787005
);

const mat3 kLMSToOKLab = mat3(
    0.2104542553, 1.9779984951, 0.0259040371,
    0.7936177850, -2.4285922050, 0.7827717662,
    -0.0040720468, 0.4505937099, -0.8086757660
);

const mat3 kOKLabToLMS = mat3(
    1.00000000, 1.00000001, 1.00000005,
    0.39633779, -0.10556134, -0.08948418,
    0.21580376, -0.06385417, -1.29148554
);

const mat3 kLMSToLinearRGB = mat3(
    4.0767416621, -1.2684380046, -0.0041960863,
   -3.3077115913,  2.6097574011, -0.7034186147,
    0.2309699292, -0.3413193965,  1.7076147010
);

void main() {
    // --- Single Texture Fetch ---
    // Filmic CRT Overscan Compensation (1.025x micro-expansion):
    vec2 presentationUV = (fragment_tex_coord - 0.5) / 1.025 + 0.5;
    vec4 src = texture(texture_sampler, presentationUV);

    vec3 rgb = clamp(src.rgb, 0.0, 1.0);
    vec3 outColor;

    if (u_wide_gamut_mode == 1) {
        // --- Perceptual Color Pipeline (Symmetrical OKLCH Gamut Master) ---
        vec3 linRGB = sRGBToLinear(rgb);

        // 1. Pure Direct Linear RGB to OKLab:
        vec3 lms = kLinRGBToLMS * linRGB;
        vec3 lms_ = pow(max(lms, vec3(0.0)), vec3(1.0 / 3.0));
        vec3 lab = kLMSToOKLab * lms_;

        float L = lab.x;
        float C = length(lab.yz);

        // [Punto Dulce HDR F1-Style v1.80] Specular Highlight Punch (Peak 1.0 Brilliance):
        // Normal colors stay natural; lights, lasers, and flashes reach 100% white brilliance
        if (L > 0.52) {
            float over = (L - 0.52) / 0.48;
            L = 0.52 + over * 0.48 * (1.0 + 0.12 * over * (1.0 - over));
            lab.x = clamp(L, 0.0, 1.0);
        }

        // [Punto Dulce HDR F1-Style v1.80] Radiant White Specular Roll-off (Abney Effect):
        // Extremely bright lights naturally roll off into brilliant radiant white cores
        float highlightWhiteRollOff = 1.0;
        if (L > 0.60) {
            float hlFactor = clamp((L - 0.60) / 0.40, 0.0, 1.0);
            highlightWhiteRollOff = 1.0 - 0.45 * (hlFactor * hlFactor);
        }

        if (C > 1e-6) {
            // Display P3 Vibrance Boost (Vivid = 0.035, Natural = 0.015)
            float maxBoost = (u_color_profile == 1) ? 0.035 : 0.015;

            // Saturation Protection: Protect already rich hues from clipping
            float satProtection = clamp(1.0 - max(0.0, C - 0.12) / 0.10, 0.0, 1.0);
            float boost = maxBoost * satProtection * (C / (C + 0.04));

            float chromaScale = 1.0 + boost;

            // Radiance desaturation for bright highlights:
            chromaScale *= highlightWhiteRollOff;

            // Clean shadow toe (prevents chroma noise on deep darks):
            if (L < 0.010) {
                chromaScale *= smoothstep(0.001, 0.010, L);
            }

            lab.yz *= chromaScale;
        }

        // Transform from OKLab to Linear RGB:
        vec3 lmsBack = kOKLabToLMS * lab;
        vec3 linColor = kLMSToLinearRGB * (lmsBack * lmsBack * lmsBack);

        // Cinema Soft-Knee Gamut Compression (preserving specular dynamic range up to 0.95):
        float thresh = 0.95;
        vec3 excess = max(linColor - vec3(thresh), vec3(0.0));
        linColor = min(linColor, vec3(thresh)) + (1.0 - thresh) * (excess / (vec3(1.0) + excess * 2.0));

        // Display Gamma Encode:
        outColor = linearToSRGB(linColor);
    } else {
        outColor = rgb;
    }

    // Unified 0-Nit True Black Hermite Gate:
    float lum = dot(outColor, vec3(0.2126, 0.7152, 0.0722));
    float finalBlackGate = smoothstep(0.0001, 0.0010, lum);
    outColor *= finalBlackGate;

    // Edge Dissolve: 2-pixel hermite falloff at the outer bezel boundary
    float edgeDist = min(min(fragment_tex_coord.x, 1.0 - fragment_tex_coord.x),
                         min(fragment_tex_coord.y, 1.0 - fragment_tex_coord.y));
    if (edgeDist < 0.0025) {
        outColor *= smoothstep(0.0, 0.0025, edgeDist);
    }

    color = vec4(clamp(outColor, 0.0, 1.0), 1.0);
}
)";

CopyTexture::CopyTexture()
    : m_mesh(VertexBufferUsage::StaticDraw, false, true)
{
    m_framebuffer.CreateColorAttachment(0, 0);

    m_mesh.SetRenderPrimitiveType(Mesh::PrimitiveType::TriangleStrip);

    m_mesh.SetVertexCount(4);
    m_mesh.Vertices().Set({{-1.0, 1.0},
                           {1.0, 1.0},
                           {-1.0, -1.0},
                           {1.0, -1.0}});

    m_mesh.UVs().Set({{0.0, 1.0},
                      {1.0, 1.0},
                      {0.0, 0.0},
                      {1.0, 0.0}});

    m_mesh.Indices().Set({0, 1, 2, 3});

    m_mesh.Update();
}

void CopyTexture::Draw(ShaderCache& shaderCache,
                       const std::shared_ptr<class Texture>& originalTexture,
                       bool flipVertical, bool flipHorizontal)
{
    if (originalTexture == nullptr)
    {
        return;
    }

    // Just bind the texture and draw it to the currently bound buffer.
    originalTexture->Bind(0);
    Copy(shaderCache, flipVertical, flipHorizontal);
}

void CopyTexture::Draw(ShaderCache& shaderCache,
                       const std::shared_ptr<class Texture>& originalTexture,
                       const std::shared_ptr<class Texture>& targetTexture,
                       bool flipVertical, bool flipHorizontal)
{
    if (originalTexture == nullptr ||
        originalTexture->Empty() ||
        (targetTexture != nullptr && targetTexture->Empty()) ||
        originalTexture == targetTexture)
    {
        return;
    }

    if (targetTexture == nullptr)
    {
        UpdateTextureSize(originalTexture->Width(), originalTexture->Height());
    }
    else
    {
        UpdateTextureSize(targetTexture->Width(), targetTexture->Height());
    }

    if (m_width == 0 || m_height == 0)
    {
        return;
    }

    std::shared_ptr<class Texture> internalTexture;

    m_framebuffer.Bind(0);

    // Draw from unflipped texture
    originalTexture->Bind(0);

    if (targetTexture)
    {
        internalTexture = m_framebuffer.GetColorAttachmentTexture(0, 0);
        m_framebuffer.GetAttachment(0, TextureAttachment::AttachmentType::Color, 0)->Texture(targetTexture);
    }

    Copy(shaderCache, flipVertical, flipHorizontal);

    // Rebind our internal texture.
    if (targetTexture)
    {
        m_framebuffer.GetAttachment(0, TextureAttachment::AttachmentType::Color, 0)->Texture(internalTexture);
    }

    Framebuffer::Unbind();
}

void CopyTexture::Draw(ShaderCache& shaderCache,
                       const std::shared_ptr<class Texture>& originalTexture,
                       Framebuffer& framebuffer, int framebufferIndex,
                       bool flipVertical, bool flipHorizontal)
{
    if (originalTexture == nullptr                                               //
        || originalTexture->Empty()                                              //
        || framebuffer.GetColorAttachmentTexture(framebufferIndex, 0) == nullptr //
        || framebuffer.GetColorAttachmentTexture(framebufferIndex, 0)->Empty())
    {
        return;
    }

    UpdateTextureSize(framebuffer.Width(), framebuffer.Height());

    if (m_width == 0 || m_height == 0)
    {
        return;
    }

    m_framebuffer.Bind(0);

    // Draw from unflipped texture
    originalTexture->Bind(0);

    Copy(shaderCache, flipVertical, flipHorizontal);

    // Swap texture attachments
    auto tempAttachment = framebuffer.GetAttachment(framebufferIndex, TextureAttachment::AttachmentType::Color, 0);
    framebuffer.RemoveColorAttachment(framebufferIndex, 0);
    framebuffer.SetAttachment(framebufferIndex, 0, m_framebuffer.GetAttachment(0, TextureAttachment::AttachmentType::Color, 0));
    m_framebuffer.RemoveColorAttachment(0, 0);
    m_framebuffer.SetAttachment(0, 0, tempAttachment);

    Framebuffer::Unbind();
}

void CopyTexture::Draw(ShaderCache& shaderCache,
                       const std::shared_ptr<class Texture>& originalTexture,
                       const std::shared_ptr<class Texture>& targetTexture,
                       int left, int top, int width, int height)
{
    if (originalTexture == nullptr ||
        originalTexture->Empty() ||
        targetTexture == nullptr ||
        targetTexture->Empty() ||
        originalTexture == targetTexture)
    {
        return;
    }

    UpdateTextureSize(targetTexture->Width(), targetTexture->Height());

    if (m_width == 0 || m_height == 0)
    {
        return;
    }

    std::shared_ptr<class Texture> internalTexture;

    m_framebuffer.Bind(0);

    // Draw from original texture
    originalTexture->Bind(0);
    internalTexture = m_framebuffer.GetColorAttachmentTexture(0, 0);
    m_framebuffer.GetAttachment(0, TextureAttachment::AttachmentType::Color, 0)->Texture(targetTexture);

    Copy(shaderCache, left, top, width, height);

    // Rebind our internal texture.
    m_framebuffer.GetAttachment(0, TextureAttachment::AttachmentType::Color, 0)->Texture(internalTexture);

    Framebuffer::Unbind();
}

void CopyTexture::Draw(ShaderCache& shaderCache,
                       GLuint originalTexture,
                       int viewportWidth, int viewportHeight,
                       int left, int top, int width, int height)
{
    if (originalTexture == 0)
    {
        return;
    }

    if (viewportWidth == 0 || viewportHeight == 0)
    {
        return;
    }

    int oldWidth = m_width;
    int oldHeight = m_height;

    m_width = viewportWidth;
    m_height = viewportHeight;

    // Draw from original texture
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, originalTexture);
    Copy(shaderCache, left, top, width, height);

    m_width = oldWidth;
    m_height = oldHeight;
}

auto CopyTexture::Texture() -> std::shared_ptr<class Texture>
{
    return m_framebuffer.GetColorAttachmentTexture(0, 0);
}

void CopyTexture::UpdateTextureSize(int width, int height)
{
    if (m_width == width &&
        m_height == height)
    {
        return;
    }

    m_width = width;
    m_height = height;

    m_framebuffer.SetSize(m_width, m_height);
}

void CopyTexture::Copy(ShaderCache& shaderCache,
                       bool flipVertical, bool flipHorizontal)
{
    glm::mat4x4 flipMatrix(1.0);

    flipMatrix[0][0] = flipHorizontal ? -1.0 : 1.0;
    flipMatrix[1][1] = flipVertical ? -1.0 : 1.0;

    std::shared_ptr<Shader> shader = BindShader(shaderCache);

    shader->SetUniformInt("texture_sampler", 0);
    if (m_isScreenPresentation) {
        shader->SetUniformInt("u_wide_gamut_mode", s_wideGamutEnabled.load(std::memory_order_relaxed) ? 1 : 0);
        shader->SetUniformInt("u_color_profile", s_colorProfile.load(std::memory_order_relaxed));
    }
    shader->SetUniformMat4x4("vertex_transformation", flipMatrix);

    m_sampler.Bind(0);

    m_mesh.Draw();

    glBindTexture(GL_TEXTURE_2D, 0);
    Mesh::Unbind();
    Sampler::Unbind(0);
    Shader::Unbind();
}

void CopyTexture::Copy(ShaderCache& shaderCache,
                       int left, int top, int width, int height)
{
    glm::mat4x4 translationMatrix(1.0);
    translationMatrix[0][0] = static_cast<float>(width) / static_cast<float>(m_width);
    translationMatrix[1][1] = static_cast<float>(height) / static_cast<float>(m_height);

    translationMatrix[3][0] = static_cast<float>(left) / static_cast<float>(m_width);
    translationMatrix[3][1] = static_cast<float>(top) / static_cast<float>(m_height);

    std::shared_ptr<Shader> shader = BindShader(shaderCache);

    shader->SetUniformInt("texture_sampler", 0);
    if (m_isScreenPresentation) {
        shader->SetUniformInt("u_wide_gamut_mode", s_wideGamutEnabled.load(std::memory_order_relaxed) ? 1 : 0);
        shader->SetUniformInt("u_color_profile", s_colorProfile.load(std::memory_order_relaxed));
    }
    shader->SetUniformMat4x4("vertex_transformation", translationMatrix);

    m_sampler.Bind(0);

    m_mesh.Draw();

    Mesh::Unbind();
    Sampler::Unbind(0);
    Shader::Unbind();
}

std::shared_ptr<Shader> CopyTexture::BindShader(ShaderCache& shaderCache)
{
    const char* shaderName = m_isScreenPresentation ? "copy_texture_presentation" : "copy_texture_passthrough";
    auto shader = (m_isScreenPresentation ? m_presentationShader : m_copyShader).lock();

    if (!shader)
    {
        shader = shaderCache.Get(shaderName);
    }

    if (!shader)
    {
        std::string vertexShader(ShaderVersion);
        std::string fragmentShader(ShaderVersion);
        vertexShader.append(CopyTextureVertexShader);

        if (m_isScreenPresentation) {
            fragmentShader.append(PresentationTextureFragmentShader);
        } else {
            fragmentShader.append(CopyTextureFragmentShader);
        }

        shader = std::make_shared<Shader>();
        shader->CompileProgram(vertexShader, fragmentShader);

        if (m_isScreenPresentation) {
            m_presentationShader = shader;
        } else {
            m_copyShader = shader;
        }
        shaderCache.Insert(shaderName, shader);
    }

    shader->Bind();

    return shader;
}

void CopyTexture::WarmUp(ShaderCache& shaderCache)
{
    bool savedState = m_isScreenPresentation;

    m_isScreenPresentation = false;
    BindShader(shaderCache);

    m_isScreenPresentation = true;
    BindShader(shaderCache);

    m_isScreenPresentation = savedState;
    Shader::Unbind();
}

} // namespace Renderer
} // namespace libprojectM
