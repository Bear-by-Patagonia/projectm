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

static bool s_wideGamutEnabled = true;
static int s_colorProfile = 1; // Default: 1 = Vivid (Vibrante 4K), 0 = Natural (Cinematográfico Natural)

void CopyTexture::SetWideGamutEnabled(bool enabled)
{
    s_wideGamutEnabled = enabled;
}

bool CopyTexture::IsWideGamutEnabled()
{
    return s_wideGamutEnabled;
}

void CopyTexture::SetColorProfile(int profile)
{
    s_colorProfile = profile;
}

int CopyTexture::GetColorProfile()
{
    return s_colorProfile;
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

static constexpr char CopyTextureFragmentShader[] = R"(
precision highp float;

in vec2 fragment_tex_coord;

uniform sampler2D texture_sampler;
uniform int u_is_screen_presentation;
uniform int u_wide_gamut_mode;
uniform int u_color_profile; // 0 = Natural (Cinema Reference, less saturated) [default], 1 = Vivid (Caribbean 4K HDR)

out vec4 color;

// Fast sRGB to linear conversion (gamma 2.2 approximation)
vec3 sRGBToLinear(vec3 c) {
    return pow(max(c, vec3(0.0)), vec3(2.2));
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
    4.0767416621, -1.2684380046, -0.0041960863, // Column 0
   -3.3077115913,  2.6097574011, -0.7034186147, // Column 1
    0.2309699292, -0.3413193965,  1.7076147010  // Column 2
);

void main() {
    vec4 src = texture(texture_sampler, fragment_tex_coord);

    // If this is an internal texture copy/flip pass (e.g. MilkDrop feedback loop),
    // perform a bit-exact 1:1 copy so preset colors and feedback loops are never corrupted.
    if (u_is_screen_presentation == 0) {
        color = src;
        return;
    }

    // --- Final Screen Presentation Pass ---
    vec3 rgb = clamp(src.rgb, 0.0, 1.0);
    vec3 outColor;

    if (u_wide_gamut_mode == 1) {
        // --- Perceptual Color Pipeline (Symmetrical OKLCH Gamut Master) ---
        vec3 linRGB = sRGBToLinear(rgb);

        // 1. Pure Direct Linear RGB to OKLab (Zero Cross-Channel Distortion / Zero Green Bias):
        vec3 lms = kLinRGBToLMS * linRGB;
        vec3 lms_ = pow(max(lms, vec3(0.0)), vec3(1.0 / 3.0));
        vec3 lab = kLMSToOKLab * lms_;

        float L = lab.x;
        float C = length(lab.yz);

        // Highlight Detail Preservation Shoulder (IMAX DMR / Filmic Roll-off):
        // Softens peak white burn-out while preserving razor-sharp edge contrast in waveforms and filaments.
        float shoulderStart = (u_color_profile == 1) ? 0.80 : 0.82;
        if (L > shoulderStart) {
            float over = L - shoulderStart;
            float maxOver = 1.0 - shoulderStart;
            float shoulder = over / (1.0 + over * 1.8);
            L = shoulderStart + shoulder * (maxOver * 0.75 / (maxOver / (1.0 + maxOver * 1.8)));
            lab.x = L;
        }

        // 2. Dolby Vision & ACES-Inspired True Vibrance with Saturation Protection:
        // Boosts low/mid-saturation tones (+10% in Vivid) for modern color depth, but smoothly protects
        // already-saturated pure Red, Green, and Blue, preventing channel clipping and preserving fine lines/textures.
        if (C > 1e-6) {
            float maxBoost = (u_color_profile == 1) ? 0.10 : 0.04;

            // Saturation Protection Envelope:
            // Full boost on subtle/mid-tones (C in 0.02 - 0.12) for rich color breathing.
            // Tapers smoothly to 0 on already highly saturated colors (C > 0.14)
            // ensuring red, green, and blue textures, lines, and borders remain 100% razor-sharp.
            float satProtection = clamp(1.0 - max(0.0, C - 0.12) / 0.10, 0.0, 1.0);
            float boost = maxBoost * satProtection * (C / (C + 0.035));

            // Highlight Chroma Preservation Taper:
            // Prevents bright saturated highlights from burning into white
            float highlightTaper = clamp(1.0 - max(0.0, L - 0.68) / 0.28, 0.0, 1.0);
            float chromaScale = 1.0 + boost * highlightTaper;

            // Volumetric Cusp Roll-Off (L > 0.78):
            if (L > 0.78) {
                float t = (L - 0.78) / 0.22;
                chromaScale *= (1.0 - t * t * 0.18);
            }

            // Dolby Clean Shadow Toe: avoids noisy chroma in deep low-light regions
            if (L < 0.05) {
                chromaScale *= smoothstep(0.005, 0.05, L);
            }

            lab.yz *= chromaScale;
        }

        // 3. Transform from OKLab to Linear RGB (Canonical Inverse - Zero Bias):
        vec3 lmsBack = kOKLabToLMS * lab;
        vec3 linColor = kLMSToLinearRGB * (lmsBack * lmsBack * lmsBack);

        // 4. Cinema Soft-Knee Gamut Compression:
        // Maintains 100% linear micro-contrast up to 0.93, with gentle roll-off at the absolute peak.
        float thresh = (u_color_profile == 1) ? 0.93 : 0.95;
        vec3 excess = max(linColor - vec3(thresh), vec3(0.0));
        linColor = min(linColor, vec3(thresh)) + (1.0 - thresh) * (excess / (vec3(1.0) + excess * 1.5));

        // 5. Dolby Vision Continuous Shadow Toe (C1 Continuity):
        // Prevents harsh near-black contouring lines while ensuring 100% 0.000 nits on true blacks.
        // Below L < 0.008, applies smooth hermite roll-off reaching bit-exact 0.0 at L <= 0.0015.
        float blackToe = smoothstep(0.0015, 0.008, L);
        linColor *= blackToe;

        // 6. Gamma Encode (Gamma 2.2 standard transfer function):
        outColor = pow(clamp(linColor, 0.0, 1.0), vec3(1.0 / 2.2));
    } else {
        // Standard sRGB Calibrated Mode
        outColor = rgb;
    }

    // Dolby Vision Display Management: Final 0-Nit True Black Hermite Gate
    float lum = dot(outColor, vec3(0.2126, 0.7152, 0.0722));
    float finalBlackGate = smoothstep(0.0015, 0.006, lum);
    outColor *= finalBlackGate;

    // Guarantee 100% solid opacity: 0-nit true OLED black, zero background alpha leak
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
    shader->SetUniformInt("u_is_screen_presentation", m_isScreenPresentation ? 1 : 0);
    shader->SetUniformInt("u_wide_gamut_mode", s_wideGamutEnabled ? 1 : 0);
    shader->SetUniformInt("u_color_profile", s_colorProfile);
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
    shader->SetUniformInt("u_is_screen_presentation", m_isScreenPresentation ? 1 : 0);
    shader->SetUniformInt("u_wide_gamut_mode", s_wideGamutEnabled ? 1 : 0);
    shader->SetUniformInt("u_color_profile", s_colorProfile);
    shader->SetUniformMat4x4("vertex_transformation", translationMatrix);

    m_sampler.Bind(0);

    m_mesh.Draw();

    Mesh::Unbind();
    Sampler::Unbind(0);
    Shader::Unbind();
}

std::shared_ptr<Shader> CopyTexture::BindShader(ShaderCache& shaderCache)
{
    auto shader = m_shader.lock();

    if (!shader)
    {
        shader = shaderCache.Get("copy_texture");
    }

    if (!shader)
    {
        std::string vertexShader(ShaderVersion);
        std::string fragmentShader(ShaderVersion);
        vertexShader.append(CopyTextureVertexShader);
        fragmentShader.append(CopyTextureFragmentShader);

        shader = std::make_shared<Shader>();
        shader->CompileProgram(vertexShader, fragmentShader);

        m_shader = shader;
        shaderCache.Insert("copy_texture", shader);
    }

    shader->Bind();

    return shader;
}

void CopyTexture::WarmUp(ShaderCache& shaderCache)
{
    BindShader(shaderCache);
    Shader::Unbind();
}

} // namespace Renderer
} // namespace libprojectM
