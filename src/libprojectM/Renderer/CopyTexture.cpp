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
static int s_colorProfile = 0; // Default: 0 = Natural (Cinema Reference, less saturated)

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

// High-frequency spatial dither to eliminate banding and posterization
float TriangularDither(vec2 coord) {
    float r1 = fract(sin(dot(coord, vec2(12.9898, 78.233))) * 43758.5453);
    float r2 = fract(sin(dot(coord + vec2(0.5, 0.5), vec2(12.9898, 78.233))) * 43758.5453);
    return (r1 + r2 - 1.0) / 255.0;
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

const mat3 kLMSToLinearP3 = mat3(
    3.12776915, -1.09101011, -0.02600875,
    -2.25713598, 2.41333293, -0.50804375,
    0.12936683, -0.32232282, 1.53405250
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
        // --- Display P3 Master Pipeline (OKLCH Perceptual Gamut Extension) ---
        vec3 linRGB = sRGBToLinear(rgb);

        // 1. Spectral Harmonization (Subpixel Cushioning):
        // Profile 0 (Natural): 3.5% cushioning -> velvety, organic, dye-transfer density (Wizard of Oz / Back to the Future)
        // Profile 1 (Vivid):   2.0% cushioning -> maximum punchy vibrancy (Caribbean 4K HDR)
        // Rows sum to 1.0, preserving neutral D65 white point bit-exactly.
        float cushion = (u_color_profile == 1) ? 0.020 : 0.035;
        float diag = 1.0 - 2.0 * cushion;
        mat3 spectralHarmonize = mat3(
            diag, cushion, cushion, // Column 0
            cushion, diag, cushion, // Column 1
            cushion, cushion, diag  // Column 2
        );
        vec3 harmLin = spectralHarmonize * linRGB;

        // 2. Transform to OKLab Perceptual Uniform Color Space:
        vec3 lms = kLinRGBToLMS * harmLin;
        vec3 lms_ = pow(max(lms, vec3(0.0)), vec3(1.0 / 3.0));
        vec3 lab = kLMSToOKLab * lms_;

        float L = lab.x;
        float C = length(lab.yz);

        // 3. Sigmoidal Gamut Extension (GEA) with Hue Angle Locking:
        // Profile 0 (Natural): Gentle +16% expansion, relaxed, non-fatiguing, wide color breathing
        // Profile 1 (Vivid):   Punchy +38% expansion into full P3 volume
        // Hue angle is 100.0% mathematically locked to prevent any Hue Twisting (Abney effect).
        if (C > 1e-6) {
            float c2 = C * C;
            float maxBoost = (u_color_profile == 1) ? 0.38 : 0.16;
            float midPoint = (u_color_profile == 1) ? 0.0049 : 0.0064;
            float chromaScale = 1.0 + maxBoost * (c2 / (c2 + midPoint));

            // Specular glint roll-off (Highlight Cusp - prevents neon highlights / plastic skin)
            float rollStart = (u_color_profile == 1) ? 0.82 : 0.78;
            if (L > rollStart) {
                float t = (L - rollStart) / (1.0 - rollStart);
                chromaScale *= (1.0 - t * 0.30);
            }

            // Shadow Clean Roll-Off: ensures pure neutral darks without colored noise
            if (L < 0.05) {
                chromaScale *= (L / 0.05);
            }

            lab.yz *= chromaScale;
        }

        // 4. Transform from OKLab to Linear Display P3:
        vec3 lmsBack = kOKLabToLMS * lab;
        vec3 p3Lin = kLMSToLinearP3 * (lmsBack * lmsBack * lmsBack);

        // 5. Cinema Soft-Knee Gamut Compression:
        // Smooth hyperbolic compression as channels approach peak, preventing hard digital clipping.
        float thresh = (u_color_profile == 1) ? 0.88 : 0.85;
        vec3 excess = max(p3Lin - vec3(thresh), vec3(0.0));
        p3Lin = min(p3Lin, vec3(thresh)) + (1.0 - thresh) * (excess / (vec3(1.0) + excess));

        // 6. Strict 0-Nit OLED Black Clamp:
        // When content is dark, organic subpixels are 100% shut off.
        if (L < 0.003) {
            p3Lin = vec3(0.0);
        }

        // 7. Display P3 Gamma Encode (Gamma 2.2 standard transfer function):
        outColor = pow(clamp(p3Lin, 0.0, 1.0), vec3(1.0 / 2.2));
    } else {
        // Standard sRGB Calibrated Mode
        outColor = rgb;
    }

    // Gate dithering strictly off on blacks (lum < 0.015)
    float lum = dot(outColor, vec3(0.2126, 0.7152, 0.0722));
    float ditherGate = smoothstep(0.015, 0.06, lum);
    outColor += vec3(TriangularDither(gl_FragCoord.xy)) * ditherGate;

    // Strict 0-nit black enforcement
    if (lum < 0.003) {
        outColor = vec3(0.0);
    }

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
