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

// Accurate standard sRGB to linear conversion (piecewise standard transfer function)
// Prevents artificial shadow crushing caused by naive power 2.2 on low display codes
vec3 sRGBToLinear(vec3 c) {
    vec3 linearLow = c / 12.92;
    vec3 linearHigh = pow((c + vec3(0.055)) / 1.055, vec3(2.4));
    return mix(linearHigh, linearLow, step(c, vec3(0.04045)));
}

// Accurate standard linear to sRGB display transfer function
vec3 linearToSRGB(vec3 c) {
    vec3 sRGBLow = c * 12.92;
    vec3 sRGBHigh = 1.055 * pow(max(c, vec3(0.0)), vec3(1.0 / 2.4)) - vec3(0.055);
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
    // Filmic CRT Overscan Compensation (1.025x micro-expansion):
    // Recreates the natural bezel edge crop of classic CRT monitors, cleanly pushing
    // hairline borders (ob_size <= 0.015) and edge-clamping artifacts outside the visible screen,
    // while keeping the internal feedback loop 100% bit-exact and physically intact.
    vec2 presentationUV = (fragment_tex_coord - 0.5) / 1.025 + 0.5;
    src = texture(texture_sampler, presentationUV);

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

        // [Punto Dulce Cinematográfico - Ajuste 1] Filmic Highlight Headroom (18% Headroom, Techo 0.82):
        // Softens peak brightness starting earlier at L > 0.52, capping maximum luminance at ~0.82.
        // Tames aggressive audio beat flashes, completely eliminating color clipping and preserving
        // razor-sharp micro-textures and internal gradient lines in bright waveforms and filaments.
        if (L > 0.52) {
            float over = L - 0.52;
            float maxOver = 1.0 - 0.52; // 0.48
            float compressed = over / (1.0 + over * 2.0);
            L = 0.52 + compressed * (0.30 / (maxOver / (1.0 + maxOver * 2.0)));
            lab.x = L;
        }

        // [Punto Dulce Cinematográfico - Ajuste 2] Low-End Detail & Shadow Body Lift:
        // Opens up shadow textures, nebulas, faint background stars, and subtle audio ripples
        // in L in [0.002, 0.42], giving substantial body and presence to the low/mid range.
        if (L > 0.001 && L < 0.42) {
            float shadowFactor = 1.0 - (L / 0.42);
            L += 0.038 * shadowFactor * shadowFactor;
            lab.x = L;
        }

        // [Punto Dulce Cinematográfico - Ajuste 3] Cinematic Color Volume & Deep Low-Color Richness:
        // Subtle, organic chroma breath (+3.5% in Vivid, +1.5% in Natural) with Saturation Protection Envelope.
        // Tapers boost to ZERO on already-saturated pure primaries (C > 0.10) to preserve razor-sharp edges.
        // Deep low-color density boost enhances chroma in dark tones (L in [0.015, 0.40]) so dark blues,
        // wine reds, and forest greens look velvety and rich like Kodak Vision3 35mm film.
        if (C > 1e-6) {
            float maxBoost = (u_color_profile == 1) ? 0.035 : 0.015;

            // Saturation Protection Envelope:
            float satProtection = clamp(1.0 - max(0.0, C - 0.10) / 0.08, 0.0, 1.0);
            float boost = maxBoost * satProtection * (C / (C + 0.035));

            // Highlight Chroma Preservation Taper:
            float highlightTaper = clamp(1.0 - max(0.0, L - 0.52) / 0.30, 0.0, 1.0);
            float chromaScale = 1.0 + boost * highlightTaper;

            // Deep low-color chromatic richness:
            if (L > 0.015 && L < 0.40) {
                float lowColorBoost = 1.0 - (L / 0.40);
                chromaScale *= (1.0 + 0.06 * lowColorBoost * lowColorBoost);
            }

            // Clean shadow toe: avoids chroma noise at extreme noise floor
            if (L < 0.015) {
                chromaScale *= smoothstep(0.001, 0.015, L);
            }

            lab.yz *= chromaScale;
        }

        // Transform from OKLab to Linear RGB (Canonical Inverse - Zero Bias):
        vec3 lmsBack = kOKLabToLMS * lab;
        vec3 linColor = kLMSToLinearRGB * (lmsBack * lmsBack * lmsBack);

        // Cinema Soft-Knee Gamut Compression:
        // Matches the 0.82 cinematic ceiling with gentle roll-off
        float thresh = 0.84;
        vec3 excess = max(linColor - vec3(thresh), vec3(0.0));
        linColor = min(linColor, vec3(thresh)) + (1.0 - thresh) * (excess / (vec3(1.0) + excess * 2.0));

        // 0-Nit True Black OLED Gate:
        // Only gates the absolute noise floor (L <= 0.0008), preserving all faint particles,
        // deep space waves, and low-energy audio reactivity down to 0.001.
        float blackToe = smoothstep(0.0001, 0.0008, L);
        linColor *= blackToe;

        // Display Gamma Encode (Precise piecewise sRGB transfer):
        outColor = linearToSRGB(clamp(linColor, 0.0, 1.0));
    } else {
        // Standard sRGB Calibrated Mode
        outColor = rgb;
    }

    // Final 0-Nit True Black Hermite Gate:
    // Guarantees absolute 0.000 nits on pure black backgrounds without crushing dark details
    float lum = dot(outColor, vec3(0.2126, 0.7152, 0.0722));
    float finalBlackGate = smoothstep(0.0001, 0.0012, lum);
    outColor *= finalBlackGate;

    // Edge Dissolve: subtle 2-pixel hermite falloff at the absolute outer bezel boundary
    // ensuring zero harsh 1-pixel seams against the TV frame
    float edgeDist = min(min(fragment_tex_coord.x, 1.0 - fragment_tex_coord.x),
                         min(fragment_tex_coord.y, 1.0 - fragment_tex_coord.y));
    float edgeFade = smoothstep(0.0, 0.0025, edgeDist);
    outColor *= edgeFade;

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
