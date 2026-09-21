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

void CopyTexture::SetWideGamutEnabled(bool enabled)
{
    s_wideGamutEnabled = enabled;
}

bool CopyTexture::IsWideGamutEnabled()
{
    return s_wideGamutEnabled;
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

static constexpr char CopyTextureFragmentShader[] = R"(
precision highp float;

in vec2 fragment_tex_coord;

uniform sampler2D texture_sampler;
uniform int u_is_screen_presentation;
uniform int u_wide_gamut_mode;

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
        // --- OLED Reference Master Color Pipeline (LG OLED Demo Quality) ---
        vec3 linRGB = sRGBToLinear(rgb);

        // 1. Spectral Harmonization (Subpixel Softening):
        // Natural physical light sources always have spectral bandwidth.
        // Injects delicate optical resonance into secondary channels so pure procedural primaries
        // don't burn like single-subpixel laser stickers, transforming harsh reds into velvet ruby
        // and harsh blues into ocean sapphire. Preserves D65 white point exactly (rows sum to 1.0).
        mat3 spectralHarmonize = mat3(
            0.910, 0.045, 0.045, // Column 0
            0.045, 0.910, 0.045, // Column 1
            0.045, 0.045, 0.910  // Column 2
        );
        vec3 harmLin = spectralHarmonize * linRGB;

        // 2. Calibrated Cinema Display P3 Projection (18% Gamut Volume Expansion)
        mat3 srgbToP3Cinema = mat3(
            0.8544, 0.0272, 0.0140, // Column 0
            0.1456, 0.9728, 0.0594, // Column 1
            0.0000, 0.0000, 0.9266  // Column 2
        );
        vec3 p3Lin = srgbToP3Cinema * harmLin;

        // 3. ACES Reference Gamut Compression (RGC Soft-Knee):
        // Compresses extreme out-of-bounds saturation towards the achromatic (neutral) axis.
        // Prevents primary channel clipping and allows delicate intermediate hues (amber, coral, teal) to breathe.
        float achromatic = max(p3Lin.r, max(p3Lin.g, p3Lin.b));
        if (achromatic > 1e-5) {
            vec3 dist = (vec3(achromatic) - p3Lin) / achromatic;
            float threshold = 0.70;
            vec3 compDist = dist;
            for (int i = 0; i < 3; i++) {
                if (dist[i] > threshold) {
                    float d = (dist[i] - threshold) / (1.0 - threshold);
                    compDist[i] = threshold + (1.0 - threshold) * (d / (1.0 + 0.5 * d));
                }
            }
            p3Lin = vec3(achromatic) - compDist * achromatic;
        }

        // 4. Filmic S-Curve Tone Mapping (ACES Fit):
        // Delivers the velvety contrast of the LG OLED demo video:
        // - Deep inky shadows with natural toe roll-off.
        // - Rich, uncompressed, tactile midtones.
        // - Smooth highlight shoulder: highlights maintain their rich color identity without glaring.
        vec3 x = p3Lin;
        vec3 filmic = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
        filmic *= 1.244; // Normalize peak 1.0 -> 1.0

        // 5. Cinema OLED Gamma BT.1886 (2.35): deepens color density without lechoso cast
        outColor = pow(clamp(filmic, 0.0, 1.0), vec3(1.0 / 2.35));
    } else {
        // Standard sRGB Calibrated Mode
        outColor = rgb;
    }

    // Gate dithering strictly off on blacks (lum < 0.015)
    // On OLED displays, adding even +1/255 dither to black pixels activates organic subpixels,
    // causing visible gray noise/speckles. Smoothstep ensures 100% pure 0-nit black while
    // providing silky, continuous banding elimination across midtones and gradients.
    float lum = dot(outColor, vec3(0.2126, 0.7152, 0.0722));
    float ditherGate = smoothstep(0.015, 0.06, lum);
    outColor += vec3(TriangularDither(gl_FragCoord.xy)) * ditherGate;

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
