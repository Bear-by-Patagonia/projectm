/**
 * @file BlurTexture.hpp
 * @brief Blurs a given texture in multiple passes and stores the results.
 */
#pragma once

#include <Renderer/Framebuffer.hpp>
#include <Renderer/Mesh.hpp>
#include <Renderer/RenderContext.hpp>
#include <Renderer/Shader.hpp>
#include <Renderer/TextureSamplerDescriptor.hpp>

#include <array>
#include <memory>

#ifndef GL_R11F_G11F_B10F
#define GL_R11F_G11F_B10F 0x8C3A
#endif
#ifndef GL_UNSIGNED_INT_10F_11F_11F_REV
#define GL_UNSIGNED_INT_10F_11F_11F_REV 0x8C3B
#endif

namespace libprojectM {
namespace MilkdropPreset {

class PerFrameContext;
class PresetState;

/**
 * @brief Blurs a given texture in multiple passes and stores the results.
 *
 * Blur textures are not stored in the texture manager to enable independent blur textures
 * for each loaded preset, e.g. during blending.
 */
class BlurTexture
{
public:
    using Values = std::array<float, 3>;

    /**
     * Maximum main texture blur level used in the shader
     */
    enum class BlurLevel : int
    {
        None,  //!< No blur used.
        Blur1, //!< First blur level (2 passes)
        Blur2, //!< Second blur level (4 passes)
        Blur3  //!< Third blur level (6 passes)
    };

    /**
     * Constructor.
     */
    BlurTexture();

    /**
     * Destructor.
     */
    virtual ~BlurTexture() = default;

    /**
     * @brief Initializes the blur texture.
     * @param renderContext
     */
    void Initialize(const Renderer::RenderContext& renderContext);

    /**
     * @brief Sets the minimum required blur level.
     * If the current level isn't high enough, it'll be increased.
     * @param level The minim blur level.
     */
    void SetRequiredBlurLevel(BlurLevel level);

    /**
     * @brief Returns the minimum blur level required by the active preset shaders.
     */
    BlurLevel GetRequiredBlurLevel() const noexcept { return m_blurLevel; }

    /**
     * @brief Returns a list of descriptors for the given blur level.
     * The blur textures don't need to be present and can be empty placeholders.
     * @param blurLevel The blur level.
     */
    auto GetDescriptorsForBlurLevel(BlurLevel blurLevel) const -> std::vector<Renderer::TextureSamplerDescriptor>;

    /**
     * @brief Renders the required blur passes on the given texture.
     * @param sourceTexture The texture to create the blur levels from.
     * @param perFrameContext The per-frame variables.
     * @param currentFbo Active draw framebuffer ID to restore after blur pass.
     */
    void Update(const Renderer::Texture& sourceTexture, const PerFrameContext& perFrameContext, GLuint currentFbo = 0);

    /**
     * @brief Binds the user-readable blur textures to the texture slots starting with the given index.
     * The shader must already be bound.
     * @param[in,out] unit The first texture unit to bind the blur textures from. Returns the next
     *                     free unit, which can be the same as the input slot if no blur textures
     *                     are used.
     */
    void Bind(GLint& unit, Renderer::Shader& shader) const;

    struct BlurStorageFormat
    {
        GLint internalFormat{GL_RGBA8};
        GLenum format{GL_RGBA};
        GLenum type{GL_UNSIGNED_BYTE};
        int bitsPerPixel{32};
        bool isFloatingPoint{false};
    };

    /**
     * @brief Selects optimal 32-bpp blur pyramid storage format:
     *        GL_R11F_G11F_B10F (5-bit exp + 6/6/5-bit mantissa, AFBC-compatible) when color-renderable,
     *        or GL_RGBA8 (32-bpp AFBC-friendly UNORM) fallback. Never unaligned 24-bit GL_RGB.
     */
    static auto SelectOptimalFormat(bool supportsR11G11B10F) noexcept -> BlurStorageFormat;

    /**
     * @brief Returns properly scaled and clamped vlur values from the given context.
     * @param perFrameContext The per-frame context to retrieve the initial values from.
     * @param blurMin The calculated min values.
     * @param blurMax The calculated max values.
     */
    static void GetSafeBlurMinMaxValues(const PerFrameContext& perFrameContext,
                                        Values& blurMin,
                                        Values& blurMax);

private:
    static constexpr int NumBlurTextures = 6; //!< Maximum number of blur passes/textures.

    /**
     * Allocates the blur textures.
     * @param sourceTexture The source texture.
     * @param currentFbo Active draw framebuffer ID to restore if format probing occurs.
     */
    void AllocateTextures(const Renderer::Texture& sourceTexture, GLuint currentFbo = 0);

    Renderer::Mesh m_blurMesh; //!< The blur mesh (a simple quad).

    std::weak_ptr<Renderer::Shader> m_blur1Shader; //!< The shader used on the first blur pass.
    std::weak_ptr<Renderer::Shader> m_blur2Shader; //!< The shader used for subsequent blur passes after the initial pass.

    int m_sourceTextureWidth{};  //!< Width of the source texture used to create the blur textures.
    int m_sourceTextureHeight{}; //!< Height of the source texture used to create the blur textures.
    bool m_hasProbedR11F{false};
    bool m_supportsR11F{false};

    Renderer::Framebuffer m_blurFramebuffer;                                        //!< The framebuffer used to draw the blur textures.
    std::shared_ptr<Renderer::Sampler> m_blurSampler;                               //!< The blur sampler.
    std::array<std::shared_ptr<Renderer::Texture>, NumBlurTextures> m_blurTextures; //!< The blur textures for each pass.
    BlurLevel m_blurLevel{BlurLevel::None};                                         //!< Current blur level.
};

} // namespace MilkdropPreset
} // namespace libprojectM
