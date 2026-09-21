#pragma once

#include "Renderer/Framebuffer.hpp"
#include "Renderer/Mesh.hpp"
#include "Renderer/ShaderCache.hpp"

namespace libprojectM {
namespace Renderer {

/**
 * @class CopyTexture
 * @brief Copies the given input texture to an output texture or framebuffer.
 *
 * Optionally flips the image horizontally or vertically during the copy operation.
 */
class CopyTexture
{
public:
    CopyTexture();

    /**
     * @brief Enables or disables Perceptual Display P3 Gamut Remastering in the blit copy pass.
     */
    static void SetWideGamutEnabled(bool enabled);
    static bool IsWideGamutEnabled();

    /**
     * @brief Sets the color profile mode:
     * 0 = Natural (Cinema Reference, relaxed saturation, dye-transfer density) [Default]
     * 1 = Vivid (Caribbean 4K HDR, expanded punchy saturation)
     */
    static void SetColorProfile(int profile);
    static int GetColorProfile();

    /**
     * @brief Marks this instance as the final presentation copier to the screen.
     * When true, applies Cinema Display P3 Remastering, 0-nit OLED black gating, and spatial dithering.
     * When false (default, e.g. internal preset flip filters), performs a pure 1:1 bit-exact texture copy.
     */
    void SetIsScreenPresentation(bool enable);
    bool IsScreenPresentation() const;

    /**
     * @brief Copies the original texture into the currently bound framebuffer.
     * @param shaderCache The global shader cache instance.
     * @param originalTexture The texture to be copied.
     * @param flipVertical Flip image on the y-axis when copying.
     * @param flipHorizontal Flip image on the x-axis when copying.
     */
    void Draw(ShaderCache& shaderCache,
              const std::shared_ptr<Texture>& originalTexture,
              bool flipVertical = false, bool flipHorizontal = false);

    /**
     * @brief Copies the original texture either into the object's internal framebuffer or a given target texture.
     * The original and target textures must not be the same.
     * @param shaderCache The global shader cache instance.
     * @param originalTexture The texture to be copied.
     * @param targetTexture Optional target texture to draw onto.
     * @param flipVertical Flip image on the y-axis when copying.
     * @param flipHorizontal Flip image on the x-axis when copying.
     */
    void Draw(ShaderCache& shaderCache,
              const std::shared_ptr<Texture>& originalTexture,
              const std::shared_ptr<Texture>& targetTexture = {},
              bool flipVertical = false, bool flipHorizontal = false);

    /**
     * @brief Copies the texture bound the given framebuffer's first color attachment.
     * This is done by drawing into a second framebuffer, then swapping the textures, so the original texture
     * can be the current color attachment of targetFramebuffer.
     * @param shaderCache The global shader cache instance.
     * @param originalTexture The texture to be copied.
     * @param targetFramebuffer Optional target texture to draw onto.
     * @param framebufferIndex The index of the framebuffer to use.
     * @param flipVertical Flip image on the y-axis when copying.
     * @param flipHorizontal Flip image on the x-axis when copying.
     */
    void Draw(ShaderCache& shaderCache,
              const std::shared_ptr<Texture>& originalTexture,
              Framebuffer& framebuffer, int framebufferIndex,
              bool flipVertical = false, bool flipHorizontal = false);

    /**
     * @brief Draws the original texture onto the specified target texture, using the provided screen coordinates to position it.
     * @param shaderCache The global shader cache instance.
     * @param originalTexture The texture to be copied.
     * @param targetTexture The target texture to draw onto.
     * @param left Left offset on the target texture in screen coordinates.
     * @param top Top offset on the target texture in screen coordinates.
     * @param width Width on the target texture in screen coordinates. Use a negative value to flip vertically.
     * @param height Height on the target texture in screen coordinates. Use a negative value to flip horizontally.
     */
    void Draw(ShaderCache& shaderCache,
              const std::shared_ptr<Texture>& originalTexture,
              const std::shared_ptr<Texture>& targetTexture,
              int left, int top, int width, int height);

    /**
     * @brief Draws a raw GL texture into the currently bound framebuffer, using the provided screen coordinates to position it.
     * @param shaderCache The global shader cache instance.
     * @param originalTexture The texture ID to be copied.
     * @param viewportWidth The target surface width.
     * @param viewportHeight The target surface height.
     * @param left Left offset on the target texture in screen coordinates.
     * @param top Top offset on the target texture in screen coordinates.
     * @param width Width on the target texture in screen coordinates. Use a negative value to flip vertically.
     * @param height Height on the target texture in screen coordinates. Use a negative value to flip horizontally.
     */
    void Draw(ShaderCache& shaderCache,
              GLuint originalTexture,
              int viewportWidth, int viewportHeight,
              int left, int top, int width, int height);

    /**
     * @brief Returns the flipped texture.
     *
     * @return The flipped texture.
     */
    auto Texture() -> std::shared_ptr<Texture>;

    /**
     * @brief Pre-compiles and warms up the copy shader so it never stalls on the first frame after a transition.
     * @param shaderCache The global shader cache instance.
     */
    void WarmUp(ShaderCache& shaderCache);

private:
    /**
     * Updates the mesh
     */
    void UpdateTextureSize(int width, int height);

    void Copy(ShaderCache& shaderCache,
              bool flipVertical, bool flipHorizontal);

    void Copy(ShaderCache& shaderCache,
              int left, int top, int width, int height);

    Mesh m_mesh;
    std::weak_ptr<Shader> m_shader;                  //!< Simple textured shader
    Framebuffer m_framebuffer{1};                    //!< Framebuffer for drawing the flipped texture
    Sampler m_sampler{GL_CLAMP_TO_EDGE, GL_NEAREST}; //!< Texture sampler settings

    int m_width{};  //!< Last known framebuffer/texture width
    int m_height{}; //!< Last known framebuffer/texture height
    bool m_isScreenPresentation{false}; //!< True only for the final output blit to the screen
    std::shared_ptr<Shader> BindShader(ShaderCache& shaderCache);
};

} // namespace Renderer
} // namespace libprojectM
