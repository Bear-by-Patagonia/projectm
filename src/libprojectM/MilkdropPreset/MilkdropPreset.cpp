/**
 * projectM -- Milkdrop-esque visualisation SDK
 * Copyright (C)2003-2004 projectM Team
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 * See 'LICENSE.txt' included within this release
 *
 */

#include "MilkdropPreset.hpp"

#include "Factory.hpp"
#include "MilkdropPresetExceptions.hpp"
#include "PresetFileParser.hpp"

#include <Logging.hpp>

namespace libprojectM {
namespace MilkdropPreset {

MilkdropPreset::MilkdropPreset(const std::string& absoluteFilePath)
    : m_absoluteFilePath(absoluteFilePath)
    , m_perFrameContext(m_state.globalMemory, &m_state.globalRegisters)
    , m_perPixelContext(m_state.globalMemory, &m_state.globalRegisters)
    , m_motionVectors(m_state)
    , m_waveform(m_state)
    , m_darkenCenter(m_state)
    , m_border(m_state)
{
    Load(absoluteFilePath);
}

MilkdropPreset::MilkdropPreset(std::istream& presetData)
    : m_perFrameContext(m_state.globalMemory, &m_state.globalRegisters)
    , m_perPixelContext(m_state.globalMemory, &m_state.globalRegisters)
    , m_motionVectors(m_state)
    , m_waveform(m_state)
    , m_darkenCenter(m_state)
    , m_border(m_state)
{
    Load(presetData);
}

void MilkdropPreset::Initialize(const Renderer::RenderContext& renderContext)
{
    assert(renderContext.textureManager);
    m_state.renderContext = renderContext;
    m_state.blurTexture.Initialize(renderContext);
    m_state.LoadShaders();

    // Initialize variables and code now we have a proper render state.
    CompileCodeAndRunInitExpressions();

    // Update framebuffer and texture sizes if needed
    m_framebuffer.SetSize(renderContext.viewportSizeX, renderContext.viewportSizeY);
    m_hasValidMotionVectorUVMap = false;
    if (m_state.mainTexture.expired())
    {
        m_state.mainTexture = m_framebuffer.GetColorAttachmentTexture(1, 0);
    }

    m_perPixelMesh.CompileWarpShader(m_state);
    m_finalComposite.CompileCompositeShader(m_state);
}

void MilkdropPreset::RenderFrame(const libprojectM::Audio::FrameAudioData& audioData, const Renderer::RenderContext& renderContext)
{
    m_state.audioData = audioData;
    m_state.renderContext = renderContext;

    // Update framebuffer and u/v texture size if needed
    if (m_framebuffer.SetSize(renderContext.viewportSizeX, renderContext.viewportSizeY))
    {
        m_isFirstFrame = true;
        m_hasValidMotionVectorUVMap = false;
    }

    m_state.mainTexture = m_framebuffer.GetColorAttachmentTexture(m_previousFrameBuffer, 0);

    // First evaluate per-frame code
    PerFrameUpdate();

    const bool mvActive = (*m_perFrameContext.mv_a >= 0.0001 &&
                           static_cast<int>(*m_perFrameContext.mv_x) > 0 &&
                           static_cast<int>(*m_perFrameContext.mv_y) > 0);
    if (mvActive)
    {
        m_motionVectorUVMap->SetSize(renderContext.viewportSizeX, renderContext.viewportSizeY);
    }

    glViewport(0, 0, renderContext.viewportSizeX, renderContext.viewportSizeY);

    m_framebuffer.Bind(m_previousFrameBuffer);
    // Motion vector field. Drawn to the previous frame texture before warping it.
    // Only do it after drawing one frame after init or resize when UV map was populated.
    if (!m_isFirstFrame && mvActive && m_hasValidMotionVectorUVMap)
    {
        m_motionVectors.Draw(m_perFrameContext, m_motionVectorUVMap->Texture());
    }

    // y-flip the previous frame and assign the flipped texture as "main"
    m_flipTexture.Draw(*renderContext.shaderCache, m_framebuffer.GetColorAttachmentTexture(m_previousFrameBuffer, 0), nullptr, true, false);
    m_state.mainTexture = m_flipTexture.Texture();

    // Invalidate depth/stencil attachments on current FBO before switching to prevent TBDR tile flushes to RAM
    const GLenum fboDiscards[] = { GL_DEPTH_ATTACHMENT, GL_STENCIL_ATTACHMENT };
    glInvalidateFramebuffer(GL_FRAMEBUFFER, 2, fboDiscards);

    // We now draw to the current framebuffer.
    m_framebuffer.Bind(m_currentFrameBuffer);

    // Only attach motion vector u/v texture (MRT GL_COLOR_ATTACHMENT1) when motion vectors are active.
    // Skipping MRT when mv_a == 0 saves ~3.7-8.3 MB/frame of RG16F tile write bandwidth.
    if (mvActive)
    {
        m_framebuffer.SetAttachment(m_currentFrameBuffer, 1, m_motionVectorUVMap);
    }

    // Draw previous frame image warped via per-pixel mesh and warp shader
    m_perPixelMesh.Draw(m_state, m_perFrameContext, m_perPixelContext);

    if (mvActive)
    {
        // Detach the u/v texture from the framebuffer (keep contents valid for next frame's MotionVectors::Draw).
        m_framebuffer.RemoveColorAttachment(m_currentFrameBuffer, 1);
        m_hasValidMotionVectorUVMap = true;
    }
    else
    {
        m_hasValidMotionVectorUVMap = false;
    }

    // Update blur textures only if active preset shader requires blur
    if (m_state.blurTexture.GetRequiredBlurLevel() != BlurTexture::BlurLevel::None)
    {
        const auto warpedImage = m_framebuffer.GetColorAttachmentTexture(m_previousFrameBuffer, 0);
        if (warpedImage)
        {
            m_state.blurTexture.Update(*warpedImage, m_perFrameContext, m_framebuffer.GetFramebufferID(m_currentFrameBuffer));
        }

        // Restore current framebuffer and viewport before rendering shapes & waveforms
        m_framebuffer.Bind(m_currentFrameBuffer);
        glViewport(0, 0, renderContext.viewportSizeX, renderContext.viewportSizeY);
    }

    // Draw audio-data-related stuff
    for (auto& shape : m_customShapes)
    {
        shape->Draw();
    }
    for (auto& wave : m_customWaveforms)
    {
        wave->Draw(m_perFrameContext);
    }
    m_waveform.Draw(m_perFrameContext);

    // Done in DrawSprites() in Milkdrop (strictly adhere to original MilkDrop logic)
    if (*m_perFrameContext.darken_center > 0)
    {
        m_darkenCenter.Draw();
    }
    m_border.Draw(m_perFrameContext);

    // y-flip the image for final compositing again
    m_flipTexture.Draw(*renderContext.shaderCache, m_framebuffer.GetColorAttachmentTexture(m_currentFrameBuffer, 0), nullptr, true, false);
    m_state.mainTexture = m_flipTexture.Texture();

    // We no longer need the previous frame image, use it to render the final composite.
    m_framebuffer.BindRead(m_currentFrameBuffer);
    m_framebuffer.BindDraw(m_previousFrameBuffer);

    // Invalidate stale depth/stencil content in destination framebuffer before drawing final composite
    const GLenum drawDiscards[] = { GL_DEPTH_ATTACHMENT, GL_STENCIL_ATTACHMENT };
    glInvalidateFramebuffer(GL_DRAW_FRAMEBUFFER, 2, drawDiscards);

    m_finalComposite.Draw(m_state, m_perFrameContext);

    // TBDR Tile-Memory Bandwidth Optimization: Invalidate read framebuffer attachments immediately
    glInvalidateFramebuffer(GL_READ_FRAMEBUFFER, 2, drawDiscards);

    if (!m_finalComposite.HasCompositeShader())
    {
        // Flip texture again in "previous" framebuffer as old-school effects are still upside down.
        m_flipTexture.Draw(*renderContext.shaderCache, m_framebuffer.GetColorAttachmentTexture(m_previousFrameBuffer, 0), m_framebuffer, m_previousFrameBuffer, true, false);
    }

    // Swap framebuffer IDs for the next frame.
    std::swap(m_currentFrameBuffer, m_previousFrameBuffer);

    m_isFirstFrame = false;
}

auto MilkdropPreset::OutputTexture() const -> std::shared_ptr<Renderer::Texture>
{
    // the composited image is always stored in the "current" framebuffer after a frame is rendered.
    return m_framebuffer.GetColorAttachmentTexture(m_currentFrameBuffer, 0);
}

void MilkdropPreset::DrawInitialImage(const std::shared_ptr<Renderer::Texture>& image, const Renderer::RenderContext& renderContext)
{
    m_framebuffer.SetSize(renderContext.viewportSizeX, renderContext.viewportSizeY);

    // Render to previous framebuffer, as this is the image used to draw the next frame on.
    m_flipTexture.Draw(*renderContext.shaderCache, image, m_framebuffer, m_previousFrameBuffer);
}

void MilkdropPreset::BindFramebuffer()
{
    if (m_framebuffer.Width() > 0 && m_framebuffer.Height() > 0)
    {
        m_framebuffer.BindDraw(m_previousFrameBuffer);
    }
}

void MilkdropPreset::RecompileShaders()
{
    m_perPixelMesh.CompileWarpShader(m_state);
    m_finalComposite.CompileCompositeShader(m_state);
}

void MilkdropPreset::PerFrameUpdate()
{
    m_perFrameContext.LoadStateVariables(m_state);
    m_perPixelContext.LoadStateReadOnlyVariables(m_state, m_perFrameContext);

    const double preExecDecay = *m_perFrameContext.decay;
    m_perFrameContext.ExecutePerFrameCode();

    // If per_frame script overwrote decay directly (e.g. `decay = 0.96`), normalize it to real-time FPS
    // so high framerates never cause premature fade-to-black.
    const double postExecDecay = *m_perFrameContext.decay;
    const double instantFps = static_cast<double>(m_state.renderContext.fps);
    if (std::abs(postExecDecay - preExecDecay) > 1e-6 && instantFps > 1.0 && postExecDecay > 0.0 && postExecDecay < 1.0)
    {
        *m_perFrameContext.decay = static_cast<PRJM_EVAL_F>(std::pow(postExecDecay, 60.0 / instantFps));
    }

    // Kinematic Slew-Rate Limiter (Anti-saltos bruscos manteniendo el efecto original):
    // Smooths out single-frame camera teleport spikes caused by unbuffered modulo wrap or abrupt jump logic
    if (m_perFrameContext.dx && m_perFrameContext.dy && m_perFrameContext.zoom && m_perFrameContext.rot)
    {
        double curDx = *m_perFrameContext.dx;
        double curDy = *m_perFrameContext.dy;
        double curZoom = *m_perFrameContext.zoom;
        double curRot = *m_perFrameContext.rot;
        double curCx = m_perFrameContext.cx ? *m_perFrameContext.cx : 0.5;
        double curCy = m_perFrameContext.cy ? *m_perFrameContext.cy : 0.5;

        if (!std::isfinite(curDx)) curDx = 0.0;
        if (!std::isfinite(curDy)) curDy = 0.0;
        if (!std::isfinite(curZoom) || curZoom <= 0.001) curZoom = 1.0;
        if (!std::isfinite(curRot)) curRot = 0.0;
        if (!std::isfinite(curCx)) curCx = 0.5;
        if (!std::isfinite(curCy)) curCy = 0.5;

        if (m_hasPrevCameraMotion)
        {
            double const deltaDx = curDx - m_prevDx;
            double const deltaDy = curDy - m_prevDy;
            double const deltaZoom = curZoom - m_prevZoom;
            double const deltaRot = curRot - m_prevRot;
            double const deltaCx = curCx - m_prevCx;
            double const deltaCy = curCy - m_prevCy;

            constexpr double maxDeltaPos = 0.12;
            constexpr double maxDeltaZoom = 0.35;
            constexpr double maxDeltaRot = 0.25;
            constexpr double maxDeltaCenter = 0.20;

            if (std::abs(deltaDx) > maxDeltaPos)
            {
                curDx = m_prevDx + std::copysign(maxDeltaPos, deltaDx) + (deltaDx - std::copysign(maxDeltaPos, deltaDx)) * 0.15;
            }
            if (std::abs(deltaDy) > maxDeltaPos)
            {
                curDy = m_prevDy + std::copysign(maxDeltaPos, deltaDy) + (deltaDy - std::copysign(maxDeltaPos, deltaDy)) * 0.15;
            }
            if (std::abs(deltaZoom) > maxDeltaZoom)
            {
                curZoom = m_prevZoom + std::copysign(maxDeltaZoom, deltaZoom) + (deltaZoom - std::copysign(maxDeltaZoom, deltaZoom)) * 0.15;
            }
            if (std::abs(deltaRot) > maxDeltaRot)
            {
                curRot = m_prevRot + std::copysign(maxDeltaRot, deltaRot) + (deltaRot - std::copysign(maxDeltaRot, deltaRot)) * 0.15;
            }
            if (std::abs(deltaCx) > maxDeltaCenter)
            {
                curCx = m_prevCx + std::copysign(maxDeltaCenter, deltaCx) + (deltaCx - std::copysign(maxDeltaCenter, deltaCx)) * 0.15;
            }
            if (std::abs(deltaCy) > maxDeltaCenter)
            {
                curCy = m_prevCy + std::copysign(maxDeltaCenter, deltaCy) + (deltaCy - std::copysign(maxDeltaCenter, deltaCy)) * 0.15;
            }
        }

        m_prevDx = curDx;
        m_prevDy = curDy;
        m_prevZoom = curZoom;
        m_prevRot = curRot;
        m_prevCx = curCx;
        m_prevCy = curCy;
        m_hasPrevCameraMotion = true;

        *m_perFrameContext.dx = static_cast<PRJM_EVAL_F>(curDx);
        *m_perFrameContext.dy = static_cast<PRJM_EVAL_F>(curDy);
        *m_perFrameContext.zoom = static_cast<PRJM_EVAL_F>(curZoom);
        *m_perFrameContext.rot = static_cast<PRJM_EVAL_F>(curRot);
        if (m_perFrameContext.cx) *m_perFrameContext.cx = static_cast<PRJM_EVAL_F>(curCx);
        if (m_perFrameContext.cy) *m_perFrameContext.cy = static_cast<PRJM_EVAL_F>(curCy);
    }

    m_perPixelContext.LoadPerFrameQVariables(m_state, m_perFrameContext);

    // Normalize video echo zoom and alpha to nominal 60 FPS
    if (m_perFrameContext.echo_zoom && instantFps > 1.0)
    {
        const double rawEchoZoom = *m_perFrameContext.echo_zoom;
        if (std::abs(rawEchoZoom - 1.0) > 1e-4 && rawEchoZoom > 0.0)
        {
            *m_perFrameContext.echo_zoom = static_cast<PRJM_EVAL_F>(std::pow(rawEchoZoom, 60.0 / instantFps));
        }
    }
    if (m_perFrameContext.echo_alpha && instantFps > 1.0)
    {
        const double rawEchoAlpha = *m_perFrameContext.echo_alpha;
        if (rawEchoAlpha > 0.0 && rawEchoAlpha < 1.0)
        {
            *m_perFrameContext.echo_alpha = static_cast<PRJM_EVAL_F>(std::pow(rawEchoAlpha, 60.0 / instantFps));
        }
    }

    // Clamp gamma and echo zoom values
    *m_perFrameContext.gamma = std::max(0.0, std::min(8.0, *m_perFrameContext.gamma));
    *m_perFrameContext.echo_zoom = std::max(0.001, std::min(1000.0, *m_perFrameContext.echo_zoom));

    m_state.UpdateUniformConstants(m_perFrameContext);
}

void MilkdropPreset::Load(const std::string& pathname)
{
    LOG_DEBUG("[MilkdropPreset] Loading preset from file \"" + pathname + "\".")

    SetFilename(ParseFilename(pathname));

    PresetFileParser parser;

    if (!parser.Read(pathname))
    {
        const std::string error = "[MilkdropPreset] Could not parse preset file \"" + pathname + "\".";
        LOG_ERROR(error)
        throw MilkdropPresetLoadException(error);
    }

    InitializePreset(parser);
}

void MilkdropPreset::Load(std::istream& stream)
{
    LOG_DEBUG("[MilkdropPreset] Loading preset from stream.");

    PresetFileParser parser;

    if (!parser.Read(stream))
    {
        const std::string error =  "[MilkdropPreset] Could not parse preset data.";
        LOG_ERROR(error)
        throw MilkdropPresetLoadException(error);
    }

    InitializePreset(parser);
}

void MilkdropPreset::InitializePreset(PresetFileParser& parsedFile)
{
    // Create the offscreen rendering surfaces.
    m_motionVectorUVMap = std::make_shared<Renderer::TextureAttachment>(GL_RG16F, GL_RG, GL_FLOAT, 0, 0);
    m_framebuffer.CreateColorAttachment(0, 0); // Main image 1
    m_framebuffer.CreateColorAttachment(1, 0); // Main image 2

    Renderer::Framebuffer::Unbind();

    // Load global init variables into the state
    m_state.Initialize(parsedFile);

    // Register code context variables
    m_perFrameContext.RegisterBuiltinVariables();
    m_perPixelContext.RegisterBuiltinVariables();

    // Custom waveforms:
    for (int i = 0; i < CustomWaveformCount; i++)
    {
        auto wave = std::make_unique<CustomWaveform>(m_state);
        wave->Initialize(parsedFile, i);
        m_customWaveforms[i] = std::move(wave);
    }

    // Custom shapes:
    for (int i = 0; i < CustomShapeCount; i++)
    {
        auto shape = std::make_unique<CustomShape>(m_state);
        shape->Initialize(parsedFile, i);
        m_customShapes[i] = std::move(shape);
    }

    // Preload shaders
    LoadShaderCode();
}

void MilkdropPreset::CompileCodeAndRunInitExpressions()
{
    m_hasPrevCameraMotion = false;

    // Per-frame init and code
    m_perFrameContext.LoadStateVariables(m_state);
    m_perFrameContext.EvaluateInitCode(m_state);
    m_perFrameContext.CompilePerFrameCode(m_state.perFrameCode);

    // Per-vertex code
    m_perPixelContext.CompilePerPixelCode(m_state.perPixelCode);

    for (int i = 0; i < CustomWaveformCount; i++)
    {
        auto& wave = m_customWaveforms[i];
        wave->CompileCodeAndRunInitExpressions(m_perFrameContext);
    }

    for (int i = 0; i < CustomShapeCount; i++)
    {
        auto& shape = m_customShapes[i];
        shape->CompileCodeAndRunInitExpressions();
    }
}

void MilkdropPreset::LoadShaderCode()
{
    m_perPixelMesh.LoadWarpShader(m_state);
    m_finalComposite.LoadCompositeShader(m_state);
}

auto MilkdropPreset::ParseFilename(const std::string& filename) -> std::string
{
    const std::size_t start = filename.find_last_of('/');

    if (start == std::string::npos || start >= (filename.length() - 1))
    {
        return "";
    }

    return filename.substr(start + 1, filename.length());
}


} // namespace MilkdropPreset
} // namespace libprojectM
