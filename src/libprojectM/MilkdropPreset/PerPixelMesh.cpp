#include <vector>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

#include "PerPixelMesh.hpp"

#include "MilkdropShader.hpp"
#include "MilkdropStaticShaders.hpp"
#include "PerFrameContext.hpp"
#include "PerPixelContext.hpp"
#include "PresetState.hpp"

#include <Logging.hpp>
#include <Renderer/BlendMode.hpp>
#include <Renderer/ShaderCache.hpp>

#include <algorithm>
#include <cmath>

namespace libprojectM {
namespace MilkdropPreset {

PerPixelMesh::PerPixelMesh()
    : m_warpMesh(Renderer::VertexBufferUsage::StreamDraw)
{
    m_warpMesh.SetRenderPrimitiveType(Renderer::Mesh::PrimitiveType::Triangles);

    m_warpMesh.Bind();
    m_radiusAngleBuffer.Bind();
    m_zoomRotWarpBuffer.Bind();
    m_centerBuffer.Bind();
    m_distanceBuffer.Bind();
    m_stretchBuffer.Bind();

    m_radiusAngleBuffer.InitializeAttributePointer(3);
    m_zoomRotWarpBuffer.InitializeAttributePointer(4);
    m_centerBuffer.InitializeAttributePointer(5);
    m_distanceBuffer.InitializeAttributePointer(6);
    m_stretchBuffer.InitializeAttributePointer(7);

    Renderer::VertexBuffer<Renderer::Point>::SetEnableAttributeArray(3, true);
    Renderer::VertexBuffer<Renderer::Point>::SetEnableAttributeArray(4, true);
    Renderer::VertexBuffer<Renderer::Point>::SetEnableAttributeArray(5, true);
    Renderer::VertexBuffer<Renderer::Point>::SetEnableAttributeArray(6, true);
    Renderer::VertexBuffer<Renderer::Point>::SetEnableAttributeArray(7, true);

    Renderer::Mesh::Unbind();
}

void PerPixelMesh::LoadWarpShader(const PresetState& presetState)
{
    // Compile warp shader if preset specifies one.
    if (presetState.warpShaderVersion > 0)
    {
        if (!presetState.warpShader.empty())
        {
            try
            {
                m_warpShader = std::make_unique<MilkdropShader>(MilkdropShader::ShaderType::WarpShader);
                m_warpShader->LoadCode(presetState.warpShader);
                LOG_DEBUG("[PerPixelMesh] Successfully loaded preset warp shader code.");
            }
            catch (Renderer::ShaderException& ex)
            {
                LOG_ERROR("[PerPixelMesh] Error loading warp shader code:" + ex.message());
                LOG_DEBUG("[PerPixelMesh] Warp shader code:\n" + presetState.warpShader);
                m_warpShader.reset();
            }
        }
    }
}

void PerPixelMesh::CompileWarpShader(PresetState& presetState)
{
    if (m_warpShader)
    {
        try
        {
            m_warpShader->LoadTexturesAndCompile(presetState);
            LOG_DEBUG("[PerPixelMesh] Successfully compiled warp shader code.");
        }
        catch (const std::exception& ex)
        {
            LOG_ERROR(std::string("[PerPixelMesh] Error compiling warp shader code: ") + ex.what());
            m_warpShader.reset();
        }
        catch (...)
        {
            LOG_ERROR("[PerPixelMesh] Unknown error compiling warp shader code.");
            m_warpShader.reset();
        }
    }
}

void PerPixelMesh::Draw(const PresetState& presetState,
                        const PerFrameContext& perFrameContext,
                        PerPixelContext& perPixelContext)
{
    if (presetState.renderContext.viewportSizeX == 0 ||
        presetState.renderContext.viewportSizeY == 0 ||
        presetState.renderContext.perPixelMeshX == 0 ||
        presetState.renderContext.perPixelMeshY == 0)
    {
        return;
    }

    // Initialize or recreate the mesh (if grid size changed)
    InitializeMesh(presetState);

    // Calculate the dynamic movement values
    CalculateMesh(presetState, perFrameContext, perPixelContext);

    // Render the resulting mesh.
    WarpedBlit(presetState, perFrameContext);
}

void PerPixelMesh::InitializeMesh(const PresetState& presetState)
{
    const bool modernMode = MilkdropShader::IsCurlNoiseEnabled();

    if (m_gridSizeX != presetState.renderContext.perPixelMeshX ||
        m_gridSizeY != presetState.renderContext.perPixelMeshY)
    {
        m_gridSizeX = presetState.renderContext.perPixelMeshX;
        m_gridSizeY = presetState.renderContext.perPixelMeshY;

        // Grid size has changed, resize buffers accordingly
        const size_t vertexCount = (m_gridSizeX + 1) * (m_gridSizeY + 1);

        m_warpMesh.SetVertexCount(vertexCount);
        m_radiusAngleBuffer.Resize(vertexCount);
        m_zoomRotWarpBuffer.Resize(vertexCount);
        m_centerBuffer.Resize(vertexCount);
        m_distanceBuffer.Resize(vertexCount);
        m_stretchBuffer.Resize(vertexCount);

        m_warpMesh.Indices().Resize(m_gridSizeX * m_gridSizeY * 6);
        m_hasCachedUniforms = false;
    }
    else if (m_viewportWidth == presetState.renderContext.viewportSizeX &&
             m_viewportHeight == presetState.renderContext.viewportSizeY &&
             m_cachedModernMode == modernMode)
    {
        // Nothing changed, just go on to the dynamic calculation.
        return;
    }

    m_cachedModernMode = modernMode;
    const float aspectX = presetState.renderContext.aspectX;
    const float aspectY = presetState.renderContext.aspectY;

    // Either viewport size, mesh size, or rendering mode changed: reinitialize the vertices.
    auto& vertices = m_warpMesh.Vertices();
    int vertexIndex{0};
    for (int gridY = 0; gridY <= m_gridSizeY; gridY++)
    {
        for (int gridX = 0; gridX <= m_gridSizeX; gridX++)
        {
            const float x = static_cast<float>(gridX) / static_cast<float>(m_gridSizeX) * 2.0f - 1.0f;
            const float y = static_cast<float>(gridY) / static_cast<float>(m_gridSizeY) * 2.0f - 1.0f;
            vertices[vertexIndex] = {x, y};

            const float r = hypotf(x * aspectX, y * aspectY);
            if (modernMode)
            {
                // Enhanced 3D Mode: Conformal Logarithmic + Hyperbolic Poincaré Disk geodesic mapping.
                // Smoothly curves radial space into a 3D stereographic dome/tunnel with 0 GPU overhead.
                const float rConf = std::log1p(r * 1.718281828f);
                const float rPoincare = (2.0f * r) / (1.0f + r * r);
                m_radiusAngleBuffer[vertexIndex].radius = 0.65f * rConf + 0.35f * rPoincare;
            }
            else
            {
                // Original Vintage 2001 Mode ("Rústico"): Flat Euclidean radial metric.
                m_radiusAngleBuffer[vertexIndex].radius = r;
            }
            if (gridY == m_gridSizeY / 2 && gridX == m_gridSizeX / 2)
            {
                m_radiusAngleBuffer[vertexIndex].angle = 0.0f;
            }
            else
            {
                m_radiusAngleBuffer[vertexIndex].angle = atan2f(y * aspectY, x * aspectX);
            }

            vertexIndex++;
        }
    }

    // Generate triangle lists for drawing the main warp mesh
    int vertexListIndex{0};
    for (int quadrant = 0; quadrant < 4; quadrant++)
    {
        for (int slice = 0; slice < m_gridSizeY / 2; slice++)
        {
            for (int gridX = 0; gridX < m_gridSizeX / 2; gridX++)
            {
                int xReference = gridX;
                int yReference = slice;

                if ((quadrant & 1) != 0)
                {
                    xReference = m_gridSizeX - 1 - xReference;
                }
                if ((quadrant & 2) != 0)
                {
                    yReference = m_gridSizeY - 1 - yReference;
                }

                int const vertex = xReference + yReference * (m_gridSizeX + 1);

                // 0 - 1      3
                //   /      /
                // 2      4 - 5
                m_warpMesh.Indices()[vertexListIndex++] = vertex;
                m_warpMesh.Indices()[vertexListIndex++] = vertex + 1;
                m_warpMesh.Indices()[vertexListIndex++] = vertex + m_gridSizeX + 1;
                m_warpMesh.Indices()[vertexListIndex++] = vertex + 1;
                m_warpMesh.Indices()[vertexListIndex++] = vertex + m_gridSizeX + 1;
                m_warpMesh.Indices()[vertexListIndex++] = vertex + m_gridSizeX + 2;
            }
        }
    }

    // Upload static mesh coordinates and polar radius/angle buffers once upon initialization
    m_warpMesh.Update();
    m_radiusAngleBuffer.Update();
    m_viewportWidth = presetState.renderContext.viewportSizeX;
    m_viewportHeight = presetState.renderContext.viewportSizeY;
}



void PerPixelMesh::CalculateMesh(const PresetState& presetState, const PerFrameContext& perFrameContext, PerPixelContext& perPixelContext)
{
    float zoom = static_cast<float>(*perFrameContext.zoom);
    float zoomExp = static_cast<float>(*perFrameContext.zoomexp);
    float rot = static_cast<float>(*perFrameContext.rot);
    float warp = static_cast<float>(*perFrameContext.warp);
    float cx = static_cast<float>(*perFrameContext.cx);
    float cy = static_cast<float>(*perFrameContext.cy);
    float dx = static_cast<float>(*perFrameContext.dx);
    float dy = static_cast<float>(*perFrameContext.dy);
    float sx = static_cast<float>(*perFrameContext.sx);
    float sy = static_cast<float>(*perFrameContext.sy);

    // Time-step kinematic independence (anti-"Pac-Man Turbo" protection):
    // When running above or below 60 FPS (e.g. 90Hz/120Hz displays, or heavy 4K shaders),
    // normalize per-frame iterative advection steps so angular, linear, and exponential
    // velocities per real-world second remain strictly constant.
    const float currentFps = presetState.renderContext.fps;
    const float kappa = (currentFps > 5.0f) ? std::clamp(60.0f / currentFps, 0.40f, 2.50f) : 1.0f;
    const bool applyTimeScale = (std::abs(kappa - 1.0f) > 0.03f);

    if (applyTimeScale)
    {
        if (zoom > 0.001f) zoom = std::pow(zoom, kappa);
        rot *= kappa;
        warp *= kappa;
        dx *= kappa;
        dy *= kappa;
        if (sx > 0.001f) sx = std::pow(sx, kappa);
        if (sy > 0.001f) sy = std::pow(sy, kappa);
    }

    int totalVertices = (m_gridSizeY + 1) * (m_gridSizeX + 1);

    if (!perPixelContext.perPixelCodeHandle)
    {
        if (m_hasCachedUniforms &&
            m_cachedTotalVertices == totalVertices &&
            m_cachedZoom == zoom &&
            m_cachedZoomExp == zoomExp &&
            m_cachedRot == rot &&
            m_cachedWarp == warp &&
            m_cachedCx == cx &&
            m_cachedCy == cy &&
            m_cachedDx == dx &&
            m_cachedDy == dy &&
            m_cachedSx == sx &&
            m_cachedSy == sy)
        {
            return;
        }

        // Broadcast uniform per-frame values directly into VBOs.
        // Avoids VAO attribute toggling bugs that leak into custom shapes / 3D objects.
        ZoomRotWarp zrw{zoom, zoomExp, rot, warp};
        Renderer::Point center{cx, cy};
        Renderer::Point dist{dx, dy};
        Renderer::Point stretch{sx, sy};

        std::fill(m_zoomRotWarpBuffer.Get().begin(), m_zoomRotWarpBuffer.Get().end(), zrw);
        std::fill(m_centerBuffer.Get().begin(), m_centerBuffer.Get().end(), center);
        std::fill(m_distanceBuffer.Get().begin(), m_distanceBuffer.Get().end(), dist);
        std::fill(m_stretchBuffer.Get().begin(), m_stretchBuffer.Get().end(), stretch);

        m_zoomRotWarpBuffer.Update();
        m_centerBuffer.Update();
        m_distanceBuffer.Update();
        m_stretchBuffer.Update();

        m_hasCachedUniforms = true;
        m_cachedTotalVertices = totalVertices;
        m_cachedZoom = zoom;
        m_cachedZoomExp = zoomExp;
        m_cachedRot = rot;
        m_cachedWarp = warp;
        m_cachedCx = cx;
        m_cachedCy = cy;
        m_cachedDx = dx;
        m_cachedDy = dy;
        m_cachedSx = sx;
        m_cachedSy = sy;
        return;
    }

    m_hasCachedUniforms = false;
    auto& vertices = m_warpMesh.Vertices();

    const double baseZoom = *perFrameContext.zoom;
    const double baseZoomExp = *perFrameContext.zoomexp;
    const double baseRot = *perFrameContext.rot;
    const double baseWarp = *perFrameContext.warp;
    const double baseCx = *perFrameContext.cx;
    const double baseCy = *perFrameContext.cy;
    const double baseDx = *perFrameContext.dx;
    const double baseDy = *perFrameContext.dy;
    const double baseSx = *perFrameContext.sx;
    const double baseSy = *perFrameContext.sy;

    for (int vertex = 0; vertex < totalVertices; ++vertex)
    {
        auto& curVertex = vertices[vertex];
        auto& curRadiusAngle = m_radiusAngleBuffer[vertex];
        auto& curZoomRotWarp = m_zoomRotWarpBuffer[vertex];
        auto& curCenter = m_centerBuffer[vertex];
        auto& curDistance = m_distanceBuffer[vertex];
        auto& curStretch = m_stretchBuffer[vertex];

        *perPixelContext.x = static_cast<double>(curVertex.X() * 0.5f + 0.5f);
        *perPixelContext.y = static_cast<double>(curVertex.Y() * 0.5f + 0.5f);
        *perPixelContext.rad = static_cast<double>(curRadiusAngle.radius);
        *perPixelContext.ang = static_cast<double>(-curRadiusAngle.angle);
        *perPixelContext.zoom = baseZoom;
        *perPixelContext.zoomexp = baseZoomExp;
        *perPixelContext.rot = baseRot;
        *perPixelContext.warp = baseWarp;
        *perPixelContext.cx = baseCx;
        *perPixelContext.cy = baseCy;
        *perPixelContext.dx = baseDx;
        *perPixelContext.dy = baseDy;
        *perPixelContext.sx = baseSx;
        *perPixelContext.sy = baseSy;

        perPixelContext.ExecutePerPixelCode();

        float vZoom = static_cast<float>(*perPixelContext.zoom);
        float vZoomExp = static_cast<float>(*perPixelContext.zoomexp);
        float vRot = static_cast<float>(*perPixelContext.rot);
        float vWarp = static_cast<float>(*perPixelContext.warp);
        float vDx = static_cast<float>(*perPixelContext.dx);
        float vDy = static_cast<float>(*perPixelContext.dy);
        float vSx = static_cast<float>(*perPixelContext.sx);
        float vSy = static_cast<float>(*perPixelContext.sy);

        if (applyTimeScale)
        {
            if (vZoom > 0.001f) vZoom = std::pow(vZoom, kappa);
            vRot *= kappa;
            vWarp *= kappa;
            vDx *= kappa;
            vDy *= kappa;
            if (vSx > 0.001f) vSx = std::pow(vSx, kappa);
            if (vSy > 0.001f) vSy = std::pow(vSy, kappa);
        }

        curZoomRotWarp.zoom = vZoom;
        curZoomRotWarp.zoomExp = vZoomExp;
        curZoomRotWarp.rot = vRot;
        curZoomRotWarp.warp = vWarp;
        curCenter = {static_cast<float>(*perPixelContext.cx), static_cast<float>(*perPixelContext.cy)};
        curDistance = {vDx, vDy};
        curStretch = {vSx, vSy};
    }

    // Phase unwrapping & mesh seam smoothing:
    // Across the polar branch cut (y = 0, x < 0), angle jumps by 2*pi.
    // If a preset formula multiplies or adds 'ang' to 'rot', adjacent vertices
    // across the horizontal seam differ by > pi. In GPU hardware interpolation,
    // this causes an inverted 360-degree polygon flip and visual screen tear.
    // Unwrapping the phase ensures shortest-arc interpolation, completely eliminating
    // the horizontal seam jump while preserving exact rotation angles.
    int const midY = m_gridSizeY / 2;
    int const midX = m_gridSizeX / 2;
    if (midY > 0)
    {
        for (int gx = 0; gx < midX; ++gx)
        {
            int const upperIdx = gx + midY * (m_gridSizeX + 1);
            int const lowerIdx = gx + (midY - 1) * (m_gridSizeX + 1);

            float const diffRot = m_zoomRotWarpBuffer[upperIdx].rot - m_zoomRotWarpBuffer[lowerIdx].rot;
            if (diffRot > 3.14159265f)
            {
                m_zoomRotWarpBuffer[upperIdx].rot -= 6.2831853f;
            }
            else if (diffRot < -3.14159265f)
            {
                m_zoomRotWarpBuffer[upperIdx].rot += 6.2831853f;
            }
        }
    }

    m_zoomRotWarpBuffer.Update();
    m_centerBuffer.Update();
    m_distanceBuffer.Update();
    m_stretchBuffer.Update();
}

void PerPixelMesh::WarpedBlit(const PresetState& presetState,
                              const PerFrameContext& perFrameContext)
{
    if (!m_warpShader)
    {
        // Warp calculations for default fallback shader
        float const warpTime = presetState.renderContext.time * presetState.warpAnimSpeed;
        float const warpScaleInverse = 1.0f / presetState.warpScale;
        glm::vec4 const warpFactors{
            11.68f + 4.0f * cosf(warpTime * 1.413f + 10),
            8.77f + 3.0f * cosf(warpTime * 1.113f + 7),
            10.54f + 3.0f * cosf(warpTime * 1.233f + 3),
            11.49f + 4.0f * cosf(warpTime * 0.933f + 5),
        };

        // Texel alignment
        glm::vec2 const texelOffsets{presetState.renderContext.texelOffsetX / static_cast<float>(presetState.renderContext.viewportSizeX),
                                     presetState.renderContext.texelOffsetY / static_cast<float>(presetState.renderContext.viewportSizeY)};

        float decay = std::min(static_cast<float>(*perFrameContext.decay), 1.0f);

        auto perPixelMeshShader = GetDefaultWarpShader(presetState);
        perPixelMeshShader->Bind();
        perPixelMeshShader->SetUniformMat4x4("vertex_transformation", PresetState::orthogonalProjection);
        perPixelMeshShader->SetUniformInt("texture_sampler", 0);
        perPixelMeshShader->SetUniformFloat4("aspect", {presetState.renderContext.aspectX,
                                                        presetState.renderContext.aspectY,
                                                        presetState.renderContext.invAspectX,
                                                        presetState.renderContext.invAspectY});
        perPixelMeshShader->SetUniformFloat("warpTime", warpTime);
        perPixelMeshShader->SetUniformFloat("warpScaleInverse", warpScaleInverse);
        perPixelMeshShader->SetUniformFloat4("warpFactors", warpFactors);
        perPixelMeshShader->SetUniformFloat2("texelOffset", texelOffsets);
        perPixelMeshShader->SetUniformFloat("decay", decay);
        perPixelMeshShader->SetUniformInt("u_curlNoise", MilkdropShader::IsCurlNoiseEnabled() ? 1 : 0);
        perPixelMeshShader->SetUniformInt("u_catmullRom", MilkdropShader::IsCatmullRomEnabled() ? 1 : 0);
    }
    else
    {
        m_warpShader->LoadVariables(presetState, perFrameContext);
    }

    assert(!presetState.mainTexture.expired());
    presetState.mainTexture.lock()->Bind(0);

    // Set wrap mode and bind the sampler to get interpolation right.
    if (*perFrameContext.wrap > 0.0001f)
    {
        m_perPixelSampler.WrapMode(GL_REPEAT);
    }
    else
    {
        m_perPixelSampler.WrapMode(GL_CLAMP_TO_EDGE);
    }
    m_perPixelSampler.Bind(0);

    m_warpMesh.Draw();

    Renderer::Mesh::Unbind();
    Renderer::Sampler::Unbind(0);
    Renderer::Shader::Unbind();
}

auto PerPixelMesh::GetDefaultWarpShader(const PresetState& presetState) -> std::shared_ptr<Renderer::Shader>
{
    auto perPixelMeshShader = m_perPixelMeshShader.lock();
    if (perPixelMeshShader)
    {
        return perPixelMeshShader;
    }

    perPixelMeshShader = presetState.renderContext.shaderCache->Get("milkdrop_default_warp_shader");
    if (perPixelMeshShader)
    {
        return perPixelMeshShader;
    }

    auto staticShaders = libprojectM::MilkdropPreset::MilkdropStaticShaders::Get();

    perPixelMeshShader = std::make_shared<Renderer::Shader>();
    perPixelMeshShader->CompileProgram(staticShaders->GetPresetWarpVertexShader(),
                                       staticShaders->GetPresetWarpFragmentShader());

    presetState.renderContext.shaderCache->Insert("milkdrop_default_warp_shader", perPixelMeshShader);
    m_perPixelMeshShader = perPixelMeshShader;

    return perPixelMeshShader;
}

} // namespace MilkdropPreset
} // namespace libprojectM
