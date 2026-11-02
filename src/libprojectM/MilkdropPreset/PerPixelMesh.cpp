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

auto PerPixelMesh::WarpShaderUsesDiscard() const -> bool
{
    return m_warpShader && m_warpShader->UsesDiscard();
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
        m_gridCache = {};
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



auto PerPixelMesh::EvaluateGridCPU(int gridSizeX,
                                   int gridSizeY,
                                   float fps,
                                   const Renderer::Point* vertices,
                                   const RadiusAngle* radiusAngles,
                                   const PerFrameContext& perFrameContext,
                                   PerPixelContext& perPixelContext,
                                   ZoomRotWarp* outZoomRotWarp,
                                   Renderer::Point* outCenter,
                                   Renderer::Point* outDistance,
                                   Renderer::Point* outStretch,
                                   GridCacheState& cache) -> DirtyBuffers
{
    const int totalVertices = (gridSizeX + 1) * (gridSizeY + 1);
    if (totalVertices <= 0)
    {
        cache.lastEvalCount = 0;
        return {};
    }

    // Time-step kinematic independence (anti-"Pac-Man Turbo" protection):
    // Normalize per-frame iterative advection steps so angular, linear, and exponential
    // velocities per real-world second remain strictly constant across 30Hz..120Hz displays.
    const float kappa = (fps > 5.0f) ? std::clamp(60.0f / fps, 0.40f, 2.50f) : 1.0f;
    const bool applyTimeScale = (std::abs(kappa - 1.0f) > 0.03f);

    if (cache.cachedTotalVertices != totalVertices)
    {
        cache.hasCachedZoomRotWarp = false;
        cache.hasCachedCenter = false;
        cache.hasCachedDistance = false;
        cache.hasCachedStretch = false;
        cache.cachedTotalVertices = totalVertices;
    }

    auto broadcastUniforms = [&](float z, float ze, float r, float w,
                                 float cx, float cy,
                                 float dx, float dy,
                                 float sx, float sy) -> DirtyBuffers {
        DirtyBuffers dirty{};
        if (!cache.hasCachedZoomRotWarp ||
            cache.cachedZoom != z ||
            cache.cachedZoomExp != ze ||
            cache.cachedRot != r ||
            cache.cachedWarp != w)
        {
            std::fill_n(outZoomRotWarp, totalVertices, ZoomRotWarp{z, ze, r, w});
            cache.cachedZoom = z;
            cache.cachedZoomExp = ze;
            cache.cachedRot = r;
            cache.cachedWarp = w;
            cache.hasCachedZoomRotWarp = true;
            dirty.zoomRotWarp = true;
        }
        if (!cache.hasCachedCenter || cache.cachedCx != cx || cache.cachedCy != cy)
        {
            std::fill_n(outCenter, totalVertices, Renderer::Point{cx, cy});
            cache.cachedCx = cx;
            cache.cachedCy = cy;
            cache.hasCachedCenter = true;
            dirty.center = true;
        }
        if (!cache.hasCachedDistance || cache.cachedDx != dx || cache.cachedDy != dy)
        {
            std::fill_n(outDistance, totalVertices, Renderer::Point{dx, dy});
            cache.cachedDx = dx;
            cache.cachedDy = dy;
            cache.hasCachedDistance = true;
            dirty.distance = true;
        }
        if (!cache.hasCachedStretch || cache.cachedSx != sx || cache.cachedSy != sy)
        {
            std::fill_n(outStretch, totalVertices, Renderer::Point{sx, sy});
            cache.cachedSx = sx;
            cache.cachedSy = sy;
            cache.hasCachedStretch = true;
            dirty.stretch = true;
        }
        return dirty;
    };

    const PRJM_EVAL_F baseZoom = *perFrameContext.zoom;
    const PRJM_EVAL_F baseZoomExp = *perFrameContext.zoomexp;
    const PRJM_EVAL_F baseRot = *perFrameContext.rot;
    const PRJM_EVAL_F baseWarp = *perFrameContext.warp;
    const PRJM_EVAL_F baseCx = *perFrameContext.cx;
    const PRJM_EVAL_F baseCy = *perFrameContext.cy;
    const PRJM_EVAL_F baseDx = *perFrameContext.dx;
    const PRJM_EVAL_F baseDy = *perFrameContext.dy;
    const PRJM_EVAL_F baseSx = *perFrameContext.sx;
    const PRJM_EVAL_F baseSy = *perFrameContext.sy;

    float uZoom = static_cast<float>(baseZoom);
    const float uZoomExp = static_cast<float>(baseZoomExp);
    float uRot = static_cast<float>(baseRot);
    float uWarp = static_cast<float>(baseWarp);
    const float uCx = static_cast<float>(baseCx);
    const float uCy = static_cast<float>(baseCy);
    float uDx = static_cast<float>(baseDx);
    float uDy = static_cast<float>(baseDy);
    float uSx = static_cast<float>(baseSx);
    float uSy = static_cast<float>(baseSy);

    if (applyTimeScale)
    {
        if (uZoom > 0.001f) uZoom = std::pow(uZoom, kappa);
        uRot *= kappa;
        uWarp *= kappa;
        uDx *= kappa;
        uDy *= kappa;
        if (uSx > 0.001f) uSx = std::pow(uSx, kappa);
        if (uSy > 0.001f) uSy = std::pow(uSy, kappa);
    }

    // Fast Path 1: No per_pixel equations present in preset.
    if (!perPixelContext.perPixelCodeHandle)
    {
        cache.lastEvalCount = 0;
        return broadcastUniforms(uZoom, uZoomExp, uRot, uWarp, uCx, uCy, uDx, uDy, uSx, uSy);
    }

    // Fast Path 2: Spatially-invariant per_pixel equations (1x/frame evaluation instead of 3,185x/frame).
    if (perPixelContext.IsSpatiallyInvariant())
    {
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
        cache.lastEvalCount = 1;

        float z = static_cast<float>(*perPixelContext.zoom);
        const float ze = static_cast<float>(*perPixelContext.zoomexp);
        float r = static_cast<float>(*perPixelContext.rot);
        float w = static_cast<float>(*perPixelContext.warp);
        const float cx = static_cast<float>(*perPixelContext.cx);
        const float cy = static_cast<float>(*perPixelContext.cy);
        float dx = static_cast<float>(*perPixelContext.dx);
        float dy = static_cast<float>(*perPixelContext.dy);
        float sx = static_cast<float>(*perPixelContext.sx);
        float sy = static_cast<float>(*perPixelContext.sy);

        if (applyTimeScale)
        {
            if (z > 0.001f) z = std::pow(z, kappa);
            r *= kappa;
            w *= kappa;
            dx *= kappa;
            dy *= kappa;
            if (sx > 0.001f) sx = std::pow(sx, kappa);
            if (sy > 0.001f) sy = std::pow(sy, kappa);
        }

        return broadcastUniforms(z, ze, r, w, cx, cy, dx, dy, sx, sy);
    }

    // Path 3: Spatially-varying per-vertex execution with minimal register resets & per-VBO dirty tracking.
    cache.lastEvalCount = static_cast<size_t>(totalVertices);
    const auto& u = perPixelContext.Usage();

    const bool mutZRW = u.writesZoom || u.writesZoomExp || u.writesRot || u.writesWarp;
    const bool mutCenter = u.writesCx || u.writesCy;
    const bool mutDist = u.writesDx || u.writesDy;
    const bool mutStretch = u.writesSx || u.writesSy;

    DirtyBuffers dirty{};
    if (mutZRW)
    {
        cache.hasCachedZoomRotWarp = false;
        dirty.zoomRotWarp = true;
    }
    else if (!cache.hasCachedZoomRotWarp ||
             cache.cachedZoom != uZoom ||
             cache.cachedZoomExp != uZoomExp ||
             cache.cachedRot != uRot ||
             cache.cachedWarp != uWarp)
    {
        std::fill_n(outZoomRotWarp, totalVertices, ZoomRotWarp{uZoom, uZoomExp, uRot, uWarp});
        cache.cachedZoom = uZoom;
        cache.cachedZoomExp = uZoomExp;
        cache.cachedRot = uRot;
        cache.cachedWarp = uWarp;
        cache.hasCachedZoomRotWarp = true;
        dirty.zoomRotWarp = true;
    }

    if (mutCenter)
    {
        cache.hasCachedCenter = false;
        dirty.center = true;
    }
    else if (!cache.hasCachedCenter || cache.cachedCx != uCx || cache.cachedCy != uCy)
    {
        std::fill_n(outCenter, totalVertices, Renderer::Point{uCx, uCy});
        cache.cachedCx = uCx;
        cache.cachedCy = uCy;
        cache.hasCachedCenter = true;
        dirty.center = true;
    }

    if (mutDist)
    {
        cache.hasCachedDistance = false;
        dirty.distance = true;
    }
    else if (!cache.hasCachedDistance || cache.cachedDx != uDx || cache.cachedDy != uDy)
    {
        std::fill_n(outDistance, totalVertices, Renderer::Point{uDx, uDy});
        cache.cachedDx = uDx;
        cache.cachedDy = uDy;
        cache.hasCachedDistance = true;
        dirty.distance = true;
    }

    if (mutStretch)
    {
        cache.hasCachedStretch = false;
        dirty.stretch = true;
    }
    else if (!cache.hasCachedStretch || cache.cachedSx != uSx || cache.cachedSy != uSy)
    {
        std::fill_n(outStretch, totalVertices, Renderer::Point{uSx, uSy});
        cache.cachedSx = uSx;
        cache.cachedSy = uSy;
        cache.hasCachedStretch = true;
        dirty.stretch = true;
    }

    // Seed all 10 per-vertex variables once per frame so read-only references see current per-frame values.
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

    for (int vertex = 0; vertex < totalVertices; ++vertex)
    {
        if (u.readsX)
        {
            *perPixelContext.x = static_cast<PRJM_EVAL_F>(vertices[vertex].X() * 0.5f + 0.5f);
        }
        if (u.readsY)
        {
            *perPixelContext.y = static_cast<PRJM_EVAL_F>(vertices[vertex].Y() * 0.5f + 0.5f);
        }
        if (u.readsRad)
        {
            *perPixelContext.rad = static_cast<PRJM_EVAL_F>(radiusAngles[vertex].radius);
        }
        if (u.readsAng)
        {
            *perPixelContext.ang = static_cast<PRJM_EVAL_F>(-radiusAngles[vertex].angle);
        }

        if (u.writesZoom) *perPixelContext.zoom = baseZoom;
        if (u.writesZoomExp) *perPixelContext.zoomexp = baseZoomExp;
        if (u.writesRot) *perPixelContext.rot = baseRot;
        if (u.writesWarp) *perPixelContext.warp = baseWarp;
        if (u.writesCx) *perPixelContext.cx = baseCx;
        if (u.writesCy) *perPixelContext.cy = baseCy;
        if (u.writesDx) *perPixelContext.dx = baseDx;
        if (u.writesDy) *perPixelContext.dy = baseDy;
        if (u.writesSx) *perPixelContext.sx = baseSx;
        if (u.writesSy) *perPixelContext.sy = baseSy;

        perPixelContext.ExecutePerPixelCode();

        if (mutZRW)
        {
            float vZoom = uZoom;
            if (u.writesZoom)
            {
                vZoom = static_cast<float>(*perPixelContext.zoom);
                if (applyTimeScale && vZoom > 0.001f)
                {
                    vZoom = std::pow(vZoom, kappa);
                }
            }
            const float vZoomExp = u.writesZoomExp ? static_cast<float>(*perPixelContext.zoomexp) : uZoomExp;
            float vRot = uRot;
            if (u.writesRot)
            {
                vRot = static_cast<float>(*perPixelContext.rot);
                if (applyTimeScale)
                {
                    vRot *= kappa;
                }
            }
            float vWarp = uWarp;
            if (u.writesWarp)
            {
                vWarp = static_cast<float>(*perPixelContext.warp);
                if (applyTimeScale)
                {
                    vWarp *= kappa;
                }
            }
            outZoomRotWarp[vertex] = {vZoom, vZoomExp, vRot, vWarp};
        }

        if (mutCenter)
        {
            outCenter[vertex] = {
                static_cast<float>(*perPixelContext.cx),
                static_cast<float>(*perPixelContext.cy)
            };
        }

        if (mutDist)
        {
            float vDx = uDx;
            if (u.writesDx)
            {
                vDx = static_cast<float>(*perPixelContext.dx);
                if (applyTimeScale)
                {
                    vDx *= kappa;
                }
            }
            float vDy = uDy;
            if (u.writesDy)
            {
                vDy = static_cast<float>(*perPixelContext.dy);
                if (applyTimeScale)
                {
                    vDy *= kappa;
                }
            }
            outDistance[vertex] = {vDx, vDy};
        }

        if (mutStretch)
        {
            float vSx = uSx;
            if (u.writesSx)
            {
                vSx = static_cast<float>(*perPixelContext.sx);
                if (applyTimeScale && vSx > 0.001f)
                {
                    vSx = std::pow(vSx, kappa);
                }
            }
            float vSy = uSy;
            if (u.writesSy)
            {
                vSy = static_cast<float>(*perPixelContext.sy);
                if (applyTimeScale && vSy > 0.001f)
                {
                    vSy = std::pow(vSy, kappa);
                }
            }
            outStretch[vertex] = {vSx, vSy};
        }
    }

    // Phase unwrapping & mesh seam smoothing across the polar branch cut (y = 0, x < 0).
    const int midY = gridSizeY / 2;
    const int midX = gridSizeX / 2;
    if (u.writesRot && midY > 0)
    {
        for (int gx = 0; gx < midX; ++gx)
        {
            const int upperIdx = gx + midY * (gridSizeX + 1);
            const int lowerIdx = gx + (midY - 1) * (gridSizeX + 1);

            const float diffRot = outZoomRotWarp[upperIdx].rot - outZoomRotWarp[lowerIdx].rot;
            if (diffRot > 3.14159265f)
            {
                outZoomRotWarp[upperIdx].rot -= 6.2831853f;
            }
            else if (diffRot < -3.14159265f)
            {
                outZoomRotWarp[upperIdx].rot += 6.2831853f;
            }
        }
    }

    return dirty;
}

void PerPixelMesh::CalculateMesh(const PresetState& presetState, const PerFrameContext& perFrameContext, PerPixelContext& perPixelContext)
{
    DirtyBuffers const dirty = EvaluateGridCPU(
        m_gridSizeX,
        m_gridSizeY,
        presetState.renderContext.fps,
        m_warpMesh.Vertices().Get().data(),
        m_radiusAngleBuffer.Get().data(),
        perFrameContext,
        perPixelContext,
        m_zoomRotWarpBuffer.Get().data(),
        m_centerBuffer.Get().data(),
        m_distanceBuffer.Get().data(),
        m_stretchBuffer.Get().data(),
        m_gridCache);

    if (dirty.zoomRotWarp)
    {
        m_zoomRotWarpBuffer.Update();
    }
    if (dirty.center)
    {
        m_centerBuffer.Update();
    }
    if (dirty.distance)
    {
        m_distanceBuffer.Update();
    }
    if (dirty.stretch)
    {
        m_stretchBuffer.Update();
    }
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

    Renderer::BlendMode::SetBlendActive(false);
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
