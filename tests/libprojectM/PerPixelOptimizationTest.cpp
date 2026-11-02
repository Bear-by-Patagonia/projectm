#include <MilkdropPreset/PerFrameContext.hpp>
#include <MilkdropPreset/PerPixelContext.hpp>
#include <MilkdropPreset/PerPixelMesh.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

using libprojectM::MilkdropPreset::PerFrameContext;
using libprojectM::MilkdropPreset::PerPixelContext;
using libprojectM::MilkdropPreset::PerPixelMesh;
using libprojectM::Renderer::Point;

void InitDefaultContexts(PerFrameContext& pfCtx, PerPixelContext& ppCtx)
{
    pfCtx.RegisterBuiltinVariables();
    *pfCtx.zoom = 1.0;
    *pfCtx.zoomexp = 1.0;
    *pfCtx.rot = 0.0;
    *pfCtx.warp = 0.0;
    *pfCtx.cx = 0.5;
    *pfCtx.cy = 0.5;
    *pfCtx.dx = 0.0;
    *pfCtx.dy = 0.0;
    *pfCtx.sx = 1.0;
    *pfCtx.sy = 1.0;
    *pfCtx.fps = 60.0;

    ppCtx.RegisterBuiltinVariables();
    *ppCtx.fps = 60.0;
    *ppCtx.meshx = 64.0;
    *ppCtx.meshy = 48.0;
    *ppCtx.pixelsx = 1920.0;
    *ppCtx.pixelsy = 1080.0;
    *ppCtx.aspectx = 1.0;
    *ppCtx.aspecty = 1.0;
}

struct TestGridGeometry
{
    int gridX{64};
    int gridY{48};
    std::vector<Point> vertices;
    std::vector<PerPixelMesh::RadiusAngle> radiusAngles;

    TestGridGeometry()
    {
        const int total = (gridX + 1) * (gridY + 1);
        vertices.resize(total);
        radiusAngles.resize(total);
        int idx = 0;
        for (int gy = 0; gy <= gridY; ++gy)
        {
            for (int gx = 0; gx <= gridX; ++gx)
            {
                const float x = static_cast<float>(gx) / static_cast<float>(gridX) * 2.0f - 1.0f;
                const float y = static_cast<float>(gy) / static_cast<float>(gridY) * 2.0f - 1.0f;
                vertices[idx] = {x, y};
                const float r = std::hypot(x, y);
                const float ang = (gx == gridX / 2 && gy == gridY / 2) ? 0.0f : std::atan2(y, x);
                radiusAngles[idx] = {r, ang};
                ++idx;
            }
        }
    }
};

} // namespace

TEST(PerPixelOptimization, SpatiallyInvariantEquationEvaluatesOncePerFrameAndMatchesExpectedValues)
{
    TestGridGeometry geom;
    const int totalVertices = (geom.gridX + 1) * (geom.gridY + 1);

    auto gmegabuf = projectm_eval_memory_buffer_create();
    PRJM_EVAL_F globalRegs[100]{};

    PerFrameContext pfCtx(gmegabuf, &globalRegs);
    PerPixelContext ppCtx(gmegabuf, &globalRegs);
    InitDefaultContexts(pfCtx, ppCtx);

    *pfCtx.bass_att = 1.5;
    *ppCtx.bass_att = 1.5;
    *pfCtx.q_vars[0] = 0.8;
    *ppCtx.q_vars[0] = 0.8;
    ppCtx.CompilePerPixelCode("zoom = 1.02 + 0.04 * bass_att; rot = -0.05 * q1;");

    EXPECT_TRUE(ppCtx.IsSpatiallyInvariant());

    std::vector<PerPixelMesh::ZoomRotWarp> zrw(totalVertices);
    std::vector<Point> center(totalVertices);
    std::vector<Point> dist(totalVertices);
    std::vector<Point> stretch(totalVertices);
    PerPixelMesh::GridCacheState cache{};

    const auto dirty1 = PerPixelMesh::EvaluateGridCPU(
        geom.gridX, geom.gridY, 60.0f,
        geom.vertices.data(), geom.radiusAngles.data(),
        pfCtx, ppCtx,
        zrw.data(), center.data(), dist.data(), stretch.data(),
        cache);

    // Must evaluate EEL2 ONLY ONCE instead of 3,185 times!
    EXPECT_EQ(cache.lastEvalCount, 1u);
    EXPECT_TRUE(dirty1.zoomRotWarp);
    EXPECT_TRUE(dirty1.center);
    EXPECT_TRUE(dirty1.distance);
    EXPECT_TRUE(dirty1.stretch);

    for (int i = 0; i < totalVertices; ++i)
    {
        EXPECT_NEAR(zrw[i].zoom, 1.08f, 1e-5f);
        EXPECT_NEAR(zrw[i].rot, -0.04f, 1e-5f);
        EXPECT_NEAR(center[i].X(), 0.5f, 1e-5f);
        EXPECT_NEAR(center[i].Y(), 0.5f, 1e-5f);
    }

    // Second frame with identical audio/q1: evaluates once, hits VBO cache -> ZERO dirty buffers!
    const auto dirty2 = PerPixelMesh::EvaluateGridCPU(
        geom.gridX, geom.gridY, 60.0f,
        geom.vertices.data(), geom.radiusAngles.data(),
        pfCtx, ppCtx,
        zrw.data(), center.data(), dist.data(), stretch.data(),
        cache);

    EXPECT_EQ(cache.lastEvalCount, 1u);
    EXPECT_FALSE(dirty2.zoomRotWarp);
    EXPECT_FALSE(dirty2.center);
    EXPECT_FALSE(dirty2.distance);
    EXPECT_FALSE(dirty2.stretch);

    projectm_eval_memory_buffer_destroy(gmegabuf);
}

TEST(PerPixelOptimization, StatefulCounterAndRandAreNotSpatiallyInvariant)
{
    TestGridGeometry geom;
    const int totalVertices = (geom.gridX + 1) * (geom.gridY + 1);

    auto gmegabuf = projectm_eval_memory_buffer_create();
    PRJM_EVAL_F globalRegs[100]{};

    PerFrameContext pfCtx(gmegabuf, &globalRegs);
    PerPixelContext ppCtx(gmegabuf, &globalRegs);
    InitDefaultContexts(pfCtx, ppCtx);
    ppCtx.CompilePerPixelCode("idx = idx + 1; zoom = 1.0 + 0.0001 * idx;");

    EXPECT_FALSE(ppCtx.IsSpatiallyInvariant());

    std::vector<PerPixelMesh::ZoomRotWarp> zrw(totalVertices);
    std::vector<Point> center(totalVertices);
    std::vector<Point> dist(totalVertices);
    std::vector<Point> stretch(totalVertices);
    PerPixelMesh::GridCacheState cache{};

    PerPixelMesh::EvaluateGridCPU(
        geom.gridX, geom.gridY, 60.0f,
        geom.vertices.data(), geom.radiusAngles.data(),
        pfCtx, ppCtx,
        zrw.data(), center.data(), dist.data(), stretch.data(),
        cache);

    EXPECT_EQ(cache.lastEvalCount, static_cast<size_t>(totalVertices));
    EXPECT_NEAR(zrw[0].zoom, 1.0001f, 1e-6f);
    EXPECT_NEAR(zrw[totalVertices - 1].zoom, 1.0f + 0.0001f * static_cast<float>(totalVertices), 1e-5f);

    projectm_eval_memory_buffer_destroy(gmegabuf);
}

TEST(PerPixelOptimization, TornadoPerAttributeDirtyTrackingSkipsUnmodifiedBuffers)
{
    TestGridGeometry geom;
    const int totalVertices = (geom.gridX + 1) * (geom.gridY + 1);

    auto gmegabuf = projectm_eval_memory_buffer_create();
    PRJM_EVAL_F globalRegs[100]{};

    PerFrameContext pfCtx(gmegabuf, &globalRegs);
    PerPixelContext ppCtx(gmegabuf, &globalRegs);
    InitDefaultContexts(pfCtx, ppCtx);
    *pfCtx.bass_att = 1.65;
    *ppCtx.bass_att = 1.65;
    ppCtx.CompilePerPixelCode("rot = rot + (rad - 0.4) * 1.7 * max(0, min((bass_att - 1.1) * 1.5, 5));");

    EXPECT_FALSE(ppCtx.IsSpatiallyInvariant());
    EXPECT_TRUE(ppCtx.Usage().readsRad);
    EXPECT_FALSE(ppCtx.Usage().readsX);
    EXPECT_TRUE(ppCtx.Usage().writesRot);
    EXPECT_FALSE(ppCtx.Usage().writesCx);
    EXPECT_FALSE(ppCtx.Usage().writesDx);
    EXPECT_FALSE(ppCtx.Usage().writesSx);

    std::vector<PerPixelMesh::ZoomRotWarp> zrw(totalVertices);
    std::vector<Point> center(totalVertices);
    std::vector<Point> dist(totalVertices);
    std::vector<Point> stretch(totalVertices);
    PerPixelMesh::GridCacheState cache{};

    // Frame 1: cold cache initializes all 4 buffers
    const auto dirty1 = PerPixelMesh::EvaluateGridCPU(
        geom.gridX, geom.gridY, 60.0f,
        geom.vertices.data(), geom.radiusAngles.data(),
        pfCtx, ppCtx,
        zrw.data(), center.data(), dist.data(), stretch.data(),
        cache);
    EXPECT_TRUE(dirty1.zoomRotWarp);
    EXPECT_TRUE(dirty1.center);
    EXPECT_TRUE(dirty1.distance);
    EXPECT_TRUE(dirty1.stretch);

    // Frame 2: bass_att changes, cx/cy, dx/dy, sx/sy remain untouched -> ONLY zoomRotWarp is dirty!
    *pfCtx.bass_att = 1.85;
    *ppCtx.bass_att = 1.85;
    const auto dirty2 = PerPixelMesh::EvaluateGridCPU(
        geom.gridX, geom.gridY, 60.0f,
        geom.vertices.data(), geom.radiusAngles.data(),
        pfCtx, ppCtx,
        zrw.data(), center.data(), dist.data(), stretch.data(),
        cache);

    EXPECT_TRUE(dirty2.zoomRotWarp);
    EXPECT_FALSE(dirty2.center);
    EXPECT_FALSE(dirty2.distance);
    EXPECT_FALSE(dirty2.stretch);

    // Verify Tornado differential torsion physics: center (rad == 0) rotates negatively, corner (rad > 0.4) positively
    const int centerIdx = (geom.gridY / 2) * (geom.gridX + 1) + (geom.gridX / 2);
    EXPECT_LT(zrw[centerIdx].rot, 0.0f);
    EXPECT_GT(zrw[0].rot, 0.0f);

    projectm_eval_memory_buffer_destroy(gmegabuf);
}

TEST(PerPixelOptimization, MicrobenchmarkPerPixelGridCPU)
{
    TestGridGeometry geom;
    const int totalVertices = (geom.gridX + 1) * (geom.gridY + 1);
    constexpr int kFrames = 120;

    auto gmegabuf = projectm_eval_memory_buffer_create();
    PRJM_EVAL_F globalRegs[100]{};

    std::vector<PerPixelMesh::ZoomRotWarp> zrw(totalVertices);
    std::vector<Point> center(totalVertices);
    std::vector<Point> dist(totalVertices);
    std::vector<Point> stretch(totalVertices);

    // 1. Spatially-varying Tornado mesh evaluation (Bytecode VM + minimal resets + dirty VBOs)
    PerFrameContext pfVar(gmegabuf, &globalRegs);
    PerPixelContext ppVar(gmegabuf, &globalRegs);
    InitDefaultContexts(pfVar, ppVar);
    ppVar.CompilePerPixelCode("rot = rot + (rad - 0.4) * 1.7 * max(0, min((bass_att - 1.1) * 1.5, 5));");

    PerPixelMesh::GridCacheState cacheVar{};
    auto t0Var = std::chrono::steady_clock::now();
    for (int f = 0; f < kFrames; ++f)
    {
        *pfVar.bass_att = 1.2 + 0.005 * static_cast<double>(f);
        *ppVar.bass_att = *pfVar.bass_att;
        PerPixelMesh::EvaluateGridCPU(
            geom.gridX, geom.gridY, 60.0f,
            geom.vertices.data(), geom.radiusAngles.data(),
            pfVar, ppVar,
            zrw.data(), center.data(), dist.data(), stretch.data(),
            cacheVar);
    }
    auto t1Var = std::chrono::steady_clock::now();
    const double varUsPerFrame =
        std::chrono::duration<double, std::micro>(t1Var - t0Var).count() / static_cast<double>(kFrames);

    // 2. Spatially-invariant audio-reactive equation (1x/frame EEL2 + broadcast)
    PerFrameContext pfInv(gmegabuf, &globalRegs);
    PerPixelContext ppInv(gmegabuf, &globalRegs);
    InitDefaultContexts(pfInv, ppInv);
    ppInv.CompilePerPixelCode("zoom = 1.02 + 0.04 * bass_att; rot = -0.05 * q1;");

    PerPixelMesh::GridCacheState cacheInv{};
    auto t0Inv = std::chrono::steady_clock::now();
    for (int f = 0; f < kFrames; ++f)
    {
        *pfInv.bass_att = 1.2 + 0.005 * static_cast<double>(f);
        *ppInv.bass_att = *pfInv.bass_att;
        *pfInv.q_vars[0] = 0.75;
        *ppInv.q_vars[0] = 0.75;
        PerPixelMesh::EvaluateGridCPU(
            geom.gridX, geom.gridY, 60.0f,
            geom.vertices.data(), geom.radiusAngles.data(),
            pfInv, ppInv,
            zrw.data(), center.data(), dist.data(), stretch.data(),
            cacheInv);
    }
    auto t1Inv = std::chrono::steady_clock::now();
    const double invUsPerFrame =
        std::chrono::duration<double, std::micro>(t1Inv - t0Inv).count() / static_cast<double>(kFrames);

    std::printf("[BENCHMARK] Grid 64x48 (%d verts): SpatiallyVarying=%.2f us/frame | SpatiallyInvariant(1x)=%.2f us/frame (%.1fx faster)\n",
                totalVertices,
                varUsPerFrame,
                invUsPerFrame,
                (invUsPerFrame > 0.0) ? (varUsPerFrame / invUsPerFrame) : 0.0);

    EXPECT_LT(invUsPerFrame, varUsPerFrame);
    projectm_eval_memory_buffer_destroy(gmegabuf);
}
