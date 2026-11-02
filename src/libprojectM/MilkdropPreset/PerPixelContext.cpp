#include "PerPixelContext.hpp"

#include "MilkdropPresetExceptions.hpp"
#include "PerFrameContext.hpp"

#include <Logging.hpp>
#include <cmath>

#define REG_VAR(var) \
    var = projectm_eval_context_register_variable(perPixelCodeContext, #var);

namespace libprojectM {
namespace MilkdropPreset {

PerPixelContext::PerPixelContext(projectm_eval_mem_buffer gmegabuf, PRJM_EVAL_F (*globalRegisters)[100])
    : perPixelCodeContext(projectm_eval_context_create(gmegabuf, globalRegisters))
{
}

PerPixelContext::~PerPixelContext()
{
    if (perPixelCodeHandle != nullptr)
    {
        projectm_eval_code_destroy(perPixelCodeHandle);
    }

    if (perPixelCodeContext != nullptr)
    {
        projectm_eval_context_destroy(perPixelCodeContext);
    }
}

void PerPixelContext::RegisterBuiltinVariables()
{
    projectm_eval_context_reset_variables(perPixelCodeContext);

    REG_VAR(zoom);
    REG_VAR(zoomexp);
    REG_VAR(rot);
    REG_VAR(warp);
    REG_VAR(cx);
    REG_VAR(cy);
    REG_VAR(dx);
    REG_VAR(dy);
    REG_VAR(sx);
    REG_VAR(sy);
    REG_VAR(time);
    REG_VAR(fps);
    REG_VAR(bass);
    REG_VAR(mid);
    REG_VAR(treb);
    REG_VAR(bass_att);
    REG_VAR(mid_att);
    REG_VAR(treb_att);
    REG_VAR(frame);
    REG_VAR(x);
    REG_VAR(y);
    REG_VAR(rad);
    REG_VAR(ang);
    for (int q = 0; q < QVarCount; q++)
    {
        std::string qvar = "q" + std::to_string(q + 1);
        q_vars[q] = projectm_eval_context_register_variable(perPixelCodeContext, qvar.c_str());
    }
    REG_VAR(progress);
    REG_VAR(meshx);
    REG_VAR(meshy);
    REG_VAR(pixelsx);
    REG_VAR(pixelsy);
    REG_VAR(aspectx);
    REG_VAR(aspecty);
}

void PerPixelContext::LoadStateReadOnlyVariables(PresetState& state, PerFrameContext& perFrameState)
{
    *time = static_cast<PRJM_EVAL_F>(*perFrameState.time);
    *fps = static_cast<PRJM_EVAL_F>(*perFrameState.fps);
    *frame = static_cast<PRJM_EVAL_F>(*perFrameState.frame);
    *progress = static_cast<PRJM_EVAL_F>(*perFrameState.progress);
    *bass = static_cast<PRJM_EVAL_F>(*perFrameState.bass);
    *mid = static_cast<PRJM_EVAL_F>(*perFrameState.mid);
    *treb = static_cast<PRJM_EVAL_F>(*perFrameState.treb);
    *bass_att = static_cast<PRJM_EVAL_F>(*perFrameState.bass_att);
    *mid_att = static_cast<PRJM_EVAL_F>(*perFrameState.mid_att);
    *treb_att = static_cast<PRJM_EVAL_F>(*perFrameState.treb_att);
    *meshx = static_cast<PRJM_EVAL_F>(state.renderContext.perPixelMeshX);
    *meshy = static_cast<PRJM_EVAL_F>(state.renderContext.perPixelMeshY);
    *pixelsx = static_cast<PRJM_EVAL_F>(state.renderContext.viewportSizeX);
    *pixelsy = static_cast<PRJM_EVAL_F>(state.renderContext.viewportSizeY);
    *aspectx = static_cast<PRJM_EVAL_F>(state.renderContext.invAspectX);
    *aspecty = static_cast<PRJM_EVAL_F>(state.renderContext.invAspectY);
}

void PerPixelContext::LoadPerFrameQVariables(PresetState& state, PerFrameContext& perFrameState)
{
    for (int q = 0; q < QVarCount; q++)
    {
        PRJM_EVAL_F qVal = *perFrameState.q_vars[q];
        if (!std::isfinite(qVal))
        {
            qVal = static_cast<PRJM_EVAL_F>(0.0);
            *perFrameState.q_vars[q] = qVal;
        }
        state.frameQVariables[q] = qVal;
        *q_vars[q] = qVal;
    }
}

void PerPixelContext::CompilePerPixelCode(const std::string& perPixelCode)
{
    if (perPixelCodeHandle != nullptr)
    {
        projectm_eval_code_destroy(perPixelCodeHandle);
        perPixelCodeHandle = nullptr;
    }
    m_usage = {};

    if (perPixelCode.empty())
    {
        return;
    }

    perPixelCodeHandle = projectm_eval_code_compile(perPixelCodeContext, perPixelCode.c_str());
    if (perPixelCodeHandle == nullptr)
    {
        std::string error;
        int line;
        int col;
        auto* errmsg = projectm_eval_get_error(perPixelCodeContext, &line, &col);
        if (errmsg)
        {
            error = "[PerPixelContext] Could not compile per-pixel code: ";
            error += errmsg;
            error += "(L" + std::to_string(line) + " C" + std::to_string(col) + ")";
        }
        else
        {
            error = "[PerPixelContext] Could not compile per-pixel code.";
        }
        LOG_DEBUG("[PerPixelContext] Failed per-pixel code:\n" + perPixelCode);
        throw MilkdropCompileException(error);
    }

    auto queryReadOrWrite = [this](const PRJM_EVAL_F* varPtr) -> bool {
        int r = 0;
        int w = 0;
        projectm_eval_code_query_variable_usage(perPixelCodeHandle, varPtr, &r, &w);
        return (r != 0) || (w != 0);
    };

    auto queryWritten = [this](const PRJM_EVAL_F* varPtr) -> bool {
        int w = 0;
        projectm_eval_code_query_variable_usage(perPixelCodeHandle, varPtr, nullptr, &w);
        return w != 0;
    };

    m_usage.readsX = queryReadOrWrite(x);
    m_usage.readsY = queryReadOrWrite(y);
    m_usage.readsRad = queryReadOrWrite(rad);
    m_usage.readsAng = queryReadOrWrite(ang);

    m_usage.writesZoom = queryWritten(zoom);
    m_usage.writesZoomExp = queryWritten(zoomexp);
    m_usage.writesRot = queryWritten(rot);
    m_usage.writesWarp = queryWritten(warp);
    m_usage.writesCx = queryWritten(cx);
    m_usage.writesCy = queryWritten(cy);
    m_usage.writesDx = queryWritten(dx);
    m_usage.writesDy = queryWritten(dy);
    m_usage.writesSx = queryWritten(sx);
    m_usage.writesSy = queryWritten(sy);

    const PRJM_EVAL_F* allowedPerVertexVars[10] = {
        zoom, zoomexp, rot, warp, cx, cy, dx, dy, sx, sy
    };
    const bool onlyAllowedWrites =
        projectm_eval_code_has_only_allowed_writes(perPixelCodeHandle, allowedPerVertexVars, 10) != 0;

    m_usage.isSpatiallyInvariant =
        onlyAllowedWrites &&
        !m_usage.readsX &&
        !m_usage.readsY &&
        !m_usage.readsRad &&
        !m_usage.readsAng;
}

void PerPixelContext::ExecutePerPixelCode()
{
    if (perPixelCodeHandle != nullptr)
    {
        projectm_eval_code_execute(perPixelCodeHandle);
    }
}

} // namespace MilkdropPreset
} // namespace libprojectM
