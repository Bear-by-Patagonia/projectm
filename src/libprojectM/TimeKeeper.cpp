#include "TimeKeeper.hpp"

#include <algorithm>
#include <random>

namespace libprojectM {

TimeKeeper::TimeKeeper(double presetDuration, double smoothDuration, double hardcutDuration, double easterEgg)
    : m_easterEgg(easterEgg)
    , m_presetDuration(presetDuration)
    , m_softCutDuration(smoothDuration)
    , m_hardCutDuration(hardcutDuration)
{
    UpdateTimers();
}

void TimeKeeper::SetFrameTime(double secondsSinceStart)
{
    m_userSpecifiedTime = secondsSinceStart;
}

double TimeKeeper::GetFrameTime() const
{
    return m_currentTime;
}

void TimeKeeper::UpdateTimers()
{
    double currentFrameTime{m_userSpecifiedTime};

    if (m_userSpecifiedTime < 0.0)
    {
        auto currentTime = std::chrono::high_resolution_clock::now();
        currentFrameTime = std::chrono::duration<double>(currentTime - m_startTime).count();
    }

    double rawDt = currentFrameTime - m_currentTime;
    if (!m_hasInitialDt)
    {
        m_hasInitialDt = true;
        rawDt = (rawDt > 0.0005 && rawDt < 0.1) ? rawDt : (1.0 / 60.0);
        m_dtHistory.fill(rawDt);
        m_secondsSinceLastFrame = rawDt;
    }
    else
    {
        // Clamp extreme outliers (e.g. app pause, resume or GC pause) to avoid time jumps
        double clampedDt = std::max(0.0005, std::min(rawDt, 0.1));
        // Shift history window
        m_dtHistory[0] = m_dtHistory[1];
        m_dtHistory[1] = m_dtHistory[2];
        m_dtHistory[2] = m_dtHistory[3];
        m_dtHistory[3] = clampedDt;

        // 4th-Order Runge-Kutta / Simpson 1/3 quadrature integration:
        // dt_rk4 = (dt0 + 2*dt1 + 2*dt2 + dt3) / 6
        // Filters out temporal jitter on mobile/Android TV while maintaining strict phase accuracy
        m_secondsSinceLastFrame = (m_dtHistory[0] + 2.0 * m_dtHistory[1] + 2.0 * m_dtHistory[2] + m_dtHistory[3]) / 6.0;
    }

    m_currentTime = currentFrameTime;
    m_presetFrameA++;
    m_presetFrameB++;
}

void TimeKeeper::StartPreset()
{
    m_isSmoothing = false;
    m_presetTimeA = m_currentTime;
    m_presetFrameA = 1;
    m_presetDurationA = sampledPresetDuration();
}

void TimeKeeper::StartSmoothing()
{
    m_isSmoothing = true;
    m_presetTimeB = m_currentTime;
    m_presetFrameB = 1;
    m_presetDurationB = sampledPresetDuration();
}

void TimeKeeper::EndSmoothing()
{
    m_isSmoothing = false;
    m_presetTimeA = m_presetTimeB;
    m_presetFrameA = m_presetFrameB;
    m_presetDurationA = m_presetDurationB;
}

bool TimeKeeper::CanHardCut()
{
    return (m_currentTime - m_presetTimeA) > m_hardCutDuration;
}

double TimeKeeper::SmoothRatio()
{
    return (m_currentTime - m_presetTimeB) / m_softCutDuration;
}

bool TimeKeeper::IsSmoothing()
{
    return m_isSmoothing;
}

double TimeKeeper::GetRunningTime()
{
    return m_currentTime;
}

double TimeKeeper::PresetProgressA()
{
    if (m_isSmoothing)
    {
        return 1.0;
    }

    return std::min((m_currentTime - m_presetTimeA) / m_presetDurationA, 1.0);
}

double TimeKeeper::PresetProgressB()
{
    return std::min((m_currentTime - m_presetTimeB) / m_presetDurationB, 1.0);
}

int TimeKeeper::PresetFrameB()
{
    return m_presetFrameB;
}

int TimeKeeper::PresetFrameA()
{
    return m_presetFrameA;
}

double TimeKeeper::PresetTimeB()
{
    return m_presetTimeB;
}

double TimeKeeper::PresetTimeA()
{
    return m_presetTimeA;
}

double TimeKeeper::sampledPresetDuration()
{
    if (m_easterEgg < 0.001)
    {
        return m_presetDuration;
    }

    std::normal_distribution<double> gaussianDistribution(m_presetDuration, m_easterEgg);
    return std::max<double>(1.0, gaussianDistribution(m_randomGenerator));
}

} // namespace libprojectM
