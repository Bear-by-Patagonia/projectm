#include "Audio/Loudness.hpp"

#include <algorithm>
#include <cmath>

namespace libprojectM {
namespace Audio {

Loudness::Loudness(Loudness::Band band)
    : m_band(band)
{
}

void Loudness::Update(const std::array<float, SpectrumSamples>& spectrumSamples, double secondsSinceLastFrame, uint32_t frame)
{
    SumBand(spectrumSamples);
    UpdateBandAverage(secondsSinceLastFrame, frame);
}

auto Loudness::CurrentRelative() const -> float
{
    return m_currentRelative;
}

auto Loudness::AverageRelative() const -> float
{
    return m_averageRelative;
}

void Loudness::SumBand(const std::array<float, SpectrumSamples>& spectrumSamples)
{
    // Canonical MilkDrop FFT Band Partitioning:
    // Only the first half of the 512 spectrum bins is used, split equally into 3 bands:
    // - Bass: bins 0..85 (~0 to 1,800 Hz)
    // - Middles: bins 85..170 (~1,800 to 3,600 Hz)
    // - Treble: bins 170..256 (~3,600 to 5,500 Hz)
    int const start = SpectrumSamples * static_cast<int>(m_band) / 6;
    int const end = SpectrumSamples * (static_cast<int>(m_band) + 1) / 6;

    m_current = 0.0f;
    for (int sample = start; sample < end && sample < static_cast<int>(SpectrumSamples); ++sample)
    {
        m_current += spectrumSamples[sample];
    }
}

void Loudness::UpdateBandAverage(double secondsSinceLastFrame, uint32_t frame)
{
    double const safeDt = std::clamp(secondsSinceLastFrame, 0.001, 0.1);

    // Standard MilkDrop ballistic envelope: 0.2 attack, 0.5 decay
    float rate = AdjustRateToFps(m_current > m_average ? 0.2f : 0.5f, safeDt);
    m_average = m_average * rate + m_current * (1.0f - rate);

    rate = AdjustRateToFps(frame < 50 ? 0.9f : 0.992f, safeDt);
    m_longAverage = m_longAverage * rate + m_current * (1.0f - rate);

    float const rawRel = std::fabs(m_longAverage) < 0.001f ? 1.0f : (m_current / m_longAverage);
    float const rawAvgRel = std::fabs(m_longAverage) < 0.001f ? 1.0f : (m_average / m_longAverage);

    // Baseline nominal loudness revolves around 1.0 (quiet <0.7, loud >1.3).
    // Clamped strictly to 2.0 to prevent audio-driven color summation from clipping to pure white.
    m_currentRelative = std::clamp(rawRel, 0.0f, 2.0f);
    m_averageRelative = std::clamp(rawAvgRel, 0.0f, 2.0f);
}

auto Loudness::AdjustRateToFps(float rate, double secondsSinceLastFrame) -> float
{
    float const perSecondDecayRateAtFps1 = std::pow(rate, 30.0f);
    float const perFrameDecayRateAtFps2 = std::pow(perSecondDecayRateAtFps1, static_cast<float>(secondsSinceLastFrame));

    return perFrameDecayRateAtFps2;
}

} // namespace Audio
} // namespace libprojectM
