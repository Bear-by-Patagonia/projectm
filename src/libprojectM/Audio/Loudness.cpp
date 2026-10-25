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
    // Psychoacoustic Logarithmic / Bark-Scale Band Partitioning (assuming ~43 Hz per FFT bin):
    // - Bass (bins 1..14, ~43 Hz - 600 Hz): isolates kick drums (50-130 Hz) & basslines without vocal bleed.
    // - Middles (bins 15..75, ~640 Hz - 3,200 Hz): fundamentals, snares, guitars, and human vocals.
    // - Treble (bins 76..255, ~3,250 Hz - 11,000 Hz): hi-hats, cymbals, and air transients with perceptual tilt.
    float rawEnergy = 0.0f;

    switch (m_band)
    {
        case Band::Bass:
        {
            for (int k = 1; k <= 5 && k < static_cast<int>(SpectrumSamples); ++k)
            {
                rawEnergy += spectrumSamples[k] * 1.45f; // Sub-bass & punchy kick resonance (43-215 Hz)
            }
            for (int k = 6; k <= 14 && k < static_cast<int>(SpectrumSamples); ++k)
            {
                rawEnergy += spectrumSamples[k] * 0.85f; // Upper bass harmonic body (250-600 Hz)
            }
            break;
        }
        case Band::Middles:
        {
            for (int k = 15; k <= 75 && k < static_cast<int>(SpectrumSamples); ++k)
            {
                rawEnergy += spectrumSamples[k];
            }
            break;
        }
        case Band::Treble:
        default:
        {
            const int endBin = std::min(255, static_cast<int>(SpectrumSamples) - 1);
            for (int k = 76; k <= endBin; ++k)
            {
                float const tilt = 1.0f + 0.75f * static_cast<float>(k - 76) / 180.0f;
                rawEnergy += spectrumSamples[k] * tilt;
            }
            break;
        }
    }

    // Half-Wave Rectified Spectral Flux (Transient Onset Detector):
    // Boosts immediate percussive attacks without muddying steady-state sustain.
    float const spectralFlux = std::max(0.0f, rawEnergy - m_prevRawEnergy);
    m_prevRawEnergy = rawEnergy;
    m_current = rawEnergy + 0.32f * spectralFlux;
}

void Loudness::UpdateBandAverage(double secondsSinceLastFrame, uint32_t frame)
{
    double const safeDt = std::clamp(secondsSinceLastFrame, 0.001, 0.1);

    // Asymmetric ballistic envelope: ultra-fast attack (0.14) on transients, musical release (0.56)
    float rate = AdjustRateToFps(m_current > m_average ? 0.14f : 0.56f, safeDt);
    m_average = m_average * rate + m_current * (1.0f - rate);

    rate = AdjustRateToFps(frame < 50 ? 0.9f : 0.992f, safeDt);
    m_longAverage = m_longAverage * rate + m_current * (1.0f - rate);

    float const rawRel = std::fabs(m_longAverage) < 0.001f ? 1.0f : (m_current / m_longAverage);
    float const rawAvgRel = std::fabs(m_longAverage) < 0.001f ? 1.0f : (m_average / m_longAverage);

    m_currentRelative = std::clamp(rawRel, 0.0f, 3.5f);
    m_averageRelative = std::clamp(rawAvgRel, 0.0f, 3.5f);
}

auto Loudness::AdjustRateToFps(float rate, double secondsSinceLastFrame) -> float
{
    float const perSecondDecayRateAtFps1 = std::pow(rate, 30.0f);
    float const perFrameDecayRateAtFps2 = std::pow(perSecondDecayRateAtFps1, static_cast<float>(secondsSinceLastFrame));

    return perFrameDecayRateAtFps2;
}

} // namespace Audio
} // namespace libprojectM
