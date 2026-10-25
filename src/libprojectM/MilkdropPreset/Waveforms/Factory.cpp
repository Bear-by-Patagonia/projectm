#include "Waveforms/Factory.hpp"

#include "Waveforms/CenteredSpiro.hpp"
#include "Waveforms/Circle.hpp"
#include "Waveforms/DerivativeLine.hpp"
#include "Waveforms/DoubleLine.hpp"
#include "Waveforms/ExplosiveHash.hpp"
#include "Waveforms/Line.hpp"
#include "Waveforms/ModernWave11.hpp"
#include "Waveforms/ModernWave9.hpp"
#include "Waveforms/ModernWaveFlower.hpp"
#include "Waveforms/ModernWaveLasso.hpp"
#include "Waveforms/ModernWaveSkewed.hpp"
#include "Waveforms/ModernWaveStar.hpp"
#include "Waveforms/ModernWaveX.hpp"
#include "Waveforms/SpectrumLine.hpp"
#include "Waveforms/XYOscillationSpiral.hpp"

namespace libprojectM {
namespace MilkdropPreset {
namespace Waveforms {

auto Factory::Create(WaveformMode mode) -> std::unique_ptr<WaveformMath>
{
    switch (mode)
    {
        case WaveformMode::Circle:
            return std::make_unique<Circle>();

        case WaveformMode::XYOscillationSpiral:
            return std::make_unique<XYOscillationSpiral>();

        case WaveformMode::CenteredSpiro:
        case WaveformMode::CenteredSpiroVolume:
            // Both "centered spiro" waveforms are identical. Only difference is the alpha value.
            return std::make_unique<CenteredSpiro>();

        case WaveformMode::DerivativeLine:
            return std::make_unique<DerivativeLine>();

        case WaveformMode::ExplosiveHash:
            return std::make_unique<ExplosiveHash>();

        case WaveformMode::Line:
            return std::make_unique<Line>();

        case WaveformMode::DoubleLine:
            return std::make_unique<DoubleLine>();

        case WaveformMode::SpectrumLine:
            return std::make_unique<SpectrumLine>();

        case WaveformMode::ModernWave9:
            return std::make_unique<ModernWave9>();

        case WaveformMode::ModernWaveX:
            return std::make_unique<ModernWaveX>();

        case WaveformMode::ModernWave11:
            return std::make_unique<ModernWave11>();

        case WaveformMode::ModernWaveSkewed:
            return std::make_unique<ModernWaveSkewed>();

        case WaveformMode::ModernWaveStar:
            return std::make_unique<ModernWaveStar>();

        case WaveformMode::ModernWaveFlower:
            return std::make_unique<ModernWaveFlower>();

        case WaveformMode::ModernWaveLasso:
            return std::make_unique<ModernWaveLasso>();

        case WaveformMode::Count:
            break;
    }

    return {};
}

} // namespace Waveforms
} // namespace MilkdropPreset
} // namespace libprojectM
