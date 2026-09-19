#include "Audio/PCM.hpp"

#include <mutex>

namespace libprojectM {
namespace Audio {

PCM::PCM()
    : m_fftWaveformSamples(AudioBufferSamples, 0.0f)
    , m_fftSpectralData(SpectrumSamples, 0.0f)
{
    m_workerRunning.store(true, std::memory_order_release);
    m_workerThread = std::thread(&PCM::WorkerLoop, this);
}

PCM::~PCM()
{
    m_workerRunning.store(false, std::memory_order_release);
    m_workerCv.notify_all();
    if (m_workerThread.joinable())
    {
        m_workerThread.join();
    }
}

void PCM::WorkerLoop()
{
    while (m_workerRunning.load(std::memory_order_acquire))
    {
        {
            std::unique_lock<std::mutex> lock(m_workerMutex);
            m_workerCv.wait_for(lock, std::chrono::milliseconds(15), [this] {
                return !m_workerRunning.load(std::memory_order_acquire) ||
                       m_audioDirty.load(std::memory_order_acquire);
            });
            if (!m_workerRunning.load(std::memory_order_acquire))
            {
                break;
            }
            m_audioDirty.store(false, std::memory_order_release);
        }

        ProcessAudioData(m_pendingSecondsSinceLastFrame.load(std::memory_order_relaxed),
                         m_pendingFrame.load(std::memory_order_relaxed));
    }
}

template<
    int signalAmplitude,
    int signalOffset,
    typename SampleType>
void PCM::AddToBuffer(
    SampleType const* const samples,
    uint32_t channels,
    size_t const sampleCount)
{
    if (channels == 0 || sampleCount == 0)
    {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(m_pcmMutex);
        for (size_t i = 0; i < sampleCount; i++)
        {
            size_t const bufferOffset = (m_start + i) % AudioBufferSamples;
            m_inputBufferL[bufferOffset] = 128.0f * (static_cast<float>(samples[0 + i * channels]) - float(signalOffset)) / float(signalAmplitude);
            if (channels > 1)
            {
                m_inputBufferR[bufferOffset] = 128.0f * (static_cast<float>(samples[1 + i * channels]) - float(signalOffset)) / float(signalAmplitude);
            }
            else
            {
                m_inputBufferR[bufferOffset] = m_inputBufferL[bufferOffset];
            }
        }
        m_start = (m_start + sampleCount) % AudioBufferSamples;
    }

    // Signal audio worker thread that fresh PCM samples have arrived
    m_audioDirty.store(true, std::memory_order_release);
    m_workerCv.notify_one();
}

void PCM::Add(float const* const samples, uint32_t channels, size_t const count)
{
    AddToBuffer<1, 0>(samples, channels, count);
}
void PCM::Add(uint8_t const* const samples, uint32_t channels, size_t const count)
{
    AddToBuffer<128, 128>(samples, channels, count);
}
void PCM::Add(int16_t const* const samples, uint32_t channels, size_t const count)
{
    AddToBuffer<32768, 0>(samples, channels, count);
}

void PCM::ProcessAudioData(double secondsSinceLastFrame, uint32_t frame)
{
    // 1. Copy audio data from input buffer (lock to prevent tearing with audio thread writes)
    {
        std::lock_guard<std::mutex> lock(m_pcmMutex);
        CopyNewWaveformData(m_inputBufferL, m_waveformL);
        CopyNewWaveformData(m_inputBufferR, m_waveformR);
    }

    // 2. Update spectrum analyzer data for both channels (zero-allocation FFT)
    UpdateSpectrum(m_waveformL, m_spectrumL);
    UpdateSpectrum(m_waveformR, m_spectrumR);

    // 3. Align waveforms
    m_alignL.Align(m_waveformL);
    m_alignR.Align(m_waveformR);

    // 4. Update beat detection values
    m_bass.Update(m_spectrumL, secondsSinceLastFrame, frame);
    m_middles.Update(m_spectrumL, secondsSinceLastFrame, frame);
    m_treble.Update(m_spectrumL, secondsSinceLastFrame, frame);

    // 5. Populate inactive snapshot slot and publish atomically
    uint32_t const nextIndex = 1 - m_activeSnapshotIndex.load(std::memory_order_relaxed);
    FrameAudioData& target = m_audioSnapshots[nextIndex];

    std::copy(m_waveformL.begin(), m_waveformL.begin() + WaveformSamples, target.waveformLeft.begin());
    std::copy(m_waveformR.begin(), m_waveformR.begin() + WaveformSamples, target.waveformRight.begin());
    std::copy(m_spectrumL.begin(), m_spectrumL.begin() + SpectrumSamples, target.spectrumLeft.begin());
    std::copy(m_spectrumR.begin(), m_spectrumR.begin() + SpectrumSamples, target.spectrumRight.begin());

    target.bass = m_bass.CurrentRelative();
    target.mid = m_middles.CurrentRelative();
    target.treb = m_treble.CurrentRelative();

    target.bassAtt = m_bass.AverageRelative();
    target.midAtt = m_middles.AverageRelative();
    target.trebAtt = m_treble.AverageRelative();

    target.vol = (target.bass + target.mid + target.treb) * 0.333f;
    target.volAtt = (target.bassAtt + target.midAtt + target.trebAtt) * 0.333f;

    m_activeSnapshotIndex.store(nextIndex, std::memory_order_release);
}

void PCM::UpdateFrameAudioData(double secondsSinceLastFrame, uint32_t frame)
{
    m_pendingSecondsSinceLastFrame.store(secondsSinceLastFrame, std::memory_order_relaxed);
    m_pendingFrame.store(frame, std::memory_order_relaxed);

    if (m_workerRunning.load(std::memory_order_relaxed))
    {
        // Worker thread handles processing; wake if dirty
        m_workerCv.notify_one();
    }
    else
    {
        // Synchronous fallback if worker thread is not active
        ProcessAudioData(secondsSinceLastFrame, frame);
    }
}

auto PCM::GetFrameAudioData() const -> FrameAudioData
{
    // 100% Lock-Free atomic snapshot read: 0 mutex contention, 0 ms latency
    uint32_t const idx = m_activeSnapshotIndex.load(std::memory_order_acquire);
    return m_audioSnapshots[idx];
}

void PCM::UpdateSpectrum(const WaveformBuffer& waveformData, SpectrumBuffer& spectrumData)
{
    if (m_fftWaveformSamples.size() != AudioBufferSamples)
    {
        m_fftWaveformSamples.resize(AudioBufferSamples);
    }

    size_t oldI{0};
    for (size_t i = 0; i < AudioBufferSamples; i++)
    {
        // Damp the input into the FFT a bit, to reduce high-frequency noise:
        m_fftWaveformSamples[i] = 0.5f * (waveformData[i] + waveformData[oldI]);
        oldI = i;
    }

    // Zero-heap allocation: reuses pre-allocated member vector
    m_fft.TimeToFrequencyDomain(m_fftWaveformSamples, m_fftSpectralData);

    std::copy(m_fftSpectralData.begin(), m_fftSpectralData.end(), spectrumData.begin());
}

void PCM::CopyNewWaveformData(const WaveformBuffer& source, WaveformBuffer& destination)
{
    auto const bufferStartIndex = m_start;

    for (size_t i = 0; i < AudioBufferSamples; i++)
    {
        destination[i] = source[(bufferStartIndex + i) % AudioBufferSamples];
    }
}


} // namespace Audio
} // namespace libprojectM
