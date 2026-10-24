#include "Audio/WaveformAligner.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace libprojectM {
namespace Audio {

WaveformAligner::WaveformAligner()
{
    static const uint32_t maxOctaves{10};
    // For AudioBufferSamples = 576 and WaveformSamples = 480:
    // floor(log2(96)) = 6
    static const uint32_t numOctaves{static_cast<uint32_t>(std::floor(std::log(AudioBufferSamples - WaveformSamples) / std::log(2.0f)))};
    m_octaves = numOctaves > maxOctaves ? maxOctaves : numOctaves;

    m_aligmentWeights.resize(m_octaves);
    m_firstNonzeroWeights.resize(m_octaves);
    m_lastNonzeroWeights.resize(m_octaves);
    m_octaveSamples.resize(m_octaves);
    m_octaveSampleSpacing.resize(m_octaves);
    m_oldWaveformMips.resize(m_octaves);
    m_newWaveformMips.resize(m_octaves);

    m_octaveSamples[0] = AudioBufferSamples;
    m_octaveSampleSpacing[0] = AudioBufferSamples - WaveformSamples;
    for (uint32_t octave = 1; octave < m_octaves; octave++)
    {
        m_octaveSamples[octave] = m_octaveSamples[octave - 1] / 2;
        m_octaveSampleSpacing[octave] = m_octaveSampleSpacing[octave - 1] / 2;
    }
}

void WaveformAligner::ResampleOctaves(std::vector<WaveformBuffer>& dstWaveformMips, WaveformBuffer& newWaveform)
{
    // Octave 0 is a direct copy of the new waveform
    std::copy(newWaveform.begin(), newWaveform.end(), dstWaveformMips[0].begin());

    // Calculate mip levels
    // This downsamples the previous octave's waveform by a factor of 2
    for (uint32_t octave = 1; octave < m_octaves; octave++)
    {
        for (uint32_t sample = 0; sample < m_octaveSamples[octave]; sample++)
        {
            dstWaveformMips[octave][sample] = 0.5f * (dstWaveformMips[octave - 1][sample * 2] + dstWaveformMips[octave - 1][sample * 2 + 1]);
        }
    }
}

void WaveformAligner::GenerateWeights()
{
    // The below is performed only on the first fill.
    for (uint32_t octave = 0; octave < m_octaves; octave++)
    {
        // For example:
        //  m_octaveSampleSpacing[octave] == 4
        //  m_octaveSamples[octave] == 36
        //  (so we test 32 samples, w/4 offsets)
        uint32_t const compareSamples = m_octaveSamples[octave] - m_octaveSampleSpacing[octave];

        for (uint32_t sample = 0; sample < compareSamples; sample++)
        {
            // Take a reference to the alignment weights and set them with the computation
            // below.
            auto& weightRef = m_aligmentWeights[octave][sample];

            // Start with pyramid-shaped PDF, from 0..1..0
            if (sample < compareSamples / 2)
            {
                weightRef = static_cast<float>(sample * 2) / static_cast<float>(compareSamples);
            }
            else
            {
                weightRef = static_cast<float>((compareSamples - 1 - sample) * 2) / static_cast<float>(compareSamples);
            }

            /*
             * TWEAK how much the center matters, vs. the edges:
             *
             * weight[i] = 5.0*((2*i/compareSamples) - 0.8) + 0.8
             * Solving for weight[i] == 0 we get:\
             *
             * 2*i/compareSamples = -0.8/5 + 0.8
             * i = 0.32*compareSamples
             *
             * The weight distribution is symmetric so the falling side gives:
             *
             * i = 0.68*compareSamples
             */
            weightRef = (weightRef - 0.8f) * 5.0f + 0.8f;

            // Clamp
            // Needed because the TWEAK above results in weights from -3.2 to 1.8
            if (weightRef > 1.0f)
            {
                weightRef = 1.0f;
            }
            if (weightRef < 0.0f)
            {
                weightRef = 0.0f;
            }
        }

        uint32_t sample{};
        // The code below also is only needed because of the TWEAK above, which zeroes
        // a total of 64% of the weights.
        while (m_aligmentWeights[octave][sample] == 0 && sample < compareSamples)
        {
            sample++;
        }
        m_firstNonzeroWeights[octave] = sample;

        sample = compareSamples - 1;
        while (m_aligmentWeights[octave][sample] == 0 && compareSamples > 1)
        {
            sample--;
        }
        m_lastNonzeroWeights[octave] = sample;
    }
}

int WaveformAligner::CalculateOffset(std::vector<WaveformBuffer>& newWaveformMips)
{
    /*
     * Note that we use signed variables here because we need to check for negatives even
     * if we clamp to only positive values and 0.
     */
    int alignOffset{};
    int offsetStart{};
    int offsetEnd{static_cast<int>(m_octaveSampleSpacing[m_octaves - 1])};

    // Find best match for alignment
    // Note that we need a signed iterator here because the termination condition is octave < 0
    for (int octave = static_cast<int>(m_octaves) - 1; octave >= 0; octave--)
    {
        int lowestErrorOffset{-1};
        float lowestErrorAmount{};

        const uint32_t firstWeight = m_firstNonzeroWeights[octave];
        const uint32_t lastWeight = m_lastNonzeroWeights[octave];
        const float* const oldMipPtr = m_oldWaveformMips[octave].data();
        const float* const weightPtr = m_aligmentWeights[octave].data();
        const float* const newMipBase = newWaveformMips[octave].data();

        // For each octave, find the offset that maximizes the correlation between waveforms.
        for (int sample = offsetStart; sample < offsetEnd; sample++)
        {
            float errorSum{};
            const float* const newMipPtr = newMipBase + sample;
            uint32_t i = firstWeight;

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
            float32x4_t vSum = vdupq_n_f32(0.0f);
            for (; i + 3 <= lastWeight; i += 4)
            {
                float32x4_t vNew = vld1q_f32(newMipPtr + i);
                float32x4_t vOld = vld1q_f32(oldMipPtr + i);
                float32x4_t vW   = vld1q_f32(weightPtr + i);
                float32x4_t vDiff = vabdq_f32(vNew, vOld);
                vSum = vmlaq_f32(vSum, vDiff, vW);
            }
#if defined(__aarch64__)
            errorSum = vaddvq_f32(vSum);
#else
            float32x2_t vPair = vadd_f32(vget_low_f32(vSum), vget_high_f32(vSum));
            vPair = vpadd_f32(vPair, vPair);
            errorSum = vget_lane_f32(vPair, 0);
#endif
#endif
            // Perform the cross-correlation. Note that we shift the new waveform but not the old
            // one because we're looking for the offset between them that produces the lowest error.
            for (; i <= lastWeight; i++)
            {
                errorSum += std::abs((newMipPtr[i] - oldMipPtr[i]) * weightPtr[i]);
            }

            if (lowestErrorOffset == -1 || errorSum < lowestErrorAmount)
            {
                lowestErrorOffset = static_cast<int>(sample);
                lowestErrorAmount = errorSum;
            }
        }

        // Now use 'lowestErrorOffset' to guide bounds of search in next octave:
        //  m_octaveSampleSpacing[octave] == 8
        //  m_octaveSamples[octave] == 72
        //     -say 'lowestErrorOffset' was 2
        //     -that corresponds to samples 4 & 5 of the next octave
        //     -also, expand about this by 2 samples?  YES.
        //  (so we'd test 64 samples, w/8->4 offsets)
        if (octave > 0)
        {
            offsetStart = lowestErrorOffset * 2 - 1;
            offsetEnd = lowestErrorOffset * 2 + 2 + 1;
            if (offsetStart < 0)
            {
                /*
                 * This line is what prevents us from checking negative offsets.
                 * There should be no impact to allowing offsetStart to be negative as long as
                 * its magnitude is less than m_firstNonzeroWeights[octave-1]. However, this
                 * is what the original milkdrop code does so we stick with that behavior.
                 */
                offsetStart = 0;
            }
            if (offsetEnd > static_cast<int>(m_octaveSampleSpacing[octave - 1]))
            {
                offsetEnd = static_cast<int>(m_octaveSampleSpacing[octave - 1]);
            }
        }
        else
        {
            alignOffset = lowestErrorOffset;
        }
    }

    return alignOffset;
}

void WaveformAligner::Align(WaveformBuffer& newWaveform)
{
    if (m_octaves < 4)
    {
        // The original code does not align if there isn't enough margin for
        // alignment but has no explanation for why the limit is 2**4 samples.
        return;
    }


    ResampleOctaves(m_newWaveformMips, newWaveform);

    if (!m_alignWaveReady)
    {
        GenerateWeights();
        // Mark that weights have been calculated.
        m_alignWaveReady = true;
    }

    int alignOffset = CalculateOffset(m_newWaveformMips);

    // Finally, apply the results by scooting the aligned samples so that they start at index 0.
    // This is the second place where we limit negative offsets.
    if (alignOffset > 0)
    {
        std::copy_n(newWaveform.begin() + alignOffset, WaveformSamples, newWaveform.begin());

        // Set remaining samples to zero.
        std::fill_n(newWaveform.begin() + WaveformSamples, AudioBufferSamples - WaveformSamples, 0.0f);
    }

    // Store mip levels for the next frame. Note that we need to recalculate the mips for the *shifted*
    // waveform, so we can't reuse the previous mips.
    ResampleOctaves(m_oldWaveformMips, newWaveform);
}


} // namespace Audio
} // namespace libprojectM
