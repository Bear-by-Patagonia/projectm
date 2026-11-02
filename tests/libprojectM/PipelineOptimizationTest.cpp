#include <Audio/MilkdropFFT.hpp>
#include <Audio/PCM.hpp>
#include <MilkdropPreset/BlurTexture.hpp>
#include <MilkdropPreset/FinalComposite.hpp>
#include <MilkdropPreset/MilkdropShader.hpp>
#include <MilkdropPreset/PerFrameContext.hpp>
#include <MilkdropPreset/PresetFileParser.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <sstream>
#include <string>
#include <vector>

namespace {

using libprojectM::Audio::MilkdropFFT;
using libprojectM::Audio::PCM;
using libprojectM::MilkdropPreset::BlurTexture;
using libprojectM::MilkdropPreset::FinalComposite;
using libprojectM::MilkdropPreset::MilkdropShader;
using libprojectM::MilkdropPreset::PerFrameContext;
using libprojectM::MilkdropPreset::PresetFileParser;

// Evaluates barycentric interpolation of UV on a 2D triangle (V0, V1, V2) at screen point (px, py).
void InterpolateTriangleUV(const std::array<FinalComposite::FullscreenVertex, 3>& tri,
                           float px,
                           float py,
                           float& outU,
                           float& outV)
{
    const float x0 = tri[0].position.X();
    const float y0 = tri[0].position.Y();
    const float x1 = tri[1].position.X();
    const float y1 = tri[1].position.Y();
    const float x2 = tri[2].position.X();
    const float y2 = tri[2].position.Y();

    const float denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
    ASSERT_GT(std::abs(denom), 1e-6f);

    const float w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denom;
    const float w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denom;
    const float w2 = 1.0f - w0 - w1;

    // Ensure point lies inside or on the boundary of the fullscreen triangle.
    EXPECT_GE(w0, -1e-5f);
    EXPECT_GE(w1, -1e-5f);
    EXPECT_GE(w2, -1e-5f);

    outU = w0 * tri[0].uv.U() + w1 * tri[1].uv.U() + w2 * tri[2].uv.U();
    outV = w0 * tri[0].uv.V() + w1 * tri[1].uv.V() + w2 * tri[2].uv.V();
}

} // namespace

TEST(BlurTextureSafeMinMax, PreventsDivisionByZeroWhenMinEqualsMaxOrInverted)
{
    auto gmegabuf = projectm_eval_memory_buffer_create();
    PRJM_EVAL_F globalRegs[100]{};
    PerFrameContext ctx(gmegabuf, &globalRegs);
    ctx.RegisterBuiltinVariables();

    // Case 1: Equal min and max (previously produced blurMax == blurMin == 0.45 -> 1/0 = +Inf)
    *ctx.blur1_min = 0.5;
    *ctx.blur1_max = 0.5;
    *ctx.blur2_min = 0.5;
    *ctx.blur2_max = 0.5;
    *ctx.blur3_min = 0.5;
    *ctx.blur3_max = 0.5;

    BlurTexture::Values bMin{};
    BlurTexture::Values bMax{};
    BlurTexture::GetSafeBlurMinMaxValues(ctx, bMin, bMax);

    for (size_t i = 0; i < 3; ++i)
    {
        EXPECT_NEAR(bMin[i], 0.45f, 1e-5f);
        EXPECT_NEAR(bMax[i], 0.55f, 1e-5f);
        const float span = bMax[i] - bMin[i];
        EXPECT_GE(span, 0.0999f);
        EXPECT_TRUE(std::isfinite(1.0f / span));
    }

    // Progressive scale calculation from BlurTexture::Update must be strictly finite
    const float scale0 = 1.0f / (bMax[0] - bMin[0]);
    const float tempMin1 = (bMin[1] - bMin[0]) / (bMax[0] - bMin[0]);
    const float tempMax1 = (bMax[1] - bMin[0]) / (bMax[0] - bMin[0]);
    const float scale1 = 1.0f / (tempMax1 - tempMin1);
    const float tempMin2 = (bMin[2] - bMin[1]) / (bMax[1] - bMin[1]);
    const float tempMax2 = (bMax[2] - bMin[1]) / (bMax[1] - bMin[1]);
    const float scale2 = 1.0f / (tempMax2 - tempMin2);

    EXPECT_TRUE(std::isfinite(scale0));
    EXPECT_TRUE(std::isfinite(scale1));
    EXPECT_TRUE(std::isfinite(scale2));
    EXPECT_NEAR(scale0, 10.0f, 1e-4f);
    EXPECT_NEAR(scale1, 1.0f, 1e-4f);
    EXPECT_NEAR(scale2, 1.0f, 1e-4f);

    // Case 2: Inverted min > max
    *ctx.blur1_min = 0.8;
    *ctx.blur1_max = 0.2;
    *ctx.blur2_min = 0.9;
    *ctx.blur2_max = 0.1;
    *ctx.blur3_min = 0.7;
    *ctx.blur3_max = 0.3;

    BlurTexture::GetSafeBlurMinMaxValues(ctx, bMin, bMax);
    for (size_t i = 0; i < 3; ++i)
    {
        EXPECT_GE(bMax[i] - bMin[i], 0.0999f);
        EXPECT_TRUE(std::isfinite(1.0f / (bMax[i] - bMin[i])));
    }

    projectm_eval_memory_buffer_destroy(gmegabuf);
}

TEST(MilkdropFFTTest, SilenceProducesExactZeroAndSinePeaksMatchAnalyticalDFT)
{
    constexpr size_t kSamplesIn = 512;
    constexpr size_t kSamplesOut = 256; // numFrequencies = 512
    MilkdropFFT fftRaw(kSamplesIn, kSamplesOut, false, -1.0f);

    // 1. Silence must never produce NaN/Inf (guards against 0 * rsqrt(0) = NaN)
    std::vector<float> silence(kSamplesIn, 0.0f);
    std::vector<float> spectrum;
    fftRaw.TimeToFrequencyDomain(silence, spectrum);
    ASSERT_EQ(spectrum.size(), kSamplesOut);
    for (float v : spectrum)
    {
        EXPECT_TRUE(std::isfinite(v));
        EXPECT_FLOAT_EQ(v, 0.0f);
    }

    // 2. Dual-tone pure sine at exact integer bins k1=16, k2=96
    // For N=512 real sine of amplitude 1.0, |X[k]| = N / 2 = 256.0
    constexpr float kTwoPi = 6.28318530717958647692f;
    std::vector<float> dualTone(kSamplesIn, 0.0f);
    for (size_t n = 0; n < kSamplesIn; ++n)
    {
        const float phase1 = kTwoPi * 16.0f * static_cast<float>(n) / static_cast<float>(kSamplesIn);
        const float phase2 = kTwoPi * 96.0f * static_cast<float>(n) / static_cast<float>(kSamplesIn);
        dualTone[n] = std::sin(phase1) + 0.5f * std::cos(phase2);
    }

    fftRaw.TimeToFrequencyDomain(dualTone, spectrum);
    ASSERT_EQ(spectrum.size(), kSamplesOut);
    EXPECT_NEAR(spectrum[16], 256.0f, 5e-2f);
    EXPECT_NEAR(spectrum[96], 128.0f, 5e-2f);

    for (size_t k = 0; k < kSamplesOut; ++k)
    {
        EXPECT_TRUE(std::isfinite(spectrum[k]));
        if (k != 16 && k != 96)
        {
            EXPECT_LT(spectrum[k], 1e-2f);
        }
    }

    // 3. Production MilkdropFFT config (equalized + bell envelope)
    MilkdropFFT fftProd(480, 512, true, 1.0f);
    std::vector<float> prodWave(576, 0.0f);
    for (size_t n = 0; n < 480; ++n)
    {
        prodWave[n] = 64.0f * std::sin(kTwoPi * 12.0f * static_cast<float>(n) / 480.0f);
    }
    std::vector<float> prodSpec;
    fftProd.TimeToFrequencyDomain(prodWave, prodSpec);
    ASSERT_EQ(prodSpec.size(), 512u);
    for (float v : prodSpec)
    {
        EXPECT_TRUE(std::isfinite(v));
        EXPECT_GE(v, 0.0f);
    }
}

TEST(PCMMonoDeduplication, MonoInputProducesIdenticalChannelsAndStereoPreservesSeparation)
{
    PCM pcm;
    std::vector<float> monoSamples(576, 0.0f);
    for (size_t i = 0; i < monoSamples.size(); ++i)
    {
        monoSamples[i] = 0.75f * std::sin(0.1f * static_cast<float>(i));
    }
    pcm.Add(monoSamples.data(), 1, monoSamples.size());
    // Wait briefly or trigger processing via a second add/sleep if worker thread is active
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
    auto monoSnap = pcm.GetFrameAudioData();
    for (size_t i = 0; i < libprojectM::Audio::WaveformSamples; ++i)
    {
        EXPECT_FLOAT_EQ(monoSnap.waveformLeft[i], monoSnap.waveformRight[i]);
    }
    for (size_t i = 0; i < libprojectM::Audio::SpectrumSamples; ++i)
    {
        EXPECT_FLOAT_EQ(monoSnap.spectrumLeft[i], monoSnap.spectrumRight[i]);
    }

    // Now feed distinct stereo channels (L = low freq, R = high freq)
    std::vector<float> stereoSamples(576 * 2, 0.0f);
    for (size_t i = 0; i < 576; ++i)
    {
        stereoSamples[i * 2 + 0] = 0.8f * std::sin(0.05f * static_cast<float>(i));
        stereoSamples[i * 2 + 1] = 0.8f * std::sin(0.45f * static_cast<float>(i));
    }
    pcm.Add(stereoSamples.data(), 2, 576);
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
    auto stereoSnap = pcm.GetFrameAudioData();
    bool foundDiff = false;
    for (size_t i = 0; i < libprojectM::Audio::SpectrumSamples; ++i)
    {
        if (std::abs(stereoSnap.spectrumLeft[i] - stereoSnap.spectrumRight[i]) > 1e-3f)
        {
            foundDiff = true;
            break;
        }
    }
    EXPECT_TRUE(foundDiff);
}

TEST(MilkdropShaderAnalysis, EmitsPerFrameStd140UboAndDetectsPolarAndHueUsageAccurately)
{
    static_assert(sizeof(MilkdropShader::PerFrameUBOData) == 384,
                  "PerFrameUBOData must remain 384 bytes (24 x vec4)");

    // 1. Martin Sobel Glass composite shader: uses texsize (_c7), time (_c2), treb (_c3), blur1 (_c5),
    //    but NO rad, NO ang, NO hue_shader, and NO 3D rot_* matrices.
    const std::string martinGlass = R"(
        shader_body
        {
            // Comment mentioning rad, ang, and hue_shader must NOT trigger false positives!
            float radical = 4.0;
            float2 d = texsize.zw * radical;
            float dx = GetBlur1(uv + float2(1.0, 0.0) * d).x - GetBlur1(uv - float2(1.0, 0.0) * d).x;
            float dy = GetBlur1(uv + float2(0.0, 1.0) * d).x - GetBlur1(uv - float2(0.0, 1.0) * d).x;
            float3 N = normalize(float3(-dx * 2.5, dy * 2.5, 0.15));
            float2 uv_refract = uv + N.xy * 0.04;
            float3 refracted_color = GetMain(uv_refract);
            float3 light_dir = normalize(float3(sin(time * 0.5), cos(time * 0.3), 1.0));
            float3 half_v = normalize(light_dir + float3(0.0, 0.0, 1.0));
            float spec = pow(max(0.0, dot(N, half_v)), 24.0) * (0.8 + treb * 1.5);
            ret = refracted_color + spec;
        }
    )";

    MilkdropShader compMartin(MilkdropShader::ShaderType::CompositeShader);
    const std::string glslMartin = compMartin.TranspileToGLSLSource(martinGlass);

    EXPECT_FALSE(compMartin.UsesPolarCoords());
    EXPECT_FALSE(compMartin.UsesHueShader());
    EXPECT_TRUE(compMartin.UsesPerFrameUBO());
    EXPECT_NE(glslMartin.find("layout(std140) uniform PerFrameBlock"), std::string::npos);
    EXPECT_EQ(glslMartin.find("uniform vec4 _c2;"), std::string::npos);
    EXPECT_EQ(glslMartin.find("uniform vec4 _c7;"), std::string::npos);
    EXPECT_EQ(glslMartin.find("rot_s1"), std::string::npos);

    // 2. Polar composite shader using rad and ang
    const std::string polarShader = R"(
        shader_body
        {
            float2 swirl = uv + 0.03 * float2(cos(ang + q1), sin(ang - q1)) * rad;
            ret = GetMain(swirl);
        }
    )";
    MilkdropShader compPolar(MilkdropShader::ShaderType::CompositeShader);
    const std::string glslPolar = compPolar.TranspileToGLSLSource(polarShader);
    EXPECT_TRUE(compPolar.UsesPolarCoords());
    EXPECT_FALSE(compPolar.UsesHueShader());
    EXPECT_TRUE(compPolar.UsesPerFrameUBO());
    EXPECT_NE(glslPolar.find("layout(std140) uniform PerFrameBlock"), std::string::npos);

    // 3. Hue-tinted composite shader using hue_shader
    const std::string hueShader = R"(
        shader_body
        {
            ret = GetMain(uv) * hue_shader;
        }
    )";
    MilkdropShader compHue(MilkdropShader::ShaderType::CompositeShader);
    compHue.TranspileToGLSLSource(hueShader);
    EXPECT_FALSE(compHue.UsesPolarCoords());
    EXPECT_TRUE(compHue.UsesHueShader());

    // 4. Shader using 3D rotation matrix rot_s1 + sampler1D + float2 texsize_ declaration
    const std::string rotAndSampler1D = R"(
        sampler1D sampler_pw_noise_lq;
        uniform float2 texsize_noise_lq;
        shader_body
        {
            float3 n = tex2D(sampler_pw_noise_lq, uv).xyz;
            float3 rotated = mul(float4(n, 1.0), rot_s1);
            ret = rotated + q2;
        }
    )";
    MilkdropShader warpRot(MilkdropShader::ShaderType::WarpShader);
    const std::string glslRot = warpRot.TranspileToGLSLSource(rotAndSampler1D);
    EXPECT_TRUE(warpRot.UsesPerFrameUBO());
    // rot_s1 must remain a loose mat3x4 uniform OUTSIDE PerFrameBlock
    EXPECT_NE(glslRot.find("uniform mat3x4 rot_s1;"), std::string::npos);
    const size_t blockStart = glslRot.find("layout(std140) uniform PerFrameBlock");
    ASSERT_NE(blockStart, std::string::npos);
    const size_t blockEnd = glslRot.find("};", blockStart);
    ASSERT_NE(blockEnd, std::string::npos);
    const std::string blockBody = glslRot.substr(blockStart, blockEnd - blockStart);
    EXPECT_EQ(blockBody.find("rot_s1"), std::string::npos);
}

TEST(FinalCompositeGeometry, FullscreenTriangleMatchesQuadCornersWithHalfTexelOffset)
{
    constexpr int kWidth = 1920;
    constexpr int kHeight = 1080;
    const float halfTexelW = 0.5f / static_cast<float>(kWidth);
    const float halfTexelH = 0.5f / static_cast<float>(kHeight);

    const auto tri = FinalComposite::ComputeFullscreenTriangle(kWidth, kHeight);

    float u = 0.0f;
    float v = 0.0f;

    // Top-left screen corner (-1, +1) -> MilkDrop math UV (0, 0) + halfTexel
    InterpolateTriangleUV(tri, -1.0f, +1.0f, u, v);
    EXPECT_NEAR(u, 0.0f + halfTexelW, 1e-6f);
    EXPECT_NEAR(v, 0.0f + halfTexelH, 1e-6f);

    // Top-right screen corner (+1, +1) -> MilkDrop math UV (1, 0) + halfTexel
    InterpolateTriangleUV(tri, +1.0f, +1.0f, u, v);
    EXPECT_NEAR(u, 1.0f + halfTexelW, 1e-6f);
    EXPECT_NEAR(v, 0.0f + halfTexelH, 1e-6f);

    // Bottom-left screen corner (-1, -1) -> MilkDrop math UV (0, 1) + halfTexel
    InterpolateTriangleUV(tri, -1.0f, -1.0f, u, v);
    EXPECT_NEAR(u, 0.0f + halfTexelW, 1e-6f);
    EXPECT_NEAR(v, 1.0f + halfTexelH, 1e-6f);

    // Bottom-right screen corner (+1, -1) -> MilkDrop math UV (1, 1) + halfTexel
    InterpolateTriangleUV(tri, +1.0f, -1.0f, u, v);
    EXPECT_NEAR(u, 1.0f + halfTexelW, 1e-6f);
    EXPECT_NEAR(v, 1.0f + halfTexelH, 1e-6f);

    // Viewport center (0, 0) -> MilkDrop math UV (0.5, 0.5) + halfTexel
    InterpolateTriangleUV(tri, 0.0f, 0.0f, u, v);
    EXPECT_NEAR(u, 0.5f + halfTexelW, 1e-6f);
    EXPECT_NEAR(v, 0.5f + halfTexelH, 1e-6f);
}

TEST(PresetFileParserOptimization, ConcatenatesMultiDigitSequentialCodeLinesAccurately)
{
    std::stringstream ss;
    ss << "MILKDROP_PRESET_VERSION=201\n";
    for (int i = 1; i <= 15; ++i)
    {
        ss << "per_frame_" << i << "=`q" << i << " = " << i << ";\n";
    }
    ss << "per_frame_17=q17 = 17;\n"; // Gap at 16 must stop iteration cleanly

    PresetFileParser parser;
    ASSERT_TRUE(parser.Read(ss));

    const std::string code = parser.GetCode("per_frame_");
    EXPECT_NE(code.find("q1 = 1;"), std::string::npos);
    EXPECT_NE(code.find("q10 = 10;"), std::string::npos);
    EXPECT_NE(code.find("q15 = 15;"), std::string::npos);
    EXPECT_EQ(code.find('`'), std::string::npos);
    EXPECT_EQ(code.find("q17 = 17;"), std::string::npos);
}
