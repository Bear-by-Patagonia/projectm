#include <MilkdropPreset/BlurTexture.hpp>
#include <Renderer/Texture.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <set>
#include <vector>

namespace {

struct RGB3f
{
    float r{};
    float g{};
    float b{};
};

// Bit-exact UNORM quantizer/dequantizer
inline auto QuantizeUnorm(float v, int bits) -> uint32_t
{
    const float clamped = std::clamp(v, 0.0f, 1.0f);
    const float maxVal = static_cast<float>((1u << bits) - 1u);
    return static_cast<uint32_t>(std::lround(clamped * maxVal));
}

inline auto DequantizeUnorm(uint32_t q, int bits) -> float
{
    const float maxVal = static_cast<float>((1u << bits) - 1u);
    return static_cast<float>(q) / maxVal;
}

inline auto RoundTripRGB565(RGB3f c) -> RGB3f
{
    return {
        DequantizeUnorm(QuantizeUnorm(c.r, 5), 5),
        DequantizeUnorm(QuantizeUnorm(c.g, 6), 6),
        DequantizeUnorm(QuantizeUnorm(c.b, 5), 5)};
}

inline auto RoundTripRGB8(RGB3f c) -> RGB3f
{
    return {
        DequantizeUnorm(QuantizeUnorm(c.r, 8), 8),
        DequantizeUnorm(QuantizeUnorm(c.g, 8), 8),
        DequantizeUnorm(QuantizeUnorm(c.b, 8), 8)};
}

// Khronos OpenGL ES 3.0 Specification Section 2.1.3 / 2.1.4:
// Unsigned 11-bit float (5-bit exponent bias 15, 6-bit mantissa) and 10-bit float (5-bit exponent bias 15, 5-bit mantissa).
inline auto EncodeUnsignedFloat(float val, int mantissaBits) -> uint32_t
{
    if (std::isnan(val) || val <= 0.0f)
    {
        return 0u;
    }

    const uint32_t maxMantissa = (1u << mantissaBits) - 1u;
    const float maxFinite = static_cast<float>((1u << mantissaBits) + maxMantissa) *
                            std::ldexp(1.0f, 15 - mantissaBits); // 65024.0 for 11f, 64512.0 for 10f
    if (std::isinf(val) || val >= maxFinite)
    {
        return (30u << mantissaBits) | maxMantissa;
    }

    const float minNormal = std::ldexp(1.0f, -14); // 6.103515625e-5f
    if (val < minNormal)
    {
        // Denormalized: value = M * 2^(-14 - mantissaBits)
        const uint32_t mant = static_cast<uint32_t>(std::lround(std::ldexp(val, 14 + mantissaBits)));
        return std::min(mant, maxMantissa);
    }

    int exp2 = 0;
    const float frac = std::frexp(val, &exp2); // val = frac * 2^exp2, frac in [0.5, 1.0)
    // Normalized: val = (1 + M / 2^mantissaBits) * 2^(exp2 - 1)
    int biasedExp = (exp2 - 1) + 15;
    float scaledMant = (frac * 2.0f - 1.0f) * static_cast<float>(1u << mantissaBits);
    uint32_t mant = static_cast<uint32_t>(std::lround(scaledMant));
    if (mant > maxMantissa)
    {
        mant = 0u;
        ++biasedExp;
        if (biasedExp >= 31)
        {
            return (30u << mantissaBits) | maxMantissa;
        }
    }
    return (static_cast<uint32_t>(biasedExp) << mantissaBits) | mant;
}

inline auto DecodeUnsignedFloatExact(uint32_t bits, int mantissaBits) -> float
{
    const uint32_t maskMant = (1u << mantissaBits) - 1u;
    const uint32_t mant = bits & maskMant;
    const uint32_t exp = (bits >> mantissaBits) & 0x1Fu;
    if (exp == 0u)
    {
        return std::ldexp(static_cast<float>(mant), -14 - mantissaBits);
    }
    return std::ldexp(static_cast<float>((1u << mantissaBits) | mant), static_cast<int>(exp) - 15 - mantissaBits);
}

inline auto PackR11G11B10F(RGB3f c) -> uint32_t
{
    const uint32_t r11 = EncodeUnsignedFloat(c.r, 6) & 0x7FFu;
    const uint32_t g11 = EncodeUnsignedFloat(c.g, 6) & 0x7FFu;
    const uint32_t b10 = EncodeUnsignedFloat(c.b, 5) & 0x3FFu;
    return r11 | (g11 << 11u) | (b10 << 22u);
}

inline auto UnpackR11G11B10F(uint32_t packed) -> RGB3f
{
    return {
        DecodeUnsignedFloatExact(packed & 0x7FFu, 6),
        DecodeUnsignedFloatExact((packed >> 11u) & 0x7FFu, 6),
        DecodeUnsignedFloatExact((packed >> 22u) & 0x3FFu, 5)};
}

inline auto RoundTripR11G11B10F(RGB3f c) -> RGB3f
{
    return UnpackR11G11B10F(PackR11G11B10F(c));
}

// IEEE-754 binary16 (FP16) encoder/decoder (1 sign, 5 exp bias 15, 10 mantissa)
inline auto EncodeFP16(float val) -> uint16_t
{
    if (std::isnan(val))
    {
        return 0x7E00u;
    }
    const uint16_t sign = (std::signbit(val) ? 1u : 0u) << 15u;
    const float absVal = std::fabs(val);
    if (absVal == 0.0f)
    {
        return sign;
    }
    if (std::isinf(absVal) || absVal >= 65504.0f)
    {
        return sign | 0x7BFFu;
    }
    const float minNormal = std::ldexp(1.0f, -14);
    if (absVal < minNormal)
    {
        const uint16_t mant = static_cast<uint16_t>(std::min<long>(1023L, std::lround(std::ldexp(absVal, 24))));
        return sign | mant;
    }
    int exp2 = 0;
    const float frac = std::frexp(absVal, &exp2);
    int biasedExp = (exp2 - 1) + 15;
    uint32_t mant = static_cast<uint32_t>(std::lround((frac * 2.0f - 1.0f) * 1024.0f));
    if (mant > 1023u)
    {
        mant = 0u;
        ++biasedExp;
        if (biasedExp >= 31)
        {
            return sign | 0x7BFFu;
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(biasedExp) << 10u) | mant);
}

inline auto DecodeFP16(uint16_t h) -> float
{
    const float sign = ((h >> 15u) != 0u) ? -1.0f : 1.0f;
    const uint32_t exp = (h >> 10u) & 0x1Fu;
    const uint32_t mant = h & 0x3FFu;
    if (exp == 0u)
    {
        return sign * std::ldexp(static_cast<float>(mant), -24);
    }
    return sign * std::ldexp(static_cast<float>(1024u | mant), static_cast<int>(exp) - 25);
}

inline auto RoundTripRGBA16F(RGB3f c) -> RGB3f
{
    return {
        DecodeFP16(EncodeFP16(c.r)),
        DecodeFP16(EncodeFP16(c.g)),
        DecodeFP16(EncodeFP16(c.b))};
}

} // namespace

TEST(TextureFormatPrecision, ShadowGradientStepCountAndSobelNormalAccuracy)
{
    constexpr int kSamples = 4096;
    std::set<float> steps565R, steps8R, stepsR11F, steps16F;

    for (int i = 0; i <= kSamples; ++i)
    {
        const float v = 0.0005f + (0.05f - 0.0005f) * (static_cast<float>(i) / static_cast<float>(kSamples));
        steps565R.insert(RoundTripRGB565({v, v, v}).r);
        steps8R.insert(RoundTripRGB8({v, v, v}).r);
        stepsR11F.insert(RoundTripR11G11B10F({v, v, v}).r);
        steps16F.insert(RoundTripRGBA16F({v, v, v}).r);
    }

    // Verify logarithmic floating-point shadow fidelity vs linear UNORM quantization
    EXPECT_EQ(steps565R.size(), 3u);
    EXPECT_EQ(steps8R.size(), 14u);
    EXPECT_GE(stepsR11F.size(), 390u);
    EXPECT_GE(steps16F.size(), 2800u);

    // Simulate Martin's Sobel normal reconstruction (GetBlur1 finite difference) across a dark glass dome:
    // h(u) = 0.010 + 0.045 * sin(u * pi), bump = 12.0, z = 0.15
    constexpr int kDomePoints = 256;
    constexpr float kBump = 12.0f;
    double sqErr565 = 0.0, sqErr8 = 0.0, sqErrR11F = 0.0, sqErr16F = 0.0;
    int zeroGradCount565 = 0, zeroGradCountR11F = 0;

    auto evalHeight = [](float u) -> float {
        return 0.010f + 0.045f * std::sin(u * 3.14159265f);
    };
    auto normalAngleDeg = [](float dxRef, float dxTest) -> double {
        const double nx0 = -dxRef * kBump, nz0 = 0.15;
        const double nx1 = -dxTest * kBump, nz1 = 0.15;
        const double dot = (nx0 * nx1 + nz0 * nz1) /
                           (std::hypot(nx0, nz0) * std::hypot(nx1, nz1));
        return std::acos(std::clamp(dot, -1.0, 1.0)) * (180.0 / 3.141592653589793);
    };

    const float du = 4.0f / 1280.0f;
    for (int i = 8; i < kDomePoints - 8; ++i)
    {
        const float u = static_cast<float>(i) / static_cast<float>(kDomePoints);
        const float hPlus = evalHeight(u + du);
        const float hMinus = evalHeight(u - du);
        const float dxExact = hPlus - hMinus;

        const float dx565 = RoundTripRGB565({hPlus, 0, 0}).r - RoundTripRGB565({hMinus, 0, 0}).r;
        const float dx8 = RoundTripRGB8({hPlus, 0, 0}).r - RoundTripRGB8({hMinus, 0, 0}).r;
        const float dxR11 = RoundTripR11G11B10F({hPlus, 0, 0}).r - RoundTripR11G11B10F({hMinus, 0, 0}).r;
        const float dx16F = RoundTripRGBA16F({hPlus, 0, 0}).r - RoundTripRGBA16F({hMinus, 0, 0}).r;

        if (std::fabs(dxExact) > 1e-4f && dx565 == 0.0f) ++zeroGradCount565;
        if (std::fabs(dxExact) > 1e-4f && dxR11 == 0.0f) ++zeroGradCountR11F;

        const double e565 = normalAngleDeg(dxExact, dx565);
        const double e8 = normalAngleDeg(dxExact, dx8);
        const double eR11 = normalAngleDeg(dxExact, dxR11);
        const double e16F = normalAngleDeg(dxExact, dx16F);

        sqErr565 += e565 * e565;
        sqErr8 += e8 * e8;
        sqErrR11F += eR11 * eR11;
        sqErr16F += e16F * e16F;
    }

    const int nSamples = kDomePoints - 16;
    const double rms565 = std::sqrt(sqErr565 / nSamples);
    const double rms8 = std::sqrt(sqErr8 / nSamples);
    const double rmsR11F = std::sqrt(sqErrR11F / nSamples);
    const double rms16F = std::sqrt(sqErr16F / nSamples);

    std::printf("[PRECISION] Shadow [0.0005..0.05] steps: RGB565=%zu | RGB8=%zu | R11F_G11F_B10F=%zu | RGBA16F=%zu\n",
                steps565R.size(), steps8R.size(), stepsR11F.size(), steps16F.size());
    std::printf("[PRECISION] Martin Sobel Normal RMS Error (deg): RGB565=%.2f (flat=%d/%d) | RGB8=%.2f | R11F_G11F_B10F=%.3f (flat=%d/%d) | RGBA16F=%.4f\n",
                rms565, zeroGradCount565, nSamples, rms8, rmsR11F, zeroGradCountR11F, nSamples, rms16F);

    EXPECT_GT(zeroGradCount565, nSamples * 8 / 10); // RGB565 flattens >80% of curved dome into dead plateaus
    EXPECT_LT(rmsR11F, rms565 * 0.15);
    EXPECT_LT(rmsR11F, rms8 * 0.30);
    EXPECT_LT(rms16F, 0.10);
}

TEST(TextureFormatPrecision, IterativeDecayAndHdrSpecularPreservation)
{
    // 1. 60-frame decay = 0.98 starting from neutral grey (0.35, 0.35, 0.35)
    RGB3f c565{0.35f, 0.35f, 0.35f};
    RGB3f c8{0.35f, 0.35f, 0.35f};
    RGB3f cR11{0.35f, 0.35f, 0.35f};
    RGB3f c16F{0.35f, 0.35f, 0.35f};
    float exact = 0.35f;

    double maxRGSkew565 = 0.0;
    double maxRGSkewR11 = 0.0;

    for (int frame = 0; frame < 60; ++frame)
    {
        exact *= 0.98f;
        c565 = RoundTripRGB565({c565.r * 0.98f, c565.g * 0.98f, c565.b * 0.98f});
        c8 = RoundTripRGB8({c8.r * 0.98f, c8.g * 0.98f, c8.b * 0.98f});
        cR11 = RoundTripR11G11B10F({cR11.r * 0.98f, cR11.g * 0.98f, cR11.b * 0.98f});
        c16F = RoundTripRGBA16F({c16F.r * 0.98f, c16F.g * 0.98f, c16F.b * 0.98f});

        maxRGSkew565 = std::max<double>(maxRGSkew565, std::fabs(c565.r - c565.g));
        maxRGSkewR11 = std::max<double>(maxRGSkewR11, std::fabs(cR11.r - cR11.g));
    }

    // RGB565 freezes because round(v * 0.98 * 31) == round(v * 31) once step > 2% of value!
    EXPECT_GT(std::fabs(c565.r - exact), 0.15f);
    EXPECT_GT(maxRGSkew565, 0.005); // Visible green/magenta Mach tint in RGB565 (5-bit R vs 6-bit G)
    EXPECT_EQ(maxRGSkewR11, 0.0);   // Identical 6-bit mantissas for R and G -> zero R/G tint skew!
    EXPECT_NEAR(cR11.r, exact, 0.015f);
    EXPECT_NEAR(c16F.r, exact, 0.001f);

    // 2. HDR Phong specular peak > 1.0 (e.g., (2.45, 1.85, 1.30))
    const RGB3f hdrPeak{2.45f, 1.85f, 1.30f};
    const RGB3f out8 = RoundTripRGB8(hdrPeak);
    const RGB3f outR11 = RoundTripR11G11B10F(hdrPeak);
    const RGB3f out16F = RoundTripRGBA16F(hdrPeak);

    // UNORM clips all channels to 1.0, destroying highlight chromaticity
    EXPECT_FLOAT_EQ(out8.r, 1.0f);
    EXPECT_FLOAT_EQ(out8.g, 1.0f);
    EXPECT_FLOAT_EQ(out8.b, 1.0f);

    // R11F_G11F_B10F and RGBA16F preserve HDR > 1.0 chromatic ratios
    EXPECT_NEAR(outR11.r, 2.45f, 0.03f);
    EXPECT_NEAR(outR11.g, 1.85f, 0.02f);
    EXPECT_NEAR(outR11.b, 1.30f, 0.03f);
    EXPECT_NEAR(out16F.r, 2.45f, 0.002f);
}

TEST(TextureFormatPrecision, BandwidthModelMaliG31VsShieldTVAndFormatSelection)
{
    // Per-frame pixel traffic model across Warp (1R+1W full) + Blur1..3 pyramid (0.6875 R+W full equiv)
    // + FinalComposite (1R main + 0.3125R blur + 1W backbuffer):
    // Legacy (with 2 m_flipTexture copies = +4.0x full R+W): ~8.0x full-frame R+W equivalents.
    // Optimized Zero-Flip pipeline: ~4.0x full-frame R+W equivalents (before AFBC ~0.68x compression factor).
    auto calcBandwidthMBps = [](int w, int h, float fps, int bytesPerPixel, double fullFrameTrafficMult, double afbcFactor) -> double {
        const double pixelsPerSec = static_cast<double>(w) * static_cast<double>(h) * static_cast<double>(fps);
        return (pixelsPerSec * static_cast<double>(bytesPerPixel) * fullFrameTrafficMult * afbcFactor) / (1024.0 * 1024.0);
    };

    // Mali-G31 MP2 target: 1280x720 @ 60 FPS on ~5,120 MB/s practical DRAM budget
    const double maliLegacyRGB8Padded = calcBandwidthMBps(1280, 720, 60.0f, 4, 8.0, 1.0);   // Uncompressed RGBX8 + m_flipTexture
    const double maliZeroFlipR11F_AFBC = calcBandwidthMBps(1280, 720, 60.0f, 4, 4.0, 0.68); // Zero-Flip + 32bpp AFBC
    const double maliZeroFlipRGBA16F = calcBandwidthMBps(1280, 720, 60.0f, 8, 4.0, 0.82);   // 64bpp RGBA16F

    // Shield TV (Tegra X1+) target: 1920x1080 @ 60 FPS on ~25,600 MB/s LPDDR4 bus
    const double shieldZeroFlipR11F = calcBandwidthMBps(1920, 1080, 60.0f, 4, 4.0, 0.72);
    const double shieldZeroFlipRGBA16F = calcBandwidthMBps(1920, 1080, 60.0f, 8, 4.0, 0.82);

    std::printf("[BANDWIDTH] Mali-G31 720p60: Legacy=%.0f MB/s | ZeroFlip+32bpp(AFBC)=%.0f MB/s | ZeroFlip+RGBA16F=%.0f MB/s\n",
                maliLegacyRGB8Padded, maliZeroFlipR11F_AFBC, maliZeroFlipRGBA16F);
    std::printf("[BANDWIDTH] Shield TV 1080p60: ZeroFlip+32bpp=%.0f MB/s (%.1f%% bus) | ZeroFlip+RGBA16F=%.0f MB/s (%.1f%% bus)\n",
                shieldZeroFlipR11F, (shieldZeroFlipR11F / 25600.0) * 100.0,
                shieldZeroFlipRGBA16F, (shieldZeroFlipRGBA16F / 25600.0) * 100.0);

    EXPECT_LT(maliZeroFlipR11F_AFBC, maliLegacyRGB8Padded * 0.40);
    EXPECT_LT(maliZeroFlipR11F_AFBC, maliZeroFlipRGBA16F * 0.50);

    // Verify BlurTexture::SelectOptimalFormat chooses R11F_G11F_B10F when supported and AFBC-friendly RGBA8 fallback otherwise
    const auto primaryFmt = libprojectM::MilkdropPreset::BlurTexture::SelectOptimalFormat(true);
    EXPECT_EQ(primaryFmt.internalFormat, GL_R11F_G11F_B10F);
    EXPECT_EQ(primaryFmt.format, static_cast<GLenum>(GL_RGB));
    EXPECT_EQ(primaryFmt.type, static_cast<GLenum>(GL_UNSIGNED_INT_10F_11F_11F_REV));
    EXPECT_EQ(primaryFmt.bitsPerPixel, 32);
    EXPECT_TRUE(primaryFmt.isFloatingPoint);

    const auto fallbackFmt = libprojectM::MilkdropPreset::BlurTexture::SelectOptimalFormat(false);
    EXPECT_EQ(fallbackFmt.internalFormat, GL_RGBA8);
    EXPECT_EQ(fallbackFmt.format, static_cast<GLenum>(GL_RGBA));
    EXPECT_EQ(fallbackFmt.type, static_cast<GLenum>(GL_UNSIGNED_BYTE));
    EXPECT_EQ(fallbackFmt.bitsPerPixel, 32);
    EXPECT_FALSE(fallbackFmt.isFloatingPoint);

    // Verify Renderer::Texture::RequiredUnpackAlignment catches unaligned 3-byte RGB widths
    EXPECT_EQ(libprojectM::Renderer::Texture::RequiredUnpackAlignment(257, 3), 1);
    EXPECT_EQ(libprojectM::Renderer::Texture::RequiredUnpackAlignment(258, 3), 2);
    EXPECT_EQ(libprojectM::Renderer::Texture::RequiredUnpackAlignment(256, 3), 4);
    EXPECT_EQ(libprojectM::Renderer::Texture::RequiredUnpackAlignment(257, 4), 4);
}
