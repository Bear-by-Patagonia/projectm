#include <gtest/gtest.h>

#include <MilkdropPreset/MilkdropShader.hpp>
#include <MilkdropStaticShaders.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace {

enum class WrapMode
{
    ClampToEdge,
    Repeat,
    MirroredRepeat
};

struct Image2D
{
    int width{0};
    int height{0};
    std::vector<float> texels;

    Image2D(int w, int h)
        : width(w)
        , height(h)
        , texels(static_cast<size_t>(w * h), 0.0f)
    {
    }

    float& At(int x, int y)
    {
        return texels[static_cast<size_t>(y * width + x)];
    }

    float At(int x, int y) const
    {
        return texels[static_cast<size_t>(y * width + x)];
    }
};

// Simulates CopyTexture(..., flipVertical = true): row k of flipTex equals row (H - 1 - k) of origTex.
static Image2D BuildVerticallyFlippedCopy(const Image2D& orig)
{
    Image2D flipped(orig.width, orig.height);
    for (int y = 0; y < orig.height; ++y)
    {
        for (int x = 0; x < orig.width; ++x)
        {
            flipped.At(x, y) = orig.At(x, orig.height - 1 - y);
        }
    }
    return flipped;
}

static int WrapTexelIndex(int i, int size, WrapMode mode)
{
    switch (mode)
    {
        case WrapMode::ClampToEdge:
            return std::max(0, std::min(size - 1, i));
        case WrapMode::Repeat: {
            int m = i % size;
            return (m < 0) ? (m + size) : m;
        }
        case WrapMode::MirroredRepeat: {
            int period = 2 * size;
            int m = i % period;
            if (m < 0)
            {
                m += period;
            }
            return (m < size) ? m : (period - 1 - m);
        }
    }
    return 0;
}

// Reference OpenGL ES 3.0 / GL 3.3 specification §3.8.6 / §8.14.2 bilinear sampler.
static float SampleBilinearGL(const Image2D& img, float u, float v, WrapMode mode)
{
    auto mapCoord = [](float c, int size, WrapMode wm) -> float {
        if (wm == WrapMode::Repeat)
        {
            float f = c - std::floor(c);
            return f * static_cast<float>(size) - 0.5f;
        }
        if (wm == WrapMode::MirroredRepeat)
        {
            float fl = std::floor(c);
            float f = c - fl;
            int flInt = static_cast<int>(fl);
            float uMirrored = ((flInt & 1) != 0) ? (1.0f - f) : f;
            return uMirrored * static_cast<float>(size) - 0.5f;
        }
        return c * static_cast<float>(size) - 0.5f;
    };

    float const fx = mapCoord(u, img.width, mode);
    float const fy = mapCoord(v, img.height, mode);

    int const x0 = static_cast<int>(std::floor(fx));
    int const y0 = static_cast<int>(std::floor(fy));
    float const ax = fx - static_cast<float>(x0);
    float const ay = fy - static_cast<float>(y0);

    int const ix0 = WrapTexelIndex(x0, img.width, mode);
    int const ix1 = WrapTexelIndex(x0 + 1, img.width, mode);
    int const iy0 = WrapTexelIndex(y0, img.height, mode);
    int const iy1 = WrapTexelIndex(y0 + 1, img.height, mode);

    float const r0 = img.At(ix0, iy0) * (1.0f - ax) + img.At(ix1, iy0) * ax;
    float const r1 = img.At(ix0, iy1) * (1.0f - ax) + img.At(ix1, iy1) * ax;
    return r0 * (1.0f - ay) + r1 * ay;
}

// Reference 4-Tap Catmull-Rom bicubic hardware-bilinear sampler matching MilkdropShader.cpp.
static float SampleCatmullRom4TapGL(const Image2D& img, float u, float v, WrapMode mode)
{
    float const tsX = static_cast<float>(img.width);
    float const tsY = static_cast<float>(img.height);
    float const itsX = 1.0f / tsX;
    float const itsY = 1.0f / tsY;

    float const spX = u * tsX;
    float const spY = v * tsY;
    float const tcX = std::floor(spX - 0.5f) + 0.5f;
    float const tcY = std::floor(spY - 0.5f) + 0.5f;
    float const fX = spX - tcX;
    float const fY = spY - tcY;

    auto weights = [](float f, float& s0, float& s1, float& f0, float& f1) {
        float const f2 = f * f;
        float const f3 = f2 * f;
        float const w0 = f2 - 0.5f * (f3 + f);
        float const w1 = 1.5f * f3 - 2.5f * f2 + 1.0f;
        float const w3 = 0.5f * (f3 - f2);
        float const w2 = 1.0f - w0 - w1 - w3;
        s0 = w0 + w1;
        s1 = w2 + w3;
        f0 = w1 / std::max(s0, 1e-5f);
        f1 = w3 / std::max(s1, 1e-5f);
    };

    float s0X{}, s1X{}, f0X{}, f1X{};
    float s0Y{}, s1Y{}, f0Y{}, f1Y{};
    weights(fX, s0X, s1X, f0X, f1X);
    weights(fY, s0Y, s1Y, f0Y, f1Y);

    float const t0X = (tcX - 1.0f + f0X) * itsX;
    float const t1X = (tcX + 1.0f + f1X) * itsX;
    float const t0Y = (tcY - 1.0f + f0Y) * itsY;
    float const t1Y = (tcY + 1.0f + f1Y) * itsY;

    return SampleBilinearGL(img, t0X, t0Y, mode) * (s0X * s0Y) +
           SampleBilinearGL(img, t1X, t0Y, mode) * (s1X * s0Y) +
           SampleBilinearGL(img, t0X, t1Y, mode) * (s0X * s1Y) +
           SampleBilinearGL(img, t1X, t1Y, mode) * (s1X * s1Y);
}

static Image2D BuildAsymmetricTestField(int w, int h)
{
    Image2D img(w, h);
    float const invW = 1.0f / static_cast<float>(w);
    float const invH = 1.0f / static_cast<float>(h);
    for (int y = 0; y < h; ++y)
    {
        float const ny = static_cast<float>(y + 1) * invH;
        for (int x = 0; x < w; ++x)
        {
            float const nx = static_cast<float>(x + 1) * invW;
            // Non-palindromic vertical + horizontal HDR field in [0, 3.5] so any Y inversion or half-texel shift is caught.
            img.At(x, y) = 0.45f * nx +
                           1.85f * ny * ny +
                           0.65f * (0.5f + 0.5f * std::sin(static_cast<float>(x) * 0.7f - static_cast<float>(y) * 1.3f));
        }
    }
    return img;
}

} // namespace

TEST(ZeroFlipMath, BilinearEquivalenceAcrossAllWrapModesAndBoundaries)
{
    for (int h : {16, 17, 24, 27})
    {
        Image2D const orig = BuildAsymmetricTestField(20, h);
        Image2D const flip = BuildVerticallyFlippedCopy(orig);

        const std::vector<float> testCoords = {
            -2.0f, -1.37f, -1.0f, -0.5f, -0.01f, 0.0f,
            0.5f / static_cast<float>(h),
            1.0f / static_cast<float>(h),
            0.1234f, 0.3333f, 0.5f, 0.7891f,
            1.0f - 0.5f / static_cast<float>(h),
            1.0f, 1.01f, 1.5f, 2.0f, 3.42f
        };

        for (WrapMode mode : {WrapMode::ClampToEdge, WrapMode::Repeat, WrapMode::MirroredRepeat})
        {
            for (float u : testCoords)
            {
                for (float v : testCoords)
                {
                    float const legacySample = SampleBilinearGL(flip, u, v, mode);
                    float const zeroFlipSample = SampleBilinearGL(orig, u, 1.0f - v, mode);
                    float const tol = 1e-5f * std::max(1.0f, std::abs(legacySample));
                    EXPECT_NEAR(legacySample, zeroFlipSample, tol)
                        << "Mismatch at u=" << u << ", v=" << v << ", h=" << h << ", wrap=" << static_cast<int>(mode);
                }
            }
        }
    }
}

TEST(ZeroFlipMath, CatmullRom4TapEquivalenceAndZeroNaNAtExactTexelCenters)
{
    Image2D const orig = BuildAsymmetricTestField(32, 24);
    Image2D const flip = BuildVerticallyFlippedCopy(orig);

    for (int row = 0; row < orig.height; ++row)
    {
        // Exact texel centers trigger f = 0.0 where unguarded w3 / s1 would divide 0/0.
        float const vCenter = (static_cast<float>(row) + 0.5f) / static_cast<float>(orig.height);
        float const uCenter = 7.5f / static_cast<float>(orig.width);
        float const legacyVal = SampleCatmullRom4TapGL(flip, uCenter, vCenter, WrapMode::ClampToEdge);
        float const zeroFlipVal = SampleCatmullRom4TapGL(orig, uCenter, 1.0f - vCenter, WrapMode::ClampToEdge);
        EXPECT_TRUE(std::isfinite(legacyVal));
        EXPECT_TRUE(std::isfinite(zeroFlipVal));
        EXPECT_NEAR(legacyVal, zeroFlipVal, 1e-5f * std::max(1.0f, std::abs(legacyVal)));
    }

    for (int step = 1; step < 40; ++step)
    {
        float const u = static_cast<float>(step) / 41.0f;
        float const v = static_cast<float>(40 - step) / 41.0f;
        for (WrapMode mode : {WrapMode::ClampToEdge, WrapMode::Repeat})
        {
            float const legacyVal = SampleCatmullRom4TapGL(flip, u, v, mode);
            float const zeroFlipVal = SampleCatmullRom4TapGL(orig, u, 1.0f - v, mode);
            EXPECT_NEAR(legacyVal, zeroFlipVal, 1e-5f * std::max(1.0f, std::abs(legacyVal)));
        }
    }
}

TEST(ZeroFlipTranspiler, CompositeMartinSobelInvertsMainButPreservesGetBlurAndUv)
{
    using libprojectM::MilkdropPreset::MilkdropShader;

    const std::string martinGlassShader = R"(
        shader_body
        {
            float2 d = texsize.zw * 4.0;
            float dx = GetBlur1(uv + float2(1.0, 0.0) * d).r - GetBlur1(uv - float2(1.0, 0.0) * d).r;
            float dy = GetBlur1(uv + float2(0.0, 1.0) * d).r - GetBlur1(uv - float2(0.0, 1.0) * d).r;
            float3 N = normalize(float3(-dx * 2.5, dy * 2.5, 0.15));
            float2 uv_refract = uv + N.xy * 0.04;
            ret = GetMain(uv_refract) + GetPixel(uv) * 0.1;
        }
    )";

    MilkdropShader compShader(MilkdropShader::ShaderType::CompositeShader);
    std::string const glsl = compShader.TranspileToGLSLSource(martinGlassShader);

    // Main texture wrapper must invert Y (1.0 - _u.y) and replace direct texture(sampler_main, ...) calls.
    EXPECT_NE(glsl.find("_pm_texMain_sampler_main("), std::string::npos) << glsl;
    EXPECT_NE(glsl.find("1.0 - _u.y"), std::string::npos) << glsl;

    // Inside PS() and main(), there must be no unflipped raw texture(sampler_main, ...) calls left.
    size_t const psBodyPos = glsl.find("void PS(");
    ASSERT_NE(psBodyPos, std::string::npos);
    std::string const psBody = glsl.substr(psBodyPos);
    EXPECT_NE(psBody.find("_pm_texMain_sampler_main("), std::string::npos) << psBody;
    EXPECT_EQ(psBody.find("texture(sampler_main,"), std::string::npos) << psBody;

    // Blur1 is already flipped in pass 0 of BlurTexture::Update, so sampler_blur1 must NOT be wrapped.
    EXPECT_NE(psBody.find("texture(sampler_blur1,"), std::string::npos) << psBody;
    EXPECT_EQ(glsl.find("_pm_texMain_sampler_blur1"), std::string::npos) << glsl;
    EXPECT_FALSE(compShader.UsesDiscard());
}

TEST(ZeroFlipTranspiler, WarpGeissVoronoiNestedUvFlipsMainOnlyAndPreservesMotionVectorCoords)
{
    using libprojectM::MilkdropPreset::MilkdropShader;

    MilkdropShader::SetCatmullRomEnabled(true);
    const std::string geissWarpShader = R"(
        shader_body
        {
            float2 uv_warped = (uv - 0.5) * (1.0 - bass * 0.08 * tex2D(sampler_noise_lq, uv).x) + 0.5;
            ret = tex2D(sampler_main, uv_warped).xyz * 0.98;
        }
    )";

    MilkdropShader warpShader(MilkdropShader::ShaderType::WarpShader);
    std::string const glsl = warpShader.TranspileToGLSLSource(geissWarpShader);

    size_t const psBodyPos = glsl.find("void PS(");
    ASSERT_NE(psBodyPos, std::string::npos);
    std::string const psBody = glsl.substr(psBodyPos);

    // 1-tap WarpShader with Catmull-Rom enabled must route _pm_texMain_sampler_main through sampleCatmullRom4Tap with 1.0 - _u.y.
    EXPECT_NE(glsl.find("sampleCatmullRom4Tap(sampler_main, vec2(_u.x, 1.0 - _u.y))"), std::string::npos) << glsl;
    EXPECT_NE(psBody.find("_pm_texMain_sampler_main("), std::string::npos) << psBody;
    EXPECT_EQ(psBody.find("texture(sampler_main,"), std::string::npos) << psBody;

    // Noise texture must remain unflipped.
    EXPECT_NE(psBody.find("texture(sampler_noise_lq,"), std::string::npos) << psBody;
    EXPECT_EQ(glsl.find("_pm_texMain_sampler_noise_lq"), std::string::npos) << glsl;

    // Motion vector output attachment (COLOR1) must store unflipped (_uv).xy so MotionVectors.vert works unchanged.
    EXPECT_NE(psBody.find("(_mv_tex_coords).xy = (_uv).xy"), std::string::npos) << psBody;
}

TEST(ZeroFlipTranspiler, SupportsPrefixedAndUppercaseMainSamplersPlusMultiTapAndLod)
{
    using libprojectM::MilkdropPreset::MilkdropShader;

    const std::string multiSamplerShader = R"(
        shader_body
        {
            float3 a = tex2D(sampler_fc_main, uv + float2(0.01, 0.0)).xyz;
            float3 b = tex2D(sampler_pw_main, uv - float2(0.01, 0.0)).xyz;
            float3 c = tex2D(sampler_FC_main, uv + float2(0.0, 0.01)).xyz;
            float3 d = tex2Dlod(sampler_main, float4(uv, 0.0, 1.0)).xyz;
            ret = (a + b + c + d) * 0.25;
        }
    )";

    MilkdropShader compShader(MilkdropShader::ShaderType::CompositeShader);
    std::string const glsl = compShader.TranspileToGLSLSource(multiSamplerShader);

    size_t const psBodyPos = glsl.find("void PS(");
    ASSERT_NE(psBodyPos, std::string::npos);
    std::string const psBody = glsl.substr(psBodyPos);

    EXPECT_NE(psBody.find("_pm_texMain_sampler_fc_main("), std::string::npos) << psBody;
    EXPECT_NE(psBody.find("_pm_texMain_sampler_pw_main("), std::string::npos) << psBody;
    EXPECT_NE(psBody.find("_pm_texMainLod_sampler_main("), std::string::npos) << psBody;
    EXPECT_EQ(psBody.find("texture(sampler_fc_main,"), std::string::npos) << psBody;
    EXPECT_EQ(psBody.find("texture(sampler_pw_main,"), std::string::npos) << psBody;
    EXPECT_EQ(psBody.find("tex2Dlod0(sampler_main,"), std::string::npos) << psBody;
    EXPECT_EQ(psBody.find("sampler_main,"), std::string::npos) << psBody;
}

TEST(ZeroFlipTranspiler, DetectsDiscardFromClipIntrinsicAndDefaultWarpShaderFlipsMainOnly)
{
    using libprojectM::MilkdropPreset::MilkdropShader;
    using libprojectM::MilkdropPreset::MilkdropStaticShaders;

    MilkdropShader clipShader(MilkdropShader::ShaderType::CompositeShader);
    std::string const clipGlsl = clipShader.TranspileToGLSLSource(R"(
        shader_body
        {
            clip(uv.x - 0.1);
            ret = GetMain(uv);
        }
    )");
    EXPECT_TRUE(clipShader.UsesDiscard()) << clipGlsl;

    std::string const defaultWarpFrag = MilkdropStaticShaders::Get()->GetPresetWarpFragmentShader();
    EXPECT_NE(defaultWarpFrag.find("1.0 - frag_TEXCOORD0.y"), std::string::npos) << defaultWarpFrag;
    EXPECT_NE(defaultWarpFrag.find("max(s0, vec2(1e-5))"), std::string::npos) << defaultWarpFrag;
    EXPECT_NE(defaultWarpFrag.find("texCoords = frag_TEXCOORD0.xy;"), std::string::npos) << defaultWarpFrag;
}
