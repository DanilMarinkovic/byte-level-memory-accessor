#include <gtest/gtest.h>
#include <limits>
#include <cmath>
#include <random>
#include "byte-level-memory-accessor/scalar.hpp"
#include "byte-level-memory-accessor/simd.hpp"


template <std::size_t precision>
struct PrecisionTraits
{
    static constexpr double tolerance{2.0 / static_cast<double>(1ull << (precision - 12))};
};

template <typename T>
class RoundTripTest : public testing::Test {};

using AllPrecisions = testing::Types<
    std::integral_constant<std::size_t, 16>,
    std::integral_constant<std::size_t, 24>,
    std::integral_constant<std::size_t, 32>,
    std::integral_constant<std::size_t, 48>
>;

TYPED_TEST_SUITE(RoundTripTest, AllPrecisions);

//Tests different values across all precisions
TYPED_TEST(RoundTripTest, PositiveValue)
{
    constexpr std::size_t P{TypeParam::value};
    double original {3.14159};
    auto compressed {template_compress<P>(original)};
    double result {template_decompress(compressed)};
    EXPECT_NEAR(original, result, std::abs(original) * PrecisionTraits<P>::tolerance);
}

TYPED_TEST(RoundTripTest, NegativeValue)
{
    constexpr std::size_t P {TypeParam::value};
    double original {-42.7};
    auto compressed {template_compress<P>(original)};
    double result {template_decompress(compressed)};
    EXPECT_NEAR(original, result, std::abs(original) * PrecisionTraits<P>::tolerance);
}

TYPED_TEST(RoundTripTest, Zero)
{
    constexpr std::size_t P {TypeParam::value};
    double original {};
    auto compressed {template_compress<P>(original)};
    double result {template_decompress(compressed)};
    EXPECT_DOUBLE_EQ(original, result);
}

TYPED_TEST(RoundTripTest, NegativeZero)
{
    constexpr std::size_t P {TypeParam::value};
    double original {-0.0};
    auto compressed {template_compress<P>(original)};
    double result {template_decompress(compressed)};
    //Ensures sign bit is preserved
    uint64_t original_bits{}, result_bits{};
    std::memcpy(&original_bits, &original, sizeof(double));
    std::memcpy(&result_bits, &result, sizeof(double));
    EXPECT_EQ(original_bits, result_bits);
}

TYPED_TEST(RoundTripTest, LargePositiveValue)
{
    constexpr std::size_t P {TypeParam::value};
    double original {1e300};
    auto compressed {template_compress<P>(original)};
    double result {template_decompress(compressed)};
    EXPECT_NEAR(original, result, std::abs(original) * PrecisionTraits<P>::tolerance);
}

TYPED_TEST(RoundTripTest, LargeNegativeValue)
{
    constexpr std::size_t P {TypeParam::value};
    double original {1e-300};
    auto compressed {template_compress<P>(original)};
    double result {template_decompress(compressed)};
    EXPECT_NEAR(original, result, std::abs(original) * PrecisionTraits<P>::tolerance);
}

//Infinity stays infinite
TYPED_TEST(RoundTripTest, PositiveInfinity)
{
    constexpr std::size_t P {TypeParam::value};
    double original {std::numeric_limits<double>::infinity()};
    auto compressed {template_compress<P>(original)};
    double result {template_decompress(compressed)};
    EXPECT_TRUE(std::isinf(result));
    EXPECT_GT(result, 0);
}

//After round trip infinity is still infite and negative
TYPED_TEST(RoundTripTest, NegativeInfinity)
{
    constexpr std::size_t P {TypeParam::value};
    double original {-std::numeric_limits<double>::infinity()};
    auto compressed {template_compress<P>(original)};
    double result {template_decompress(compressed)};
    EXPECT_TRUE(std::isinf(result));
    EXPECT_LT(result, 0);
}

//Compressing and decompressing NaN stays NaN
TYPED_TEST(RoundTripTest, NaN)
{
    constexpr std::size_t P {TypeParam::value};
    double original {std::numeric_limits<double>::quiet_NaN()};
    auto compressed {template_compress<P>(original)};
    double result {template_decompress(compressed)};
    EXPECT_TRUE(std::isnan(result));
}

template <typename T>
class ArrayTest : public testing::Test {};

using PrecisionTypes = testing::Types<
    std::integral_constant<std::size_t, 32>,
    std::integral_constant<std::size_t, 16>
>;

TYPED_TEST_SUITE(ArrayTest, PrecisionTypes);

//Compressing and decompressing C-array with scalar produces same output as SIMD for array of 4 values
TYPED_TEST(ArrayTest, SIMDMatchesScalarCArray)
{
    constexpr std::size_t P {TypeParam::value};
    constexpr double original[4] {1.2,-1.3,1.4,-1.5};
    compressed_type_t<P> scalar_compressed[4] {};
    compressed_type_t<P> simd_compressed[4] {};

    intrinsics_compress<P>(original, simd_compressed);
    scalar_compress<P>(original, scalar_compressed);
    for (size_t i{}; i < 4; i++){
        EXPECT_EQ(scalar_compressed[i], simd_compressed[i]);
    }
    
    double scalar_decompressed[4] {};
    double simd_decompressed[4] {};

    intrinsics_decompress(simd_compressed, simd_decompressed);
    scalar_decompress(scalar_compressed, scalar_decompressed);

    for (size_t i{}; i < 4; i++){
        EXPECT_EQ(scalar_decompressed[i], simd_decompressed[i]);
    }
}

//Compressing and decompressing C-array with scalar produces same output as SIMD for array of more than 4 values
TYPED_TEST(ArrayTest, SIMDMatchesScalarCArrayOfFive)
{
    constexpr std::size_t P {TypeParam::value};
    constexpr double original[5] {1.2,-1.3,1.4,-1.5,1.6};
    compressed_type_t<P> scalar_compressed[5] {};
    compressed_type_t<P> simd_compressed[5] {};

    intrinsics_compress<P>(original, simd_compressed);
    scalar_compress<P>(original, scalar_compressed);
    
    for (size_t i{}; i < 5; i++){
        EXPECT_EQ(scalar_compressed[i], simd_compressed[i]);
    }
    
    double scalar_decompressed[5] {};
    double simd_decompressed[5] {};

    intrinsics_decompress(simd_compressed, simd_decompressed);
    scalar_decompress(scalar_compressed, scalar_decompressed);

    for (size_t i{}; i < 5; i++){
        EXPECT_EQ(scalar_decompressed[i], simd_decompressed[i]);
    }
}

//Compressing and decompressing C-array with scalar produces same output as SIMD for array of less than 4 values
TYPED_TEST(ArrayTest, SIMDMatchesScalarCArrayOfThree)
{
    constexpr std::size_t P {TypeParam::value};
    constexpr double original[3] {1.2,-1.3,1.4};
    compressed_type_t<P> scalar_compressed[3] {};
    compressed_type_t<P> simd_compressed[3] {};

    intrinsics_compress<P>(original, simd_compressed);
    scalar_compress<P>(original, scalar_compressed);
    
    for (size_t i{}; i < 3; i++){
        EXPECT_EQ(scalar_compressed[i], simd_compressed[i]);
    }
    
    double scalar_decompressed[3] {};
    double simd_decompressed[3] {};

    intrinsics_decompress(simd_compressed, simd_decompressed);
    scalar_decompress(scalar_compressed, scalar_decompressed);

    for (size_t i{}; i < 3; i++){
        EXPECT_EQ(scalar_decompressed[i], simd_decompressed[i]);
    }
}

//Compressing and decompressing vector with scalar produces same output as SIMD for array of 4 values
TYPED_TEST(ArrayTest, SIMDMatchesScalarVector)
{
    constexpr std::size_t P {TypeParam::value};
    std::vector<double> original {1.2,-1.3,1.4,-1.5};
    const size_t N {original.size()};

    std::vector<compressed_type_t<P>> scalar_compressed(N);
    std::vector<compressed_type_t<P>> simd_compressed(N);

    intrinsics_compress<P>(original.data(), simd_compressed.data(), N);
    scalar_compress<P>(original.data(), scalar_compressed.data(), N);
    
    for (size_t i{}; i < N; i++){
        EXPECT_EQ(scalar_compressed[i], simd_compressed[i]);
    }
    
    std::vector<double> scalar_decompressed(N);
    std::vector<double> simd_decompressed(N);

    intrinsics_decompress(simd_compressed.data(), simd_decompressed.data(), N);
    scalar_decompress(scalar_compressed.data(), scalar_decompressed.data(), N);

    for (size_t i{}; i < N; i++){
        EXPECT_EQ(scalar_decompressed[i], simd_decompressed[i]);
    }
}

//Compressing and decompressing array with scalar produces same output as SIMD for array of 4 values
TYPED_TEST(ArrayTest, SIMDMatchesScalarArray)
{
    constexpr std::size_t P {TypeParam::value};
    constexpr size_t N {4};
    std::array<double, N> original {1.2,-1.3,1.4,-1.5};

    std::array<compressed_type_t<P>, N> scalar_compressed {};
    std::array<compressed_type_t<P>, N> simd_compressed {};

    intrinsics_compress<P>(original, simd_compressed);
    scalar_compress<P>(original, scalar_compressed);
    
    for (size_t i{}; i < N; i++){
        EXPECT_EQ(scalar_compressed[i], simd_compressed[i]);
    }
    
    std::array<double, N> scalar_decompressed {};
    std::array<double, N> simd_decompressed {};

    intrinsics_decompress(simd_compressed, simd_decompressed);
    scalar_decompress(scalar_compressed, scalar_decompressed);

    for (size_t i{}; i < N; i++){
        EXPECT_EQ(scalar_decompressed[i], simd_decompressed[i]);
    }
}

TYPED_TEST(ArrayTest, LargeRandomArraySIMDMatchesScalar)
{
    constexpr std::size_t P {TypeParam::value};
    constexpr size_t N {10007};

    std::mt19937_64 rng {1};
    std::uniform_real_distribution<double> dist {-1e10, 1e10};

    std::vector<double> input(N);
    for (auto& x : input) x = dist(rng);

    std::vector<compressed_type_t<P>> scalar_compressed(N);
    std::vector<compressed_type_t<P>> simd_compressed(N);
    std::vector<double> scalar_decompressed(N);
    std::vector<double> simd_decompressed(N);

    scalar_compress<P>(input.data(), scalar_compressed.data(), N);
    intrinsics_compress<P>(input.data(), simd_compressed.data(), N);

    for (size_t i{}; i < N; i++)
        EXPECT_EQ(scalar_compressed[i], simd_compressed[i])
            << "compress mismatch at index " << i;

    scalar_decompress(scalar_compressed.data(), scalar_decompressed.data(), N);
    intrinsics_decompress(simd_compressed.data(), simd_decompressed.data(), N);

    for (size_t i{}; i < N; i++)
        EXPECT_EQ(scalar_decompressed[i], simd_decompressed[i])
            << "decompress mismatch at index " << i;
}

template <typename T>
class SplitTest : public testing::Test
{};

using SplitPrecisions = testing::Types<std::integral_constant<std::size_t, 24>,
                                       std::integral_constant<std::size_t, 48>>;

TYPED_TEST_SUITE(SplitTest, SplitPrecisions);

//Split precision must truncate exactly as keeping its leading bits does
TYPED_TEST(SplitTest, MatchesTruncation)
{
    constexpr std::size_t P{TypeParam::value};
    constexpr size_t N{10007};
    std::mt19937_64 rng{1};
    std::uniform_real_distribution<double> dist{-1e10, 1e10};

    std::vector<double> input(N);
    for (auto& value : input) value = dist(rng);

    for (size_t i{}; i < N; i++)
    {
        uint64_t bits{};
        std::memcpy(&bits, &input[i], sizeof(double));
        const uint64_t kept{bits & (~uint64_t{} << (64 - P))};
        double expected;
        std::memcpy(&expected, &kept, sizeof(double));

        EXPECT_EQ(template_decompress(template_compress<P>(input[i])), expected)
            << "at index " << i;
    }
}
TYPED_TEST(SplitTest, SIMDMatchesScalar)
{
    constexpr std::size_t P{TypeParam::value};
    constexpr size_t N{10007};
    std::mt19937_64 rng{1};
    std::uniform_real_distribution<double> dist{-1e10, 1e10};

    std::vector<double> input(N);
    for (auto& value : input) value = dist(rng);

    const split_array<P> from_scalar{scalar_compress<P>(input)};
    const split_array<P> from_simd{intrinsics_compress<P>(input)};

    for (size_t i{}; i < N; i++)
    {
        EXPECT_EQ(from_scalar.high[i], from_simd.high[i]) << "high mismatch at index " << i;
        EXPECT_EQ(from_scalar.low[i], from_simd.low[i]) << "low mismatch at index " << i;
    }

    std::vector<double> scalar_out(N), simd_out(N);
    scalar_decompress<P>(from_scalar, scalar_out.data());
    intrinsics_decompress<P>(from_simd, simd_out.data());
    for (size_t i{}; i < N; i++)
        EXPECT_EQ(scalar_out[i], simd_out[i]) << "decompress mismatch at index " << i;
}
