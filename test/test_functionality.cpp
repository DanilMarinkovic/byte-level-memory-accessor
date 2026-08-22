#include <gtest/gtest.h>
#include <limits>
#include <cmath>
#include <random>
#include "byte-level-memory-accessor/scalar.hpp"
#include "byte-level-memory-accessor/simd.hpp"

using namespace bytelevel;


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
    auto compressed {compress_value<P>(original)};
    double result {decompress_value(compressed)};
    EXPECT_NEAR(original, result, std::abs(original) * PrecisionTraits<P>::tolerance);
}

TYPED_TEST(RoundTripTest, NegativeValue)
{
    constexpr std::size_t P {TypeParam::value};
    double original {-42.7};
    auto compressed {compress_value<P>(original)};
    double result {decompress_value(compressed)};
    EXPECT_NEAR(original, result, std::abs(original) * PrecisionTraits<P>::tolerance);
}

TYPED_TEST(RoundTripTest, Zero)
{
    constexpr std::size_t P {TypeParam::value};
    double original {};
    auto compressed {compress_value<P>(original)};
    double result {decompress_value(compressed)};
    EXPECT_DOUBLE_EQ(original, result);
}

TYPED_TEST(RoundTripTest, NegativeZero)
{
    constexpr std::size_t P {TypeParam::value};
    double original {-0.0};
    auto compressed {compress_value<P>(original)};
    double result {decompress_value(compressed)};
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
    auto compressed {compress_value<P>(original)};
    double result {decompress_value(compressed)};
    EXPECT_NEAR(original, result, std::abs(original) * PrecisionTraits<P>::tolerance);
}

TYPED_TEST(RoundTripTest, LargeNegativeValue)
{
    constexpr std::size_t P {TypeParam::value};
    double original {1e-300};
    auto compressed {compress_value<P>(original)};
    double result {decompress_value(compressed)};
    EXPECT_NEAR(original, result, std::abs(original) * PrecisionTraits<P>::tolerance);
}

//Infinity stays infinite
TYPED_TEST(RoundTripTest, PositiveInfinity)
{
    constexpr std::size_t P {TypeParam::value};
    double original {std::numeric_limits<double>::infinity()};
    auto compressed {compress_value<P>(original)};
    double result {decompress_value(compressed)};
    EXPECT_TRUE(std::isinf(result));
    EXPECT_GT(result, 0);
}

//After round trip infinity is still infite and negative
TYPED_TEST(RoundTripTest, NegativeInfinity)
{
    constexpr std::size_t P {TypeParam::value};
    double original {-std::numeric_limits<double>::infinity()};
    auto compressed {compress_value<P>(original)};
    double result {decompress_value(compressed)};
    EXPECT_TRUE(std::isinf(result));
    EXPECT_LT(result, 0);
}

//Compressing and decompressing NaN stays NaN
TYPED_TEST(RoundTripTest, NaN)
{
    constexpr std::size_t P {TypeParam::value};
    double original {std::numeric_limits<double>::quiet_NaN()};
    auto compressed {compress_value<P>(original)};
    double result {decompress_value(compressed)};
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

    compress<P>(original, simd_compressed);
    scalar_compress<P>(original, scalar_compressed);
    for (size_t i{}; i < 4; i++){
        EXPECT_EQ(scalar_compressed[i], simd_compressed[i]);
    }
    
    double scalar_decompressed[4] {};
    double simd_decompressed[4] {};

    decompress(simd_compressed, simd_decompressed);
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

    compress<P>(original, simd_compressed);
    scalar_compress<P>(original, scalar_compressed);
    
    for (size_t i{}; i < 5; i++){
        EXPECT_EQ(scalar_compressed[i], simd_compressed[i]);
    }
    
    double scalar_decompressed[5] {};
    double simd_decompressed[5] {};

    decompress(simd_compressed, simd_decompressed);
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

    compress<P>(original, simd_compressed);
    scalar_compress<P>(original, scalar_compressed);
    
    for (size_t i{}; i < 3; i++){
        EXPECT_EQ(scalar_compressed[i], simd_compressed[i]);
    }
    
    double scalar_decompressed[3] {};
    double simd_decompressed[3] {};

    decompress(simd_compressed, simd_decompressed);
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

    compress<P>(original.data(), simd_compressed.data(), N);
    scalar_compress<P>(original.data(), scalar_compressed.data(), N);
    
    for (size_t i{}; i < N; i++){
        EXPECT_EQ(scalar_compressed[i], simd_compressed[i]);
    }
    
    std::vector<double> scalar_decompressed(N);
    std::vector<double> simd_decompressed(N);

    decompress(simd_compressed.data(), simd_decompressed.data(), N);
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

    compress<P>(original, simd_compressed);
    scalar_compress<P>(original, scalar_compressed);
    
    for (size_t i{}; i < N; i++){
        EXPECT_EQ(scalar_compressed[i], simd_compressed[i]);
    }
    
    std::array<double, N> scalar_decompressed {};
    std::array<double, N> simd_decompressed {};

    decompress(simd_compressed, simd_decompressed);
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
    compress<P>(input.data(), simd_compressed.data(), N);

    for (size_t i{}; i < N; i++)
        EXPECT_EQ(scalar_compressed[i], simd_compressed[i])
            << "compress mismatch at index " << i;

    scalar_decompress(scalar_compressed.data(), scalar_decompressed.data(), N);
    decompress(simd_compressed.data(), simd_decompressed.data(), N);

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

        EXPECT_EQ(decompress_value(compress_value<P>(input[i])), expected)
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
    const split_array<P> from_simd{compress<P>(input)};

    for (size_t i{}; i < N; i++)
    {
        EXPECT_EQ(from_scalar.high[i], from_simd.high[i]) << "high mismatch at index " << i;
        EXPECT_EQ(from_scalar.low[i], from_simd.low[i]) << "low mismatch at index " << i;
    }

    std::vector<double> scalar_out(N), simd_out(N);
    scalar_decompress<P>(from_scalar, scalar_out.data());
    decompress<P>(from_simd, simd_out.data());
    for (size_t i{}; i < N; i++)
        EXPECT_EQ(scalar_out[i], simd_out[i]) << "decompress mismatch at index " << i;
}

//Compares two doubles by their bits, so that a NaN matches a NaN
static bool same_bits(const double left, const double right)
{
    uint64_t left_bits, right_bits;
    std::memcpy(&left_bits, &left, sizeof(left_bits));
    std::memcpy(&right_bits, &right, sizeof(right_bits));
    return left_bits == right_bits;
}

//Indexing a split array reads the same value as decompressing it through the kernels
TYPED_TEST(SplitTest, IndexingMatchesDecompress)
{
    constexpr std::size_t P{TypeParam::value};
    constexpr size_t N{10007};
    std::mt19937_64 rng{1};
    std::uniform_real_distribution<double> dist{-1e10, 1e10};

    std::vector<double> input(N);
    for (auto& value : input) value = dist(rng);

    const split_array<P> compressed{compress<P>(input)};
    std::vector<double> decompressed;
    decompress(compressed, decompressed);

    ASSERT_EQ(compressed.size(), N);
    EXPECT_FALSE(compressed.empty());
    for (size_t i{}; i < N; i++)
        EXPECT_EQ(compressed[i], decompressed[i]) << "index mismatch at index " << i;
}

//Iterating a split array reads the same values as indexing it
TYPED_TEST(SplitTest, IterationMatchesIndexing)
{
    constexpr std::size_t P{TypeParam::value};
    constexpr size_t N{10007};
    std::mt19937_64 rng{1};
    std::uniform_real_distribution<double> dist{-1e10, 1e10};

    std::vector<double> input(N);
    for (auto& value : input) value = dist(rng);

    const split_array<P> compressed{compress<P>(input)};

    const std::vector<double> copied(compressed.begin(), compressed.end());
    ASSERT_EQ(copied.size(), N);
    for (size_t i{}; i < N; i++)
        EXPECT_EQ(copied[i], compressed[i]) << "copy mismatch at index " << i;

    size_t seen{};
    for (const double value : compressed)
    {
        EXPECT_EQ(value, compressed[seen]) << "iteration mismatch at index " << seen;
        seen++;
    }
    EXPECT_EQ(seen, N);
}

//An empty split array holds nothing and iterates over nothing
TYPED_TEST(SplitTest, EmptyArrayIteratesOverNothing)
{
    constexpr std::size_t P{TypeParam::value};
    const split_array<P> compressed{compress<P>(std::vector<double>{})};

    EXPECT_EQ(compressed.size(), 0u);
    EXPECT_TRUE(compressed.empty());
    EXPECT_TRUE(compressed.begin() == compressed.end());
}

//The vector overloads size their output to their input, whether it is fresh or already used
TYPED_TEST(SplitTest, VectorOverloadsSizeTheirOutput)
{
    constexpr std::size_t P{TypeParam::value};
    constexpr size_t N{10007};
    constexpr size_t shorter_size{17};
    std::mt19937_64 rng{1};
    std::uniform_real_distribution<double> dist{-1e10, 1e10};

    std::vector<double> input(N);
    for (auto& value : input) value = dist(rng);

    split_array<P> compressed;
    std::vector<double> decompressed;

    scalar_compress<P>(input, compressed);
    ASSERT_EQ(compressed.size(), N);
    scalar_decompress(compressed, decompressed);
    ASSERT_EQ(decompressed.size(), N);

    const std::vector<double> shorter(input.begin(), input.begin() + shorter_size);
    compress<P>(shorter, compressed);
    ASSERT_EQ(compressed.size(), shorter_size);
    decompress(compressed, decompressed);
    ASSERT_EQ(decompressed.size(), shorter_size);

    for (size_t i{}; i < shorter_size; i++)
        EXPECT_EQ(decompressed[i], compressed[i]) << "reuse mismatch at index " << i;
}

//The AVX2 fallback compresses and decompresses exactly as the scalar path does
TYPED_TEST(SplitTest, AVX2MatchesScalar)
{
    constexpr std::size_t P{TypeParam::value};
    constexpr size_t N{10007};
    std::mt19937_64 rng{1};
    std::uniform_real_distribution<double> dist{-1e10, 1e10};

    std::vector<double> input(N);
    for (auto& value : input) value = dist(rng);

    const split_array<P> from_scalar{scalar_compress<P>(input)};
    split_array<P> from_avx2{make_split_array<P>(N)};
    implementation::compress_avx2<P>(input.data(), from_avx2.high_data(), from_avx2.low_data(), N);

    for (size_t i{}; i < N; i++)
    {
        EXPECT_EQ(from_scalar.high_data()[i], from_avx2.high_data()[i])
            << "high mismatch at index " << i;
        EXPECT_EQ(from_scalar.low_data()[i], from_avx2.low_data()[i])
            << "low mismatch at index " << i;
    }

    std::vector<double> scalar_out(N), avx2_out(N);
    scalar_decompress(from_scalar, scalar_out.data());
    implementation::decompress_avx2<P>(from_avx2.high_data(), from_avx2.low_data(), avx2_out.data(), N);

    for (size_t i{}; i < N; i++)
        EXPECT_EQ(scalar_out[i], avx2_out[i]) << "decompress mismatch at index " << i;
}

//Every tail length of the four wide and eight wide kernels agrees with the scalar path
TYPED_TEST(SplitTest, TailLengthsMatchScalar)
{
    constexpr std::size_t P{TypeParam::value};
    std::mt19937_64 rng{1};
    std::uniform_real_distribution<double> dist{-1e10, 1e10};

    for (size_t N{}; N <= 17; N++)
    {
        std::vector<double> input(N);
        for (auto& value : input) value = dist(rng);

        const split_array<P> from_scalar{scalar_compress<P>(input)};
        const split_array<P> from_simd{compress<P>(input)};
        split_array<P> from_avx2{make_split_array<P>(N)};
        implementation::compress_avx2<P>(input.data(), from_avx2.high_data(), from_avx2.low_data(), N);

        for (size_t i{}; i < N; i++)
        {
            EXPECT_EQ(from_scalar.high_data()[i], from_simd.high_data()[i])
                << "high mismatch at size " << N << " index " << i;
            EXPECT_EQ(from_scalar.low_data()[i], from_simd.low_data()[i])
                << "low mismatch at size " << N << " index " << i;
            EXPECT_EQ(from_scalar.high_data()[i], from_avx2.high_data()[i])
                << "AVX2 high mismatch at size " << N << " index " << i;
            EXPECT_EQ(from_scalar.low_data()[i], from_avx2.low_data()[i])
                << "AVX2 low mismatch at size " << N << " index " << i;
        }
    }
}

//Zeroes, infinities and NaNs come back through the AVX2 fallback bit for bit
TYPED_TEST(SplitTest, AVX2KeepsSpecialValues)
{
    constexpr std::size_t P{TypeParam::value};
    const std::vector<double> input{0.0,
                                    -0.0,
                                    std::numeric_limits<double>::infinity(),
                                    -std::numeric_limits<double>::infinity(),
                                    std::numeric_limits<double>::quiet_NaN(),
                                    std::numeric_limits<double>::denorm_min(),
                                    std::numeric_limits<double>::max(),
                                    std::numeric_limits<double>::lowest(),
                                    std::numeric_limits<double>::epsilon()};
    const size_t N{input.size()};

    split_array<P> compressed{make_split_array<P>(N)};
    implementation::compress_avx2<P>(input.data(), compressed.high_data(), compressed.low_data(), N);

    std::vector<double> output(N);
    implementation::decompress_avx2<P>(compressed.high_data(), compressed.low_data(), output.data(), N);

    for (size_t i{}; i < N; i++)
        EXPECT_TRUE(same_bits(output[i], decompress_value(compress_value<P>(input[i]))))
            << "special value mismatch at index " << i;
}

//A split value carries its precision, so it decompresses without being told it
TYPED_TEST(SplitTest, DecompressValueTakesSplitValue)
{
    constexpr std::size_t P{TypeParam::value};
    constexpr size_t N{1009};
    std::mt19937_64 rng{1};
    std::uniform_real_distribution<double> dist{-1e10, 1e10};

    for (size_t i{}; i < N; i++)
    {
        const double value{dist(rng)};
        const split_value<P> compressed{compress_value<P>(value)};
        EXPECT_EQ(decompress_value(compressed), decompress_value(compressed))
            << "single value mismatch at index " << i;
    }
}

//The high half on its own is the compression of the whole precision it is built from
TYPED_TEST(SplitTest, HighHalfMatchesWholePrecision)
{
    constexpr std::size_t P{TypeParam::value};
    constexpr std::size_t whole{sizeof(typename split_halves<P>::high_t) * 8};
    constexpr size_t N{10007};
    std::mt19937_64 rng{1};
    std::uniform_real_distribution<double> dist{-1e10, 1e10};

    std::vector<double> input(N);
    for (auto& value : input) value = dist(rng);

    const split_array<P> split{compress<P>(input)};
    std::vector<compressed_type_t<whole>> whole_compressed;
    compress<whole>(input, whole_compressed);

    ASSERT_EQ(whole_compressed.size(), N);
    for (size_t i{}; i < N; i++)
        EXPECT_EQ(split.high_data()[i], whole_compressed[i])
            << "high half is not the " << whole << " bit compression at index " << i;
}
