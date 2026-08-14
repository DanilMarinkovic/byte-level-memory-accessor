#pragma once

#include <memory>
#include <vector>
#include <immintrin.h>
#include "types.hpp"
#include "scalar.hpp"

template<std::size_t precision, typename T, std::size_t N>
std::unique_ptr<compressed_type_t<precision>[]> intrinsics_compress(const T (&input)[N])
{
    static_assert(precision == 16 || precision == 32,
        "Precision must be 16 or 32 bits.");
    static_assert(sizeof(T) == 8,
        "Code uses IEE754 doubles.");

    auto output {std::make_unique<compressed_type_t<precision>[]>(N)};
    
    size_t i{};
    if constexpr (N>=4)
    {
        for(; i < N - 4; i+= 4)
        {
            //Loads the bits from the numbers into a 256 wide register
            register256i in {_mm256_loadu_si256(reinterpret_cast<const register256i*>(&input[i]))};
            //Shifts each 64 bit lane right by 32 bits, shifting in zeroes
            register256i shifted {_mm256_srli_epi64(in, (sizeof(T) * 8) - precision)};
            //Holds indices that will be used to place compressed values contiguously, arguments are ordered highest lane to lowest
            register256i idx {_mm256_set_epi32(0, 0, 0, 0, 6, 4, 2, 0)};
            //Places our compressed values in the right lanes, lanes 0-3 hold truncated 32 bit values other lanes are irrelevant and discarded
            register256i compressed {_mm256_permutevar8x32_epi32(shifted, idx)};
            //Extract only values from lanes 0-3 and store them in output based on precision
            if constexpr (precision == 32)
            {
                _mm_storeu_si128(reinterpret_cast<register128i*>(&output[i]), _mm256_castsi256_si128(compressed));
            }
            else if constexpr (precision == 16)
            {
                register128i cast_128 {_mm256_castsi256_si128(compressed)};
                _mm_storeu_si64(reinterpret_cast<uint64_t*>(&output[i]), _mm_packus_epi32(cast_128, _mm_setzero_si128()));
            }

        }  
    }
    for(; i < N; i++)
    {
        output[i] = template_compress<precision>(input[i]);
    } 
    return output;
}

template<std::size_t precision, typename T>
void intrinsics_compress_avx2(const T* input, compressed_type_t<precision>* output, size_t N)
{
    static_assert(precision == 16 || precision == 32,
        "Precision must be 16 or 32 bits.");
    static_assert(sizeof(T) == 8,
        "Code uses IEE754 doubles.");
    
    size_t i{};
    if (N>=4)
    {
        const size_t simd_end {N & ~size_t(3)};
        for(; i < simd_end; i+= 4)
        {
            //Loads the bits from the numbers into a 256 wide register
            register256i in {_mm256_loadu_si256(reinterpret_cast<const register256i*>(&input[i]))};
            //Shifts each 64 bit lane right by 32 bits, shifting in zeroes
            register256i shifted {_mm256_srli_epi64(in, (sizeof(T) * 8) - precision)};
            //Holds indices that will be used to place compressed values contiguously, arguments are ordered highest lane to lowest
            register256i idx {_mm256_set_epi32(0, 0, 0, 0, 6, 4, 2, 0)};
            //Places our compressed values in the right lanes, lanes 0-3 hold truncated 32 bit values other lanes are irrelevant and discarded
            register256i compressed {_mm256_permutevar8x32_epi32(shifted, idx)};
            //Extract only values from lanes 0-3 and store them in output based on precision
            if constexpr (precision == 32)
            {
                _mm_storeu_si128(reinterpret_cast<register128i*>(&output[i]), _mm256_castsi256_si128(compressed));
            }
            else if constexpr (precision == 16)
            {
                register128i cast_128 {_mm256_castsi256_si128(compressed)};
                _mm_storeu_si64(reinterpret_cast<uint64_t*>(&output[i]), _mm_packus_epi32(cast_128, _mm_setzero_si128()));
            }
        }  
    }
    for(; i < N; i++)
    {
        output[i] = template_compress<precision>(input[i]);
    } 
}

#ifdef __AVX512F__
template<std::size_t precision, typename T>
void intrinsics_compress_avx512(const T* input, compressed_type_t<precision>* output, size_t N)
{
    static_assert(precision == 16 || precision == 32,
        "Precision must be 16 or 32 bits.");
    static_assert(sizeof(T) == 8,
        "Code uses IEE754 doubles.");
    
    size_t i{};
    if (N>=8)
    {
        const size_t simd_end {N & ~size_t(7)};
        for(; i < simd_end; i+= 8)
        {
            //Loads the bits from the numbers into a 5123 wide register
            register512i in {_mm512_loadu_si512(reinterpret_cast<const register512i*>(&input[i]))};
            //Shifts each 64 bit lane right shifting in zeroes
            register512i shifted {_mm512_srli_epi64(in, (sizeof(T) * 8) - precision)};
            //Holds indices that will be used to place compressed values contiguously, arguments are ordered highest lane to lowest
            register512i idx {_mm512_set_epi32(0,0,0,0,0,0,0,0,14,12,10,8,6,4,2,0)};
            //Places our compressed values in the right lanes, lanes 0-3 hold truncated 32 bit values other lanes are irrelevant and discarded
            register512i compressed {_mm512_permutexvar_epi32(idx, shifted)};
            //Extract only values from lanes 0-3 and store them in output based on precision
            if constexpr (precision == 32)
            {
                _mm256_storeu_si256(reinterpret_cast<register256i*>(&output[i]), _mm512_castsi512_si256(compressed));
            }
            else if constexpr (precision == 16)
            {
                register256i cast_256 {_mm512_castsi512_si256(compressed)};
                _mm_storeu_si128(reinterpret_cast<register128i*>(&output[i]), _mm256_cvtepi32_epi16(cast_256));
            }
        }  
    }
    for(; i < N; i++)
    {
        output[i] = template_compress<precision>(input[i]);
    } 
}
#endif

template<std::size_t precision, typename T>
void intrinsics_compress(const T* input, compressed_type_t<precision>* output, size_t N)
{
#ifdef __AVX512F__
    intrinsics_compress_avx512<precision>(input, output, N);
#else
    intrinsics_compress_avx2<precision>(input, output, N);
#endif
}

template<std::size_t precision, typename T, std::size_t N>
void intrinsics_compress(const T (&input)[N], compressed_type_t<precision>* output)
{
    intrinsics_compress<precision>(input, output, N);
}

//Wrapper for std::array
template <std::size_t precision, typename T, std::size_t N>
void intrinsics_compress(const std::array<T, N>& input, std::array<compressed_type_t<precision>, N>& output)
{
    intrinsics_compress<precision>(input.data(), output.data(), N);
}

template <typename T, std::size_t N>
std::unique_ptr<double[]> intrinsics_decompress(const T (&input)[N])
{
    auto output {std::make_unique<double[]>(N)};

    register128i in;
    register256i wide;

    if constexpr (sizeof(T) * 8 == 32)
    {
        in = _mm_loadu_si128(reinterpret_cast<const register128i*>(input));
        wide = _mm256_cvtepu32_epi64(in);
    }
    else if constexpr (sizeof(T) * 8 == 16)
    {
        in = _mm_loadl_epi64(reinterpret_cast<const register128i*>(input));
        wide = _mm256_cvtepu16_epi64(in);
    }

    register256i result {_mm256_slli_epi64(wide, 64 - sizeof(T) * 8)};
    _mm256_storeu_si256(reinterpret_cast<register256i*>(output.get()), result);

    return output;
}

template <typename T>
void intrinsics_decompress_avx2(const T* input, double* output, size_t N)
{
    register128i in;
    register256i wide;
    size_t i{};
    
    if (N>=4)
    {
        const size_t simd_end {N & ~size_t(3)};
        if constexpr (sizeof(T) * 8 == 32)
        {
            for(; i < simd_end; i += 4)
            {
                in = _mm_loadu_si128(reinterpret_cast<const register128i*>(&input[i]));
                wide = _mm256_cvtepu32_epi64(in);
                register256i result {_mm256_slli_epi64(wide, 64 - sizeof(T) * 8)};
                _mm256_storeu_si256(reinterpret_cast<register256i*>(&output[i]), result);
            }
        }
        else if constexpr (sizeof(T) * 8 == 16)
        {
                for(; i < simd_end; i += 4)
                {
                in = _mm_loadl_epi64(reinterpret_cast<const register128i*>(&input[i]));
                wide = _mm256_cvtepu16_epi64(in);
                register256i result {_mm256_slli_epi64(wide, 64 - sizeof(T) * 8)};
                _mm256_storeu_si256(reinterpret_cast<register256i*>(&output[i]), result);
                }
        }
    }

    for(; i < N; i++)
    {
        output[i] = template_decompress(input[i]);
    }
}

#ifdef __AVX512F__
template <typename T>
register512d intrinsics_decompress_register(const T* input)
{
    if constexpr (sizeof(T) * 8 == 32)
    {
        const register512i wide {_mm512_cvtepu32_epi64(_mm256_loadu_si256(reinterpret_cast<const register256i*>(input)))};
        return _mm512_castsi512_pd(_mm512_slli_epi64(wide, 64 - sizeof(T) * 8));
    }
    else if constexpr (sizeof(T) * 8 == 16)
    {
        const register512i wide {_mm512_cvtepu16_epi64(_mm_loadu_si128(reinterpret_cast<const register128i*>(input)))};
        return _mm512_castsi512_pd(_mm512_slli_epi64(wide, 64 - sizeof(T) * 8));
    }
}

template <typename T>
register512d decompress_register(const T* values)
{
    static_assert(is_value_type_v<T>, "Value must be a double, uint32_t or uint16_t.\n");
    if constexpr (sizeof(T) * 8 == 64) return _mm512_loadu_pd(values);
    else return intrinsics_decompress_register(values);
}

//Widens eight stored halves into the 64 bit lanes they belong in
template <typename T>
register512i widen_half(const T* values)
{
    if constexpr (sizeof(T) == 4)
        return _mm512_cvtepu32_epi64(_mm256_loadu_si256(reinterpret_cast<const register256i*>(values)));
    else if constexpr (sizeof(T) == 2)
        return _mm512_cvtepu16_epi64(_mm_loadu_si128(reinterpret_cast<const register128i*>(values)));
    else
        return _mm512_cvtepu8_epi64(_mm_loadl_epi64(reinterpret_cast<const register128i*>(values)));
}

//Narrows eight 64 bit lanes back down and stores them as halves
template <typename T>
void narrow_half(T* values, const register512i lanes)
{
    if constexpr (sizeof(T) == 4)
        _mm256_storeu_si256(reinterpret_cast<register256i*>(values), _mm512_cvtepi64_epi32(lanes));
    else if constexpr (sizeof(T) == 2)
        _mm_storeu_si128(reinterpret_cast<register128i*>(values), _mm512_cvtepi64_epi16(lanes));
    else
        _mm_storeu_si64(values, _mm512_cvtepi64_epi8(lanes));
}

template <std::size_t precision>
void intrinsics_compress(const double* input, typename split_halves<precision>::high_t* high,
                         typename split_halves<precision>::low_t* low, size_t N)
{
    static_assert(is_split_v<precision>, "Only split precisions have two halves.\n");

    size_t i{};
    const size_t simd_end{N & ~size_t(7)};
    for (; i < simd_end; i += 8)
    {
        const register512i in{_mm512_loadu_si512(reinterpret_cast<const register512i*>(&input[i]))};
        narrow_half(&high[i], _mm512_srli_epi64(in, high_shift<precision>));
        narrow_half(&low[i], _mm512_srli_epi64(in, low_shift<precision>));
    }
    for (; i < N; i++)
    {
        const split_value<precision> compressed{template_compress<precision>(input[i])};
        high[i] = compressed.high;
        low[i] = compressed.low;
    }
}

//Reads eight values from the two halves straight into a register
template <std::size_t precision>
register512d decompress_register(const typename split_halves<precision>::high_t* high,
                                 const typename split_halves<precision>::low_t* low)
{
    static_assert(is_split_v<precision>, "Only split precisions have two halves.\n");
    return _mm512_castsi512_pd(
        _mm512_or_epi64(_mm512_slli_epi64(widen_half(high), high_shift<precision>),
                        _mm512_slli_epi64(widen_half(low), low_shift<precision>)));
}

template <std::size_t precision>
void intrinsics_decompress(const typename split_halves<precision>::high_t* high,
                           const typename split_halves<precision>::low_t* low, double* output,
                           size_t N)
{
    static_assert(is_split_v<precision>, "Only split precisions have two halves.\n");

    size_t i{};
    const size_t simd_end{N & ~size_t(7)};
    for (; i < simd_end; i += 8)
        _mm512_storeu_pd(&output[i], decompress_register<precision>(&high[i], &low[i]));
    for (; i < N; i++)
        output[i] = template_decompress(split_value<precision>{high[i], low[i]});
}

template <std::size_t precision>
void intrinsics_compress(const double* input, split_array<precision>& output)
{
    intrinsics_compress<precision>(input, output.high.get(), output.low.get(), output.size);
}

template <std::size_t precision>
void intrinsics_decompress(const split_array<precision>& input, double* output)
{
    intrinsics_decompress<precision>(input.high.get(), input.low.get(), output, input.size);
}

template <std::size_t precision>
split_array<precision> intrinsics_compress(const double* input, size_t N)
{
    static_assert(is_split_v<precision>, "Only split precisions have two halves.\n");
    split_array<precision> output{make_split_array<precision>(N)};
    intrinsics_compress<precision>(input, output);
    return output;
}

template <std::size_t precision>
split_array<precision> intrinsics_compress(const std::vector<double>& input)
{
    static_assert(is_split_v<precision>, "Only split precisions have two halves.\n");
    return intrinsics_compress<precision>(input.data(), input.size());
}

template <typename T>
void intrinsics_decompress_avx512(const T* input, double* output, size_t N)
{
    size_t i{};

    if constexpr (sizeof(T) * 8 == 32)
    {
        if (N>=8)
        {
            const size_t simd_end {N & ~size_t(7)};
            for(; i < simd_end; i += 8)
            {
                _mm512_storeu_pd(&output[i], intrinsics_decompress_register(&input[i]));
            }
        }

        const size_t remaining {N - i};
        if (remaining > 0)
        {
            const __mmask8 mask {static_cast<__mmask8>((1u << remaining) - 1)};
            register256i in {_mm256_maskz_loadu_epi32(mask, &input[i])};
            register512i wide {_mm512_cvtepu32_epi64(in)};
            register512i result {_mm512_slli_epi64(wide, 64 - sizeof(T) * 8)};
            _mm512_mask_storeu_pd(&output[i], mask, _mm512_castsi512_pd(result));
        }
    }
    else if constexpr (sizeof(T) * 8 == 16)
    {
        if (N>=8)
        {
            const size_t simd_end {N & ~size_t(7)};
            for(; i < simd_end; i += 8)
            {
                _mm512_storeu_pd(&output[i], intrinsics_decompress_register(&input[i]));
            }
        }

        const size_t remaining {N - i};
        if (remaining > 0)
        {
            const __mmask8 mask {static_cast<__mmask8>((1u << remaining) - 1)};
            register128i in {_mm_maskz_loadu_epi16(mask, &input[i])};
            register512i wide {_mm512_cvtepu16_epi64(in)};
            register512i result {_mm512_slli_epi64(wide, 64 - sizeof(T) * 8)};
            _mm512_mask_storeu_pd(&output[i], mask, _mm512_castsi512_pd(result));
        }
    }
}
#endif

template <typename T>
void intrinsics_decompress(const T* input, double* output, size_t N)
{
#ifdef __AVX512F__
    intrinsics_decompress_avx512(input, output, N);
#else
    intrinsics_decompress_avx2(input, output, N);
#endif
}

//Wrapper for when N is known at compile time
template <typename T, std::size_t N>
void intrinsics_decompress(const T(&input)[N], double* output)
{
    intrinsics_decompress(input, output, N);
}

//Wrapper for std::array
template <typename T, std::size_t N>
void intrinsics_decompress(const std::array<T, N>& input, std::array<double, N>& output)
{
    intrinsics_decompress(input.data(), output.data(), N);
}

