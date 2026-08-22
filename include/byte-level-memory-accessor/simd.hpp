// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Danil Marinkovic

#pragma once

#include <vector>
#include <array>
#include <immintrin.h>
#include "types.hpp"
#include "scalar.hpp"
namespace bytelevel
{
namespace implementation
{
template<std::size_t precision, typename T>
void compress_avx2(const T* input, compressed_type_t<precision>* output, size_t N)
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
        output[i] = compress_value<precision>(input[i]);
    } 
}
} // namespace implementation

#ifdef __AVX512F__
namespace implementation
{
template<std::size_t precision, typename T>
void compress_avx512(const T* input, compressed_type_t<precision>* output, size_t N)
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
        output[i] = compress_value<precision>(input[i]);
    } 
}
} // namespace implementation
#endif

template<std::size_t precision, typename T>
void compress(const T* input, compressed_type_t<precision>* output, size_t N)
{
#ifdef __AVX512F__
    implementation::compress_avx512<precision>(input, output, N);
#else
    implementation::compress_avx2<precision>(input, output, N);
#endif
}

template<std::size_t precision, typename T, std::size_t N>
void compress(const T (&input)[N], compressed_type_t<precision>* output)
{
    compress<precision>(input, output, N);
}

//Wrapper for std::array
template <std::size_t precision, typename T, std::size_t N>
void compress(const std::array<T, N>& input, std::array<compressed_type_t<precision>, N>& output)
{
    compress<precision>(input.data(), output.data(), N);
}

namespace implementation
{
template <typename T>
void decompress_avx2(const T* input, double* output, size_t N)
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
        output[i] = decompress_value(input[i]);
    }
}
} // namespace implementation

#ifdef __AVX512F__
namespace implementation
{
template <typename T>
register512d decompress_register_avx512(const T* input)
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
} // namespace implementation

template <typename T>
register512d decompress_register(const T* values)
{
    static_assert(is_value_type_v<T>, "Value must be a double, uint32_t or uint16_t.\n");
    if constexpr (sizeof(T) * 8 == 64) return _mm512_loadu_pd(values);
    else return implementation::decompress_register_avx512(values);
}

namespace implementation
{
//Widens eight stored halves into the 64 bit lanes they belong in
template <typename T>
register512i widen_half_avx512(const T* values)
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
void narrow_half_avx512(T* values, const register512i lanes)
{
    if constexpr (sizeof(T) == 4)
        _mm256_storeu_si256(reinterpret_cast<register256i*>(values), _mm512_cvtepi64_epi32(lanes));
    else if constexpr (sizeof(T) == 2)
        _mm_storeu_si128(reinterpret_cast<register128i*>(values), _mm512_cvtepi64_epi16(lanes));
    else
        _mm_storeu_si64(values, _mm512_cvtepi64_epi8(lanes));
}
template <std::size_t precision>
void compress_avx512(const double* input,
                                typename split_halves<precision>::high_t* high,
                                typename split_halves<precision>::low_t* low, size_t N)
{
    static_assert(is_split_v<precision>, "Only split precisions have two halves.\n");

    size_t i{};
    const size_t simd_end{N & ~size_t(7)};
    for (; i < simd_end; i += 8)
    {
        const register512i in{_mm512_loadu_si512(reinterpret_cast<const register512i*>(&input[i]))};
        narrow_half_avx512(&high[i], _mm512_srli_epi64(in, high_shift<precision>));
        narrow_half_avx512(&low[i], _mm512_srli_epi64(in, low_shift<precision>));
    }
    for (; i < N; i++)
    {
        const split_value<precision> compressed{compress_value<precision>(input[i])};
        high[i] = compressed.high;
        low[i] = compressed.low;
    }
}
} // namespace implementation

//Reads eight values from the two halves straight into a register
template <std::size_t precision>
register512d decompress_register(const typename split_halves<precision>::high_t* high,
                                 const typename split_halves<precision>::low_t* low)
{
    static_assert(is_split_v<precision>, "Only split precisions have two halves.\n");
    return _mm512_castsi512_pd(
        _mm512_or_epi64(_mm512_slli_epi64(implementation::widen_half_avx512(high), high_shift<precision>),
                        _mm512_slli_epi64(implementation::widen_half_avx512(low), low_shift<precision>)));
}

namespace implementation
{
template <std::size_t precision>
void decompress_avx512(const typename split_halves<precision>::high_t* high,
                                  const typename split_halves<precision>::low_t* low,
                                  double* output, size_t N)
{
    static_assert(is_split_v<precision>, "Only split precisions have two halves.\n");

    size_t i{};
    const size_t simd_end{N & ~size_t(7)};
    for (; i < simd_end; i += 8)
        _mm512_storeu_pd(&output[i], decompress_register<precision>(&high[i], &low[i]));
    for (; i < N; i++)
        output[i] = decompress_value(split_value<precision>{high[i], low[i]});
}
template <typename T>
void decompress_avx512(const T* input, double* output, size_t N)
{
    size_t i{};

    if constexpr (sizeof(T) * 8 == 32)
    {
        if (N>=8)
        {
            const size_t simd_end {N & ~size_t(7)};
            for(; i < simd_end; i += 8)
            {
                _mm512_storeu_pd(&output[i], decompress_register_avx512(&input[i]));
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
                _mm512_storeu_pd(&output[i], decompress_register_avx512(&input[i]));
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
} // namespace implementation
#endif

template <typename T>
void decompress(const T* input, double* output, size_t N)
{
#ifdef __AVX512F__
    implementation::decompress_avx512(input, output, N);
#else
    implementation::decompress_avx2(input, output, N);
#endif
}

//Wrapper for when N is known at compile time
template <typename T, std::size_t N>
void decompress(const T(&input)[N], double* output)
{
    decompress(input, output, N);
}

//Wrapper for std::array
template <typename T, std::size_t N>
void decompress(const std::array<T, N>& input, std::array<double, N>& output)
{
    decompress(input.data(), output.data(), N);
}

//Vector wrappers, so the vectorised route is reachable without unwrapping to pointers
template <std::size_t precision, typename T>
void compress(const std::vector<T>& input,
                         std::vector<compressed_type_t<precision>>& output)
{
    output.resize(input.size());
    compress<precision>(input.data(), output.data(), input.size());
}

template <typename T>
void decompress(const std::vector<T>& input, std::vector<double>& output)
{
    output.resize(input.size());
    decompress(input.data(), output.data(), input.size());
}


namespace implementation
{
//Widens four stored halves into the 64 bit lanes they belong in
template <typename T>
register256i widen_half_avx2(const T* values)
{
    if constexpr (sizeof(T) == 4)
        return _mm256_cvtepu32_epi64(_mm_loadu_si128(reinterpret_cast<const register128i*>(values)));
    else if constexpr (sizeof(T) == 2)
        return _mm256_cvtepu16_epi64(_mm_loadl_epi64(reinterpret_cast<const register128i*>(values)));
    else
    {
        int32_t packed;
        std::memcpy(&packed, values, sizeof(packed));
        return _mm256_cvtepu8_epi64(_mm_cvtsi32_si128(packed));
    }
}
//Narrows four 64 bit lanes back down and stores them as halves
template <typename T>
void narrow_half_avx2(T* values, const register256i lanes)
{
    //AVX2 has no truncating down convert, so the low half of every lane is gathered by hand
    const register256i idx{_mm256_set_epi32(0, 0, 0, 0, 6, 4, 2, 0)};
    const register128i gathered{_mm256_castsi256_si128(_mm256_permutevar8x32_epi32(lanes, idx))};

    if constexpr (sizeof(T) == 4)
        _mm_storeu_si128(reinterpret_cast<register128i*>(values), gathered);
    else if constexpr (sizeof(T) == 2)
    {
        //The shift leaves the bits above the half in place, and packing saturates rather than
        //truncating, so those bits are masked off first
        const register128i masked{_mm_and_si128(gathered, _mm_set1_epi32(0xFFFF))};
        _mm_storeu_si64(values, _mm_packus_epi32(masked, _mm_setzero_si128()));
    }
    else
    {
        const register128i masked{_mm_and_si128(gathered, _mm_set1_epi32(0xFF))};
        const register128i shorts{_mm_packus_epi32(masked, _mm_setzero_si128())};
        const int32_t packed{_mm_cvtsi128_si32(_mm_packus_epi16(shorts, _mm_setzero_si128()))};
        std::memcpy(values, &packed, sizeof(packed));
    }
}

//Reads four values from the two halves straight into a register
template <std::size_t precision>
register256d decompress_register_avx2(const typename split_halves<precision>::high_t* high,
                                      const typename split_halves<precision>::low_t* low)
{
    static_assert(is_split_v<precision>, "Only split precisions have two halves.\n");
    return _mm256_castsi256_pd(
        _mm256_or_si256(_mm256_slli_epi64(widen_half_avx2(high), high_shift<precision>),
                        _mm256_slli_epi64(widen_half_avx2(low), low_shift<precision>)));
}

//Four doubles per iteration: shift each half into place, then narrow and store it.
//Measured slower than the loop the compiler writes, because AVX2 has no two source permute
//and so each store is only half width. Kept as the vectorised reference for this path.
template <std::size_t precision>
void compress_avx2(const double* input, typename split_halves<precision>::high_t* high,
                              typename split_halves<precision>::low_t* low, size_t N)
{
    static_assert(is_split_v<precision>, "Only split precisions have two halves.\n");

    size_t i{};
    const size_t simd_end{N & ~size_t(3)};
    for (; i < simd_end; i += 4)
    {
        const register256i in{_mm256_loadu_si256(reinterpret_cast<const register256i*>(&input[i]))};
        narrow_half_avx2(&high[i], _mm256_srli_epi64(in, high_shift<precision>));
        narrow_half_avx2(&low[i], _mm256_srli_epi64(in, low_shift<precision>));
    }
    for (; i < N; i++)
    {
        const split_value<precision> compressed{compress_value<precision>(input[i])};
        high[i] = compressed.high;
        low[i] = compressed.low;
    }
}

template <std::size_t precision>
void decompress_avx2(const typename split_halves<precision>::high_t* high,
                                const typename split_halves<precision>::low_t* low, double* output,
                                size_t N)
{
    static_assert(is_split_v<precision>, "Only split precisions have two halves.\n");

    size_t i{};
    const size_t simd_end{N & ~size_t(3)};
    for (; i < simd_end; i += 4)
        _mm256_storeu_pd(&output[i], decompress_register_avx2<precision>(&high[i], &low[i]));
    for (; i < N; i++)
        output[i] = decompress_value(split_value<precision>{high[i], low[i]});
}
} // namespace implementation

//A split precision is kept as two arrays, so it takes an output for each half
template <std::size_t precision>
void compress(const double* input, typename split_halves<precision>::high_t* high,
                         typename split_halves<precision>::low_t* low, size_t N)
{
#ifdef __AVX512F__
    implementation::compress_avx512<precision>(input, high, low, N);
#else
    implementation::compress_avx2<precision>(input, high, low, N);
#endif
}

template <std::size_t precision>
void decompress(const typename split_halves<precision>::high_t* high,
                           const typename split_halves<precision>::low_t* low, double* output,
                           size_t N)
{
#ifdef __AVX512F__
    implementation::decompress_avx512<precision>(high, low, output, N);
#else
    implementation::decompress_avx2<precision>(high, low, output, N);
#endif
}

template <std::size_t precision>
void compress(const double* input, split_array<precision>& output)
{
    compress<precision>(input, output.high_data(), output.low_data(), output.size());
}

template <std::size_t precision>
void decompress(const split_array<precision>& input, double* output)
{
    decompress<precision>(input.high_data(), input.low_data(), output, input.size());
}

template <std::size_t precision>
split_array<precision> compress(const double* input, size_t N)
{
    static_assert(is_split_v<precision>, "Only split precisions have two halves.\n");
    split_array<precision> output{make_split_array<precision>(N)};
    compress<precision>(input, output);
    return output;
}

template <std::size_t precision>
split_array<precision> compress(const std::vector<double>& input)
{
    static_assert(is_split_v<precision>, "Only split precisions have two halves.\n");
    return compress<precision>(input.data(), input.size());
}

//Sizes its output to its input, matching how the whole precisions take a vector
template <std::size_t precision>
void compress(const std::vector<double>& input, split_array<precision>& output)
{
    static_assert(is_split_v<precision>, "Only split precisions have two halves.\n");
    if (output.size() != input.size()) output = make_split_array<precision>(input.size());
    compress<precision>(input.data(), output);
}

template <std::size_t precision>
void decompress(const split_array<precision>& input, std::vector<double>& output)
{
    static_assert(is_split_v<precision>, "Only split precisions have two halves.\n");
    output.resize(input.size());
    decompress<precision>(input, output.data());
}

} // namespace bytelevel
