#pragma once

#include <immintrin.h>
#include <cstdint>
#include <type_traits>
#include <memory>
#include <cstddef>

using register256i = __m256i;
using register256d = __m256d;
using register128i = __m128i;

using register512i = __m512i;
using register512d = __m512d;

template <std::size_t precision>
struct compressed_type;

template <>
struct compressed_type<16> 
{   
    using type = uint16_t; 
};

template <>
struct compressed_type<32> 
{ 
    using type = uint32_t; 
};

template <std::size_t precision>
struct split_halves;

template <>
struct split_halves<24>
{
    using high_t = uint16_t;
    using low_t = uint8_t;
};

template <>
struct split_halves<48>
{
    using high_t = uint32_t;
    using low_t = uint16_t;
};

template <std::size_t precision, typename = void>
inline constexpr bool is_split_v{false};

template <std::size_t precision>
inline constexpr bool is_split_v<precision,
                                 std::void_t<typename split_halves<precision>::high_t>>{true};

template <std::size_t precision>
struct split_value
{
    using high_t = typename split_halves<precision>::high_t;
    using low_t = typename split_halves<precision>::low_t;
    static_assert((sizeof(high_t) + sizeof(low_t)) * 8 == precision,
                  "The halves must add up to the precision.\n");

    high_t high{};
    low_t low{};
};

//Bits each half occupies within the value, counting from the least significant end
template <std::size_t precision>
inline constexpr std::size_t high_shift{64 - sizeof(typename split_halves<precision>::high_t) * 8};

template <std::size_t precision>
inline constexpr std::size_t low_shift{64 - precision};

template <>
struct compressed_type<24>
{
    using type = split_value<24>;
};

template <>
struct compressed_type<48>
{
    using type = split_value<48>;
};

template <std::size_t precision>
struct split_array
{
    std::unique_ptr<typename split_halves<precision>::high_t[]> high;
    std::unique_ptr<typename split_halves<precision>::low_t[]> low;
    std::size_t size{};
};

template <std::size_t precision>
inline split_array<precision> make_split_array(const std::size_t size)
{
    return {std::make_unique<typename split_halves<precision>::high_t[]>(size),
            std::make_unique<typename split_halves<precision>::low_t[]>(size), size};
}

template <std::size_t precision>
using compressed_type_t = typename compressed_type<precision>::type;

template <typename T>
inline constexpr bool is_value_type_v{std::is_same_v<T, double>
                                      || std::is_same_v<T, compressed_type_t<32>>
                                      || std::is_same_v<T, compressed_type_t<16>>};

