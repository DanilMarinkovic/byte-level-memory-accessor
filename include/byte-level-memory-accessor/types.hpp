#pragma once

#include <immintrin.h>
#include <cstdint>
#include <type_traits>
#include <memory>
#include <cstddef>
#include <cstring>
#include <iterator>

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

//The halves recombine into the value, the bits below them reading as zero
template <std::size_t precision>
double template_decompress(const split_value<precision> value)
{
    const uint64_t bits{(static_cast<uint64_t>(value.high) << high_shift<precision>)
                        | (static_cast<uint64_t>(value.low) << low_shift<precision>)};
    double result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

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

//Holds the two halves as separate arrays, but reads back as a single array of values
template <std::size_t precision>
struct split_array
{
    using high_t = typename split_halves<precision>::high_t;
    using low_t = typename split_halves<precision>::low_t;
    using value_type = double;

    //Recombining two halves builds a value, so there is no stored double to refer to
    class const_iterator
    {
        const split_array* array{};
        std::size_t index{};

    public:
        using iterator_category = std::input_iterator_tag;
        using value_type = double;
        using difference_type = std::ptrdiff_t;
        using pointer = void;
        using reference = double;

        const_iterator() = default;
        const_iterator(const split_array* source, const std::size_t position)
            : array{source}, index{position}
        {
        }

        double operator*() const { return (*array)[index]; }
        const_iterator& operator++()
        {
            index++;
            return *this;
        }
        const_iterator operator++(int)
        {
            const_iterator previous{*this};
            index++;
            return previous;
        }
        bool operator==(const const_iterator& other) const { return index == other.index; }
        bool operator!=(const const_iterator& other) const { return index != other.index; }
    };

    std::unique_ptr<high_t[]> high;
    std::unique_ptr<low_t[]> low;
    std::size_t count{};

    std::size_t size() const { return count; }
    bool empty() const { return count == 0; }

    double operator[](const std::size_t index) const
    {
        return template_decompress(split_value<precision>{high[index], low[index]});
    }

    //The halves stay reachable, so a lower precision can read the high array on its own
    high_t* high_data() { return high.get(); }
    const high_t* high_data() const { return high.get(); }
    low_t* low_data() { return low.get(); }
    const low_t* low_data() const { return low.get(); }

    const_iterator begin() const { return {this, 0}; }
    const_iterator end() const { return {this, count}; }
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

