#pragma once

#include <iostream>
#include <bitset>
#include <cstdint>
#include <iomanip>
#include <cstring>
#include <vector>

#include "types.hpp"

template <std::size_t precision, typename T>
auto template_compress(T value)
{
    static_assert(precision == 16 || precision == 32 || is_split_v<precision>,
    "Precision must be 16, 32 or a split precision.\n");
    uint64_t bits;
    std::memcpy(&bits, &value, sizeof(value));
    if constexpr (is_split_v<precision>)
    {
        using result_t = split_value<precision>;
        return result_t{static_cast<typename result_t::high_t>(bits >> high_shift<precision>),
                        static_cast<typename result_t::low_t>(bits >> low_shift<precision>)};
    }
    else
    {
        bits >>= (sizeof(T) * 8 - precision);
        if constexpr (precision == 16) return static_cast<uint16_t>(bits);
        else if constexpr (precision == 32) return static_cast<uint32_t>(bits);
    }
}

template <std::size_t precision>
void scalar_compress(const double* input, compressed_type_t<precision>* output, size_t N)
{
    for (size_t i = 0; i < N; i++)
        output[i] = template_compress<precision>(input[i]);
}

template <std::size_t precision, std::size_t N>
void scalar_compress(const double (&input)[N], compressed_type_t<precision>* output)
{
    scalar_compress<precision>(input, output, N);
}

template < std::size_t precision, std::size_t N>
void scalar_compress(const std::array<double, N>& input, std::array<compressed_type_t<precision>, N>& output)
{
    scalar_compress<precision>(input.data(), output.data(), N);
}

template <std::size_t precision>
void scalar_compress(const std::vector<double>& input, std::vector<compressed_type_t<precision>>& output)
{
    output.resize(input.size());
    scalar_compress<precision>(input.data(), output.data(), input.size());
}

template <typename T>
double template_decompress(T value)
{
    size_t shift = 64 - sizeof(T) * 8;
    uint64_t bits {static_cast<uint64_t>(value) << shift};
    double d;
    std::memcpy(&d, &bits, sizeof(d));
    return d;
}

template <typename T>
double decompress_scalar(const T value)
{
    static_assert(is_value_type_v<T>, "Value must be a double, uint32_t or uint16_t.\n");
    if constexpr (sizeof(T) * 8 == 64) return value;
    else                               return template_decompress(value);
}

//A split value carries its precision in its type, so it decompresses without being told
template <std::size_t precision>
double decompress_scalar(const split_value<precision> value)
{
    return template_decompress(value);
}

//A split precision is kept as two arrays, so it takes an output for each half
template <std::size_t precision>
void scalar_compress(const double* input, typename split_halves<precision>::high_t* high,
                     typename split_halves<precision>::low_t* low, size_t N)
{
    static_assert(is_split_v<precision>, "Only split precisions have two halves.\n");
    for (size_t i{}; i < N; i++)
    {
        const split_value<precision> compressed{template_compress<precision>(input[i])};
        high[i] = compressed.high;
        low[i] = compressed.low;
    }
}

template <std::size_t precision>
void scalar_decompress(const typename split_halves<precision>::high_t* high,
                       const typename split_halves<precision>::low_t* low, double* output, size_t N)
{
    static_assert(is_split_v<precision>, "Only split precisions have two halves.\n");
    for (size_t i{}; i < N; i++)
        output[i] = template_decompress(split_value<precision>{high[i], low[i]});
}

template <std::size_t precision>
void scalar_compress(const double* input, split_array<precision>& output)
{
    static_assert(is_split_v<precision>, "Only split precisions have two halves.\n");
    scalar_compress<precision>(input, output.high_data(), output.low_data(), output.size());
}

template <std::size_t precision>
void scalar_decompress(const split_array<precision>& input, double* output)
{
    static_assert(is_split_v<precision>, "Only split precisions have two halves.\n");
    scalar_decompress<precision>(input.high_data(), input.low_data(), output, input.size());
}

template <std::size_t precision>
split_array<precision> scalar_compress(const double* input, size_t N)
{
    static_assert(is_split_v<precision>, "Only split precisions have two halves.\n");
    split_array<precision> output{make_split_array<precision>(N)};
    scalar_compress<precision>(input, output);
    return output;
}

template <std::size_t precision>
split_array<precision> scalar_compress(const std::vector<double>& input)
{
    static_assert(is_split_v<precision>, "Only split precisions have two halves.\n");
    return scalar_compress<precision>(input.data(), input.size());
}

//Sizes its output to its input, matching how the whole precisions take a vector
template <std::size_t precision>
void scalar_compress(const std::vector<double>& input, split_array<precision>& output)
{
    static_assert(is_split_v<precision>, "Only split precisions have two halves.\n");
    if (output.size() != input.size()) output = make_split_array<precision>(input.size());
    scalar_compress<precision>(input.data(), output);
}

template <std::size_t precision>
void scalar_decompress(const split_array<precision>& input, std::vector<double>& output)
{
    static_assert(is_split_v<precision>, "Only split precisions have two halves.\n");
    output.resize(input.size());
    scalar_decompress<precision>(input, output.data());
}

template <typename T>
void scalar_decompress(const T* input, double* output, size_t N)
{
    for (size_t i{}; i < N; i++)
        output[i] = template_decompress(input[i]);
}

template <typename T, std::size_t N>
void scalar_decompress(const T (&input)[N], double* output)
{
    scalar_decompress(input, output, N);
}

template <typename T, std::size_t N>
void scalar_decompress(const std::array<T, N>& input, std::array<double, N>& output)
{
    scalar_decompress(input.data(), output.data(), N);
}

template <typename T>
void scalar_decompress(const std::vector<T>& input, std::vector<double>& output)
{
    output.resize(input.size());
    scalar_decompress(input.data(), output.data(), input.size());
}



template<std::size_t precision>
void template_roundTrip(double d)
{
    uint64_t original_bits;
    std::memcpy(&original_bits, &d, sizeof(original_bits));
    
    auto compressed {template_compress<precision>(d)};
    
    auto decompressed {template_decompress(compressed)};
    uint64_t decompressed_bits;
    std::memcpy(&decompressed_bits, &decompressed, sizeof(decompressed_bits));

    std::cout << "Original: " << std::setprecision(20) << d << " bits: " << std::bitset<64>{original_bits} << "\n";
    std::cout << "Compressed bits: " << std::bitset<precision>{compressed} << "\n";
    std::cout << std::setprecision(20) << "Decompressed: " << decompressed << " decompressed bits: " 
    << std::bitset<64> {decompressed_bits} << "\n";
}
