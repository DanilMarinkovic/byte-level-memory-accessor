// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Danil Marinkovic

//48 bit values are held as two arrays, but a bytelevel::split_array reads back as one array of doubles.

#include <byte-level-memory-accessor/simd.hpp>

#include <cstdio>
#include <vector>

int main()
{
    //! [compress-vector-48]
    const std::vector<double> input{1.5, -2.25, 3.75, 0.1, 1e10, 42.0, 7.125, 9.0};

    bytelevel::split_array<48> compressed;
    bytelevel::compress<48>(input, compressed);

    const double fourth{compressed[3]};

    std::vector<double> decompressed;
    bytelevel::decompress(compressed, decompressed);
    //! [compress-vector-48]

    std::printf("compressed.size() = %zu, compressed[3] = %.17g\n\n", compressed.size(), fourth);

    //! [iterate-48]
    for (const double value : compressed) std::printf("  %.17g\n", value);
    //! [iterate-48]

    std::printf("\n%zu doubles took %zu bytes, now %zu bytes across two arrays\n",
                input.size(), input.size() * sizeof(double),
                compressed.size() * (sizeof(uint32_t) + sizeof(uint16_t)));

    for (std::size_t i{}; i < input.size(); i++)
        if (decompressed[i] != compressed[i]) std::printf("mismatch at %zu\n", i);
    return 0;
}
