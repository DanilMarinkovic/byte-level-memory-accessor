// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Danil Marinkovic

//Compressing a vector of doubles to 32 bits and reading it back.

#include <byte-level-memory-accessor/simd.hpp>

#include <cstdio>
#include <vector>

int main()
{
    //! [compress-vector-32]
    const std::vector<double> input{1.5, -2.25, 3.75, 0.1, 1e10, 42.0, 7.125, 9.0};

    std::vector<bytelevel::compressed_type_t<32>> compressed;
    bytelevel::compress<32>(input, compressed);

    std::vector<double> decompressed;
    bytelevel::decompress(compressed, decompressed);
    //! [compress-vector-32]

    std::printf("%-12s %-24s %-24s %s\n", "index", "original", "decompressed", "relative error");
    for (std::size_t i{}; i < input.size(); i++)
    {
        const double error{input[i] == 0.0 ? 0.0
                                           : (input[i] - decompressed[i]) / input[i]};
        std::printf("%-12zu %-24.17g %-24.17g %.3e\n", i, input[i], decompressed[i], error);
    }

    std::printf("\n%zu doubles took %zu bytes, now %zu bytes\n",
                input.size(), input.size() * sizeof(double),
                compressed.size() * sizeof(bytelevel::compressed_type_t<32>));
    return 0;
}
