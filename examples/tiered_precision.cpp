// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Danil Marinkovic

//The high half of a split array is exactly the compression of the whole precision it is
//built from, so a coarser pass can read it alone and ignore the low half entirely.

#include <byte-level-memory-accessor/simd.hpp>

#include <cstdio>
#include <vector>

int main()
{
    const std::vector<double> input{1.5, -2.25, 3.75, 0.1, 1e10, 42.0, 7.125, 9.0};

    //! [tiering]
    //A 48 bit array carries a 32 bit array inside its high half
    const bytelevel::split_array<48> bits48{bytelevel::compress<48>(input)};

    std::vector<bytelevel::compressed_type_t<32>> bits32;
    bytelevel::compress<32>(input, bits32);

    //bits48.high_data() and bits32 hold the same bits, so either can be read on its own
    bool identical{true};
    for (std::size_t i{}; i < input.size(); i++)
        if (bits48.high_data()[i] != bits32[i]) identical = false;
    //! [tiering]

    std::printf("high half matches the 32 bit compression: %s\n\n", identical ? "yes" : "no");

    std::printf("%-24s %-24s %s\n", "original", "48 bit", "32 bit (high half only)");
    for (std::size_t i{}; i < input.size(); i++)
        std::printf("%-24.17g %-24.17g %.17g\n", input[i], bits48[i], bytelevel::decompress_value(bits32[i]));
    return 0;
}
