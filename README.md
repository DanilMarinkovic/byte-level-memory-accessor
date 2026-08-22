# Byte Level Memory Accessor

Truncation-based compression of IEEE-754 doubles with scalar and AVX-512 kernels.

Values are compressed to 16, 24, 32 or 48 bits by keeping the leading bits of the
double's representation and discarding the rest. Documentation is generated
with Doxygen; see `docs/`.

## Requirements

- A C++17 compiler
- An x86-64 CPU with AVX-512F, AVX-512DQ, AVX-512BW and AVX-512VL

Linking `bytelevel::bytelevel` enables those instruction sets, so binaries built against it
require an AVX-512 capable CPU.

## Using the library

The library is header-only, so adding `include/` to your include path is enough. The CMake package is the supported route.

After installing:

```cmake
find_package(bytelevel 1.0 REQUIRED)
target_link_libraries(your_app PRIVATE bytelevel::bytelevel)
```

Or vendored directly:

```cmake
add_subdirectory(byte-level-memory-accessor)
target_link_libraries(your_app PRIVATE bytelevel::bytelevel)
```

Or fetched at configure time:

```cmake
include(FetchContent)
FetchContent_Declare(bytelevel
    GIT_REPOSITORY https://github.com/<user>/byte-level-memory-accessor.git
    GIT_TAG        v1.0.0)
FetchContent_MakeAvailable(bytelevel)
target_link_libraries(your_app PRIVATE bytelevel::bytelevel)
```

Include whichever header you need:

```cpp
#include <byte-level-memory-accessor/scalar.hpp>   // scalar kernels
#include <byte-level-memory-accessor/simd.hpp>     // vectorised kernels, pulls in scalar.hpp
```

## Building

```sh
cmake -B build
cmake --build build
ctest --test-dir build
```

| Option | Effect |
| --- | --- |
| `BYTELEVEL_BUILD_TESTS` | Builds the GoogleTest suite |
| `BYTELEVEL_BUILD_EXAMPLES` | Builds the programs in `examples/` |
| `BYTELEVEL_INSTALL` | Generates the install and export targets |

Installing:

```sh
cmake -B build -DCMAKE_INSTALL_PREFIX=/your/prefix
cmake --build build --target install
```

## License

Released under the MIT License. See [LICENSE](LICENSE).
