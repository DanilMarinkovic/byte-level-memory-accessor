cmake_minimum_required(VERSION 3.21)

project(byte-level-memory-accessor
    VERSION 1.0.0
    DESCRIPTION "Truncation-based compression of IEEE-754 doubles with scalar and AVX-512 kernels"
    LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

add_library(bytelevel INTERFACE)
add_library(bytelevel::bytelevel ALIAS bytelevel)

target_compile_features(bytelevel INTERFACE cxx_std_17)

target_include_directories(bytelevel INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_compile_options(bytelevel INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-mavx2 -mavx512f -mavx512dq -mavx512bw -mavx512vl>)

option(BYTELEVEL_BUILD_TESTS "Build byte level memory accessor tests" ${PROJECT_IS_TOP_LEVEL})
option(BYTELEVEL_INSTALL "Generate install target for bytelevel" ${PROJECT_IS_TOP_LEVEL})

if(BYTELEVEL_BUILD_TESTS)
    include(FetchContent)
    set(INSTALL_GTEST OFF)
    FetchContent_Declare(googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip)
    FetchContent_MakeAvailable(googletest)

    enable_testing()
    include(GoogleTest)

    add_executable(test_functionality test/test_functionality.cpp)
    target_link_libraries(test_functionality PRIVATE bytelevel GTest::gtest_main)
    target_compile_options(test_functionality PRIVATE -O3)
    gtest_discover_tests(test_functionality)
endif()

if(BYTELEVEL_INSTALL)
    install(DIRECTORY include/byte-level-memory-accessor
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

    install(TARGETS bytelevel EXPORT bytelevel-targets)

    install(EXPORT bytelevel-targets
        FILE bytelevel-targets.cmake
        NAMESPACE bytelevel::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/bytelevel)

    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/bytelevel-config-version.cmake"
        COMPATIBILITY SameMajorVersion
        ARCH_INDEPENDENT)

    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/bytelevel-config.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/bytelevel-config.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/bytelevel)

    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/bytelevel-config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/bytelevel-config-version.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/bytelevel)
endif()
