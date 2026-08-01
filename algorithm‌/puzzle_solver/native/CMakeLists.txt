cmake_minimum_required(VERSION 3.16)

project(puzzle_solver_native LANGUAGES CXX)

add_library(puzzle_solver_native SHARED native_search.cpp)
target_compile_features(puzzle_solver_native PRIVATE cxx_std_17)
target_compile_definitions(puzzle_solver_native PRIVATE PUZZLE_NATIVE_BUILD)

if(MSVC)
    target_compile_options(puzzle_solver_native PRIVATE /O2 /W4 /EHsc)
else()
    target_compile_options(puzzle_solver_native PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()

set_target_properties(
    puzzle_solver_native
    PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES
)
