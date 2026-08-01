#pragma once

#include <cstdint>

#if defined(_WIN32)
#if defined(PUZZLE_NATIVE_BUILD)
#define PUZZLE_NATIVE_API __declspec(dllexport)
#else
#define PUZZLE_NATIVE_API __declspec(dllimport)
#endif
#else
#define PUZZLE_NATIVE_API __attribute__((visibility("default")))
#endif

extern "C" {

PUZZLE_NATIVE_API std::int32_t puzzle_search_abi_version();

// Candidate masks are stored candidate-major, with word_count uint64 words per
// candidate. piece_offsets contains piece_count + 1 cumulative candidate counts.
PUZZLE_NATIVE_API std::int32_t puzzle_search(
    const std::uint64_t* masks,
    const std::int32_t* piece_offsets,
    const std::int32_t* cell_counts,
    std::int32_t piece_count,
    std::int32_t word_count,
    std::int32_t rectangle_cells,
    std::int32_t allowed_holes,
    std::int32_t allowed_overlap,
    std::int64_t max_nodes,
    double max_seconds,
    std::int32_t max_solutions,
    std::int32_t* output_solution_count,
    std::int64_t* output_nodes,
    std::int32_t* output_timed_out,
    std::int32_t* output_candidate_indices,
    std::int32_t* output_hole_cells,
    std::int32_t* output_overlap_cells);

}
