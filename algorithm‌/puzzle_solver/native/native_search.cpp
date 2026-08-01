#include "native_search.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace {

constexpr std::int32_t kAbiVersion = 1;

std::int32_t popcount64(std::uint64_t value) {
#if defined(_MSC_VER)
    return static_cast<std::int32_t>(__popcnt64(value));
#else
    return static_cast<std::int32_t>(__builtin_popcountll(value));
#endif
}

struct CandidateRef {
    std::int32_t local_index;
    std::int32_t global_index;
    std::int32_t overlap;
};

struct Solution {
    std::int32_t holes;
    std::int32_t overlap;
    std::int64_t score_key;
    std::vector<std::int32_t> candidate_indices;
};

class Search {
public:
    Search(
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
        std::int32_t max_solutions)
        : masks_(masks),
          piece_offsets_(piece_offsets),
          cell_counts_(cell_counts),
          piece_count_(piece_count),
          word_count_(word_count),
          rectangle_cells_(rectangle_cells),
          allowed_holes_(allowed_holes),
          allowed_overlap_(allowed_overlap),
          max_nodes_(max_nodes),
          max_seconds_(max_seconds),
          max_solutions_(max_solutions),
          started_(Clock::now()),
          chosen_(static_cast<std::size_t>(piece_count), -1) {}

    void run() {
        std::vector<std::uint64_t> occupied(static_cast<std::size_t>(word_count_), 0);
        std::vector<std::int32_t> remaining(static_cast<std::size_t>(piece_count_));
        std::iota(remaining.begin(), remaining.end(), 0);
        dfs(remaining, occupied, 0);
        std::stable_sort(solutions_.begin(), solutions_.end(), solution_less);
    }

    std::int64_t nodes() const { return nodes_; }
    bool timed_out() const { return timed_out_; }
    const std::vector<Solution>& solutions() const { return solutions_; }

private:
    using Clock = std::chrono::steady_clock;

    const std::uint64_t* mask(std::int32_t global_index) const {
        return masks_ + static_cast<std::size_t>(global_index) * word_count_;
    }

    std::int32_t intersection_count(
        const std::vector<std::uint64_t>& occupied,
        std::int32_t global_index) const {
        const auto* candidate = mask(global_index);
        std::int32_t count = 0;
        for (std::int32_t word = 0; word < word_count_; ++word) {
            count += popcount64(occupied[static_cast<std::size_t>(word)] & candidate[word]);
        }
        return count;
    }

    static std::int32_t bit_count(const std::vector<std::uint64_t>& words) {
        std::int32_t count = 0;
        for (const auto word : words) {
            count += popcount64(word);
        }
        return count;
    }

    bool deadline_reached() const {
        if (max_seconds_ <= 0.0) {
            return false;
        }
        const std::chrono::duration<double> elapsed = Clock::now() - started_;
        return elapsed.count() >= max_seconds_;
    }

    static bool solution_less(const Solution& first, const Solution& second) {
        return first.score_key < second.score_key;
    }

    void save_solution(std::int32_t holes, std::int32_t overlap) {
        Solution solution{
            holes,
            overlap,
            static_cast<std::int64_t>(holes) + 5LL * overlap,
            chosen_,
        };
        solutions_.push_back(std::move(solution));
        std::stable_sort(solutions_.begin(), solutions_.end(), solution_less);
        if (static_cast<std::int32_t>(solutions_.size()) > max_solutions_) {
            solutions_.resize(static_cast<std::size_t>(max_solutions_));
        }
    }

    void dfs(
        const std::vector<std::int32_t>& remaining,
        const std::vector<std::uint64_t>& occupied,
        std::int32_t overlap_cells) {
        if (timed_out_ || nodes_ >= max_nodes_) {
            return;
        }
        if ((nodes_ & 255LL) == 0 && deadline_reached()) {
            timed_out_ = true;
            return;
        }
        ++nodes_;

        if (remaining.empty()) {
            const auto holes = rectangle_cells_ - bit_count(occupied);
            if (holes <= allowed_holes_ && overlap_cells <= allowed_overlap_) {
                save_solution(holes, overlap_cells);
            }
            return;
        }

        std::vector<std::vector<CandidateRef>> viable_by_piece(
            static_cast<std::size_t>(piece_count_));
        std::vector<std::uint64_t> possible_coverage = occupied;
        std::int32_t maximum_new_cells = 0;
        std::int32_t selected_piece = -1;
        std::size_t selected_count = std::numeric_limits<std::size_t>::max();

        for (const auto piece : remaining) {
            auto& viable = viable_by_piece[static_cast<std::size_t>(piece)];
            std::vector<std::uint64_t> piece_union(static_cast<std::size_t>(word_count_), 0);
            std::int32_t piece_maximum = 0;
            const auto start = piece_offsets_[piece];
            const auto end = piece_offsets_[piece + 1];
            viable.reserve(static_cast<std::size_t>(end - start));
            for (auto global = start; global < end; ++global) {
                const auto overlap = intersection_count(occupied, global);
                if (overlap_cells + overlap > allowed_overlap_) {
                    continue;
                }
                viable.push_back(CandidateRef{global - start, global, overlap});
                const auto* candidate = mask(global);
                for (std::int32_t word = 0; word < word_count_; ++word) {
                    piece_union[static_cast<std::size_t>(word)] |= candidate[word];
                }
                piece_maximum = std::max(piece_maximum, cell_counts_[global] - overlap);
            }
            if (viable.empty()) {
                return;
            }
            for (std::int32_t word = 0; word < word_count_; ++word) {
                possible_coverage[static_cast<std::size_t>(word)] |=
                    piece_union[static_cast<std::size_t>(word)];
            }
            maximum_new_cells += piece_maximum;
            if (viable.size() < selected_count) {
                selected_piece = piece;
                selected_count = viable.size();
            }
        }

        if (rectangle_cells_ - bit_count(possible_coverage) > allowed_holes_) {
            return;
        }
        if (rectangle_cells_ - (bit_count(occupied) + maximum_new_cells) > allowed_holes_) {
            return;
        }

        auto& selected = viable_by_piece[static_cast<std::size_t>(selected_piece)];
        std::stable_sort(
            selected.begin(),
            selected.end(),
            [this](const CandidateRef& first, const CandidateRef& second) {
                if (first.overlap != second.overlap) {
                    return first.overlap < second.overlap;
                }
                return cell_counts_[first.global_index] > cell_counts_[second.global_index];
            });

        std::vector<std::int32_t> next_remaining;
        next_remaining.reserve(remaining.size() - 1);
        for (const auto piece : remaining) {
            if (piece != selected_piece) {
                next_remaining.push_back(piece);
            }
        }

        for (const auto& candidate_ref : selected) {
            if (timed_out_) {
                return;
            }
            std::vector<std::uint64_t> next_occupied = occupied;
            const auto* candidate = mask(candidate_ref.global_index);
            for (std::int32_t word = 0; word < word_count_; ++word) {
                next_occupied[static_cast<std::size_t>(word)] |= candidate[word];
            }
            chosen_[static_cast<std::size_t>(selected_piece)] = candidate_ref.local_index;
            dfs(next_remaining, next_occupied, overlap_cells + candidate_ref.overlap);
            chosen_[static_cast<std::size_t>(selected_piece)] = -1;
        }
    }

    const std::uint64_t* masks_;
    const std::int32_t* piece_offsets_;
    const std::int32_t* cell_counts_;
    std::int32_t piece_count_;
    std::int32_t word_count_;
    std::int32_t rectangle_cells_;
    std::int32_t allowed_holes_;
    std::int32_t allowed_overlap_;
    std::int64_t max_nodes_;
    double max_seconds_;
    std::int32_t max_solutions_;
    Clock::time_point started_;
    std::vector<std::int32_t> chosen_;
    std::vector<Solution> solutions_;
    std::int64_t nodes_ = 0;
    bool timed_out_ = false;
};

bool valid_request(
    const std::uint64_t* masks,
    const std::int32_t* piece_offsets,
    const std::int32_t* cell_counts,
    std::int32_t piece_count,
    std::int32_t word_count,
    std::int32_t rectangle_cells,
    std::int32_t allowed_holes,
    std::int32_t allowed_overlap,
    std::int64_t max_nodes,
    std::int32_t max_solutions,
    const std::int32_t* output_solution_count,
    const std::int64_t* output_nodes,
    const std::int32_t* output_timed_out,
    const std::int32_t* output_candidate_indices,
    const std::int32_t* output_hole_cells,
    const std::int32_t* output_overlap_cells) {
    if (masks == nullptr || piece_offsets == nullptr || cell_counts == nullptr ||
        output_solution_count == nullptr || output_nodes == nullptr ||
        output_timed_out == nullptr || output_candidate_indices == nullptr ||
        output_hole_cells == nullptr || output_overlap_cells == nullptr) {
        return false;
    }
    if (piece_count < 1 || piece_count > 4 || word_count < 1 || rectangle_cells < 1 ||
        allowed_holes < 0 || allowed_overlap < 0 || max_nodes < 1 || max_solutions < 1) {
        return false;
    }
    if (piece_offsets[0] != 0) {
        return false;
    }
    for (std::int32_t piece = 0; piece < piece_count; ++piece) {
        if (piece_offsets[piece + 1] <= piece_offsets[piece]) {
            return false;
        }
    }
    return true;
}

}  // namespace

extern "C" PUZZLE_NATIVE_API std::int32_t puzzle_search_abi_version() {
    return kAbiVersion;
}

extern "C" PUZZLE_NATIVE_API std::int32_t puzzle_search(
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
    std::int32_t* output_overlap_cells) {
    if (!valid_request(
            masks,
            piece_offsets,
            cell_counts,
            piece_count,
            word_count,
            rectangle_cells,
            allowed_holes,
            allowed_overlap,
            max_nodes,
            max_solutions,
            output_solution_count,
            output_nodes,
            output_timed_out,
            output_candidate_indices,
            output_hole_cells,
            output_overlap_cells)) {
        return -1;
    }

    try {
        Search search(
            masks,
            piece_offsets,
            cell_counts,
            piece_count,
            word_count,
            rectangle_cells,
            allowed_holes,
            allowed_overlap,
            max_nodes,
            max_seconds,
            max_solutions);
        search.run();
        const auto& solutions = search.solutions();
        *output_solution_count = static_cast<std::int32_t>(solutions.size());
        *output_nodes = search.nodes();
        *output_timed_out = search.timed_out() ? 1 : 0;
        for (std::size_t solution_index = 0; solution_index < solutions.size(); ++solution_index) {
            const auto& solution = solutions[solution_index];
            output_hole_cells[solution_index] = solution.holes;
            output_overlap_cells[solution_index] = solution.overlap;
            for (std::int32_t piece = 0; piece < piece_count; ++piece) {
                output_candidate_indices[solution_index * piece_count + piece] =
                    solution.candidate_indices[static_cast<std::size_t>(piece)];
            }
        }
        return 0;
    } catch (const std::exception&) {
        return -2;
    } catch (...) {
        return -3;
    }
}
