#pragma once

#include <iterator>
#include <random>

namespace cilantro {
    // Copied from https://gist.github.com/cbsmith/5538174
    template <typename RandomGeneratorT = std::mt19937>
    struct RandomElementSelector {
        typedef RandomGeneratorT RandomGenerator;

        // On most platforms, you probably want to use
        // std::random_device("/dev/urandom")()
        RandomElementSelector(RandomGenerator g = RandomGenerator(std::random_device{}()))
                : gen_(g)
        {}

        template <typename Iter>
        Iter select(Iter start, Iter end) {
            std::uniform_int_distribution<size_t> dis(0, std::distance(start, end) - 1);
            std::advance(start, dis(gen_));
            return start;
        }

        template <typename Iter>
        Iter operator()(Iter start, Iter end) {
            return select(start, end);
        }

        // Convenience function that works on anything with a sensible begin() and
        // end(), and returns with a ref to the value type
        template <typename Container>
        auto operator()(const Container& c) -> decltype(*begin(c))& {
            return *select(begin(c), end(c));
        }

    private:
        RandomGenerator gen_;
    };
}  // namespace cilantro
