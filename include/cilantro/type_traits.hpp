#pragma once

#include <type_traits>

namespace cilantro {
    template <typename T, typename = int>
    struct HasPoints : std::false_type {};

    template <typename T>
    struct HasPoints<T, decltype((void) T::points, 0)> : std::true_type {};

    template <typename T, typename = int>
    struct HasNormals : std::false_type {};

    template <typename T>
    struct HasNormals<T, decltype((void) T::normals, 0)> : std::true_type {};

    template <typename T, typename = int>
    struct HasColors : std::false_type {};

    template <typename T>
    struct HasColors<T, decltype((void) T::colors, 0)> : std::true_type {};
}
