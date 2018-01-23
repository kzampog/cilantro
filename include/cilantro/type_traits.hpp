#pragma once

#include <type_traits>

template <typename T, typename = int>
struct has_points : std::false_type {};

template <typename T>
struct has_points <T, decltype((void) T::points, 0)> : std::true_type {};

template <typename T, typename = int>
struct has_normals : std::false_type {};

template <typename T>
struct has_normals <T, decltype((void) T::normals, 0)> : std::true_type {};

template <typename T, typename = int>
struct has_colors : std::false_type {};

template <typename T>
struct has_colors <T, decltype((void) T::colors, 0)> : std::true_type {};
