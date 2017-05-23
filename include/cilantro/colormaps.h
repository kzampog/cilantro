#pragma once

#include <vector>
#include <Eigen/Dense>

enum ColormapType {
    COLORMAP_JET,
    COLORMAP_GREY,
    COLORMAP_BLUE2RED
};

std::vector<Eigen::Vector4f> colormap (const std::vector<float> &scalars, const float scalar_min, const float scalar_max, const ColormapType colormapType);
