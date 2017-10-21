#pragma once

#include <vector>
#include <Eigen/Dense>

namespace cilantro {
    enum struct ColormapType {NONE, JET, GREY, BLUE2RED};

    std::vector<Eigen::Vector3f> colormap(const std::vector<float> &scalars, float scalar_min, float scalar_max, ColormapType colormapType);
}
