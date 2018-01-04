#pragma once

#include <cilantro/data_containers.hpp>

namespace cilantro {
    enum struct ColormapType {JET, GRAY, BLUE2RED};

    PointMatrix<float,3> colormap(const ConstDataMatrixMap<float,1> &scalars, float scalar_min, float scalar_max, const ColormapType &colormap_type);
}
