#pragma once

#include <cilantro/data_containers.hpp>

namespace cilantro {
    enum struct ColormapType {JET, GRAY, BLUE2RED};

    VectorSet<float,3> colormap(const ConstVectorSetMatrixMap<float,1> &scalars,
                                const ColormapType &colormap_type = ColormapType::JET,
                                float scalar_min = std::numeric_limits<float>::quiet_NaN(),
                                float scalar_max = std::numeric_limits<float>::quiet_NaN());
}
