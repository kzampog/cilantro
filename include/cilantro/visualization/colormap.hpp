#pragma once

#include <cilantro/core/data_containers.hpp>

namespace cilantro {
    enum struct ColormapType { JET, GRAY, BLUE2RED };

    template <typename ValueT>
    VectorSet<float,3> colormap(const ConstVectorSetMatrixMap<ValueT,1> &scalars,
                                const ColormapType &colormap_type = ColormapType::JET,
                                ValueT scalar_min = std::numeric_limits<ValueT>::quiet_NaN(),
                                ValueT scalar_max = std::numeric_limits<ValueT>::quiet_NaN())
    {
        ValueT scalar_min_used = scalar_min;
        ValueT scalar_max_used = scalar_max;
        if (!std::isfinite(scalar_min_used)) scalar_min_used = scalars.minCoeff();
        if (!std::isfinite(scalar_max_used)) scalar_max_used = scalars.maxCoeff();
        if (!std::isfinite(scalar_min_used) || !std::isfinite(scalar_max_used)) {
            return VectorSet<float,3>::Zero(3, scalars.cols());
        }

        const ValueT scalar_range = (scalar_max_used == scalar_min_used) ? (ValueT)1.0 : (scalar_max_used - scalar_min_used);

        VectorSet<float,3> colors(3, scalars.cols());
        switch (colormap_type) {
            case ColormapType::JET:
#pragma omp parallel for
                for (size_t i = 0; i < scalars.cols(); i++) {
                    const float scalar_normalized = (scalars[i] - scalar_min_used)/scalar_range;
                    if (scalar_normalized < 0.7f)
                        colors(0,i) = std::max(std::min(4.0f*scalar_normalized - 1.5f, 1.0f), 0.0f);
                    else
                        colors(0,i) = std::max(std::min(-4.0f*scalar_normalized + 4.5f, 1.0f), 0.0f);
                    if (scalar_normalized < 0.5f)
                        colors(1,i) = std::max(std::min(4.0f*scalar_normalized - 0.5f, 1.0f), 0.0f);
                    else
                        colors(1,i) = std::max(std::min(-4.0f*scalar_normalized + 3.5f, 1.0f), 0.0f);
                    if (scalar_normalized < 0.3f)
                        colors(2,i) = std::max(std::min(4.0f*scalar_normalized + 0.5f, 1.0f), 0.0f);
                    else
                        colors(2,i) = std::max(std::min(-4.0f*scalar_normalized + 2.5f, 1.0f), 0.0f);
                }
                break;
            case ColormapType::GRAY:
#pragma omp parallel for
                for (size_t i = 0; i < scalars.cols(); i++) {
                    colors.col(i).setConstant(std::max(std::min((scalars[i] - scalar_min_used)/scalar_range, 1.0f), 0.0f));
                }
                break;
            case ColormapType::BLUE2RED:
#pragma omp parallel for
                for (size_t i = 0; i < scalars.cols(); i++) {
                    const float scalar_normalized = (scalars[i] - scalar_min_used)/scalar_range;
                    if (scalar_normalized < 0.5f) {
                        const float val = std::max(std::min(2.0f*scalar_normalized, 1.0f), 0.0f);
                        colors(0,i) = val;
                        colors(1,i) = val;
                        colors(2,i) = 1.0f;
                    } else {
                        const float val = std::max(std::min(2.0f*(1.0f - scalar_normalized), 1.0f), 0.0f);
                        colors(0,i) = 1.0f;
                        colors(1,i) = val;
                        colors(2,i) = val;
                    }
                }
                break;
        }

        return colors;
    }
}
