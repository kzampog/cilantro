#include <cilantro/colormap.hpp>

namespace cilantro {
    std::vector<Eigen::Vector3f> colormap(const std::vector<float> &scalars, float scalar_min, float scalar_max, const ColormapType &colormap_type) {
        float scalar_min_used = scalar_min;
        float scalar_max_used = scalar_max;
        if (!std::isfinite(scalar_min_used)) scalar_min_used = *std::min_element(scalars.begin(), scalars.end());
        if (!std::isfinite(scalar_max_used)) scalar_max_used = *std::max_element(scalars.begin(), scalars.end());
        if (!std::isfinite(scalar_min_used) || !std::isfinite(scalar_max_used)) {
            return std::vector<Eigen::Vector3f>(scalars.size(), Eigen::Vector3f::Zero());
        }

        float scalar_range = scalar_max_used - scalar_min_used;
        float scalar_normalized;

        std::vector<Eigen::Vector3f> colors(scalars.size());
        switch (colormap_type) {
            case ColormapType::JET:
                for (size_t i = 0; i < scalars.size(); i++) {
                    scalar_normalized = (scalars[i] - scalar_min_used)/scalar_range;
                    if (scalar_normalized < 0.7f)
                        colors[i][0] = std::max(std::min(4.0f*scalar_normalized - 1.5f, 1.0f), 0.0f);
                    else
                        colors[i][0] = std::max(std::min(-4.0f*scalar_normalized + 4.5f, 1.0f), 0.0f);
                    if (scalar_normalized < 0.5f)
                        colors[i][1] = std::max(std::min(4.0f*scalar_normalized - 0.5f, 1.0f), 0.0f);
                    else
                        colors[i][1] = std::max(std::min(-4.0f*scalar_normalized + 3.5f, 1.0f), 0.0f);
                    if (scalar_normalized < 0.3f)
                        colors[i][2] = std::max(std::min(4.0f*scalar_normalized + 0.5f, 1.0f), 0.0f);
                    else
                        colors[i][2] = std::max(std::min(-4.0f*scalar_normalized + 2.5f, 1.0f), 0.0f);
                }
                break;
            case ColormapType::GRAY:
                for (size_t i = 0; i < scalars.size(); i++) {
                    scalar_normalized = (scalars[i] - scalar_min_used)/scalar_range;
                    colors[i][0] = std::max(std::min(scalar_normalized, 1.0f), 0.0f);
                    colors[i][1] = std::max(std::min(scalar_normalized, 1.0f), 0.0f);
                    colors[i][2] = std::max(std::min(scalar_normalized, 1.0f), 0.0f);
                }
                break;
            case ColormapType::BLUE2RED:
                for (size_t i = 0; i < scalars.size(); i++) {
                    scalar_normalized = (scalars[i] - scalar_min_used)/scalar_range;
                    if (scalar_normalized < 0.5f) {
                        colors[i][0] = std::max(std::min(2.0f*scalar_normalized, 1.0f), 0.0f);
                        colors[i][1] = std::max(std::min(2.0f*scalar_normalized, 1.0f), 0.0f);
                        colors[i][2] = 1.0f;
                    } else {
                        colors[i][0] = 1.0f;
                        colors[i][1] = std::max(std::min(2.0f*(1.0f - scalar_normalized), 1.0f), 0.0f);
                        colors[i][2] = std::max(std::min(2.0f*(1.0f - scalar_normalized), 1.0f), 0.0f);
                    }
                }
                break;
        }

        return colors;
    }
}
