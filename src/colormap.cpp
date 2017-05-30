#include <cilantro/colormap.hpp>

std::vector<Eigen::Vector3f> colormap (const std::vector<float> &scalars, float scalar_min, float scalar_max, ColormapType colormapType) {
    float scalarMinUsed = scalar_min;
    float scalarMaxUsed = scalar_max;
    if (scalarMinUsed != scalarMinUsed || scalarMaxUsed != scalarMaxUsed) {
        scalarMinUsed = *std::min_element(scalars.begin(), scalars.end());
        scalarMaxUsed = *std::max_element(scalars.begin(), scalars.end());
    }

    std::vector<Eigen::Vector3f> colors(scalars.size());
    for (size_t i = 0; i < scalars.size(); i++) {
        Eigen::Vector3f color;
        float scalarNormalized = (scalars[i] - scalarMinUsed) / (scalarMaxUsed - scalarMinUsed);

        switch (colormapType) {
            case ColormapType::GREY:
                color[0] = scalarNormalized;
                color[1] = scalarNormalized;
                color[2] = scalarNormalized;
                break;

            case ColormapType::JET:
                if (scalarNormalized < 0.7)
                    color[0] = 4.0 * scalarNormalized - 1.5;
                else
                    color[0] = -4.0 * scalarNormalized + 4.5;

                if (scalarNormalized < 0.5)
                    color[1] =  4.0 * scalarNormalized - 0.5;
                else
                    color[1] =  -4.0 * scalarNormalized + 3.5;

                if (scalarNormalized < 0.3)
                    color[2] =  4.0 * scalarNormalized + 0.5;
                else
                    color[2] =  -4.0 * scalarNormalized + 2.5;

                break;

            case ColormapType::BLUE2RED:

                Eigen::Vector3f red(1, 0, 0);
                Eigen::Vector3f white(1, 1, 1);
                Eigen::Vector3f blue(0, 0, 1);

                if (scalarNormalized < 0.5) {
                    color.head(3) = white * scalarNormalized*2 + blue * (1 - scalarNormalized*2);
                } else {
                    color.head(3) = red * (scalarNormalized-0.5)*2 + white * (1 - scalarNormalized)*2;
                }
        }

        color[0] = std::max(std::min(color[0], 1.0f), 0.0f);
        color[1] = std::max(std::min(color[1], 1.0f), 0.0f);
        color[2] = std::max(std::min(color[2], 1.0f), 0.0f);

        colors[i] = color;
    }

    return colors;
}