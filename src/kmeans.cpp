#include <cilantro/kmeans.hpp>

namespace cilantro {
    template class KMeans<float,2,KDTreeDistanceAdaptors::L2>;
    template class KMeans<float,3,KDTreeDistanceAdaptors::L2>;
}
