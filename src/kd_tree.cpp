#include <cilantro/kd_tree.hpp>

namespace cilantro {
    template class KDTree<float,2,KDTreeDistanceAdaptors::L2>;
    template class KDTree<float,3,KDTreeDistanceAdaptors::L2>;
}
