#pragma once

#include <vector>
#include <Eigen/Dense>

namespace cilantro {
    template <typename ScalarT>
    struct Neighbor {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef ScalarT Scalar;

        size_t index;
        ScalarT value;

        inline Neighbor() {}

        inline Neighbor(size_t ind, ScalarT dist) : index(ind), value(dist) {}

        struct IndexLessComparator {
            inline bool operator()(const Neighbor &nn1, const Neighbor &nn2) const {
                return nn1.index < nn2.index;
            }
        };

        struct IndexGreaterComparator {
            inline bool operator()(const Neighbor &nn1, const Neighbor &nn2) const {
                return nn1.index > nn2.index;
            }
        };

        struct ValueLessComparator {
            inline bool operator()(const Neighbor &nn1, const Neighbor &nn2) const {
                return nn1.value < nn2.value;
            }
        };

        struct ValueGreaterComparator {
            inline bool operator()(const Neighbor &nn1, const Neighbor &nn2) const {
                return nn1.value > nn2.value;
            }
        };
    };

    template <typename ScalarT>
    using NeighborSet = std::vector<Neighbor<ScalarT>>;

    template <typename ScalarT>
    using Neighborhood = std::vector<Neighbor<ScalarT>>;

    template <typename ScalarT>
    using NeighborhoodSet = std::vector<Neighborhood<ScalarT>>;

    struct KNNNeighborhoodSpecification {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        KNNNeighborhoodSpecification(size_t k = 0) : maxNumberOfNeighbors(k) {}

        size_t maxNumberOfNeighbors;
    };

    template <typename ScalarT>
    struct RadiusNeighborhoodSpecification {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        RadiusNeighborhoodSpecification(ScalarT r = (ScalarT)0) : radius(r) {}

        ScalarT radius;
    };

    template <typename ScalarT>
    struct KNNInRadiusNeighborhoodSpecification {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        KNNInRadiusNeighborhoodSpecification(size_t k = 0, ScalarT r = (ScalarT)0)
                : maxNumberOfNeighbors(k), radius(r)
        {}

        size_t maxNumberOfNeighbors;
        ScalarT radius;
    };
}
