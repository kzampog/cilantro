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

    enum struct NeighborhoodType {KNN, RADIUS, KNN_IN_RADIUS};

    template <typename ScalarT>
    struct NeighborhoodSpecification {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        NeighborhoodType type;
        size_t maxNumberOfNeighbors;
        ScalarT radius;

        inline NeighborhoodSpecification()
                : type(NeighborhoodType::KNN), maxNumberOfNeighbors(1)
        {}

        inline NeighborhoodSpecification(size_t knn, ScalarT radius)
                : type(NeighborhoodType::KNN_IN_RADIUS), maxNumberOfNeighbors(knn), radius(radius)
        {}

        inline NeighborhoodSpecification(NeighborhoodType type, size_t knn, ScalarT radius)
                : type(type), maxNumberOfNeighbors(knn), radius(radius)
        {}
    };

    template <typename ScalarT>
    inline NeighborhoodSpecification<ScalarT> kNNNeighborhood(size_t k) {
        return NeighborhoodSpecification<ScalarT>(NeighborhoodType::KNN, k, (ScalarT)0.0);
    }

    template <typename ScalarT>
    inline NeighborhoodSpecification<ScalarT> radiusNeighborhood(ScalarT radius) {
        return NeighborhoodSpecification<ScalarT>(NeighborhoodType::RADIUS, 0, radius);
    }

    template <typename ScalarT>
    inline NeighborhoodSpecification<ScalarT> kNNInRadiusNeighborhood(size_t k, ScalarT radius) {
        return NeighborhoodSpecification<ScalarT>(NeighborhoodType::KNN_IN_RADIUS, k, radius);
    }
}
