#pragma once

#include <cilantro/kd_tree.hpp>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2>
    class NearestNeighborGraph {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        NearestNeighborGraph() {}

        NearestNeighborGraph(const KDTree<ScalarT,EigenDim,DistAdaptor> &indexed_points_tree,
                             const ConstVectorSetMatrixMap<ScalarT,EigenDim> &query_points,
                             const NeighborhoodSpecification<ScalarT> &nh)
                : nh_(nh)
        {
            compute_(indexed_points_tree, query_points);
        }

        NearestNeighborGraph(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &indexed_points,
                             const ConstVectorSetMatrixMap<ScalarT,EigenDim> &query_points,
                             const NeighborhoodSpecification<ScalarT> &nh)
                : nh_(nh)
        {
            compute_(KDTree<ScalarT,EigenDim,DistAdaptor>(indexed_points), query_points);
        }

        NearestNeighborGraph(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points,
                             const NeighborhoodSpecification<ScalarT> &nh)
                : nh_(nh)
        {
            compute_(KDTree<ScalarT,EigenDim,DistAdaptor>(points), points);
        }

        ~NearestNeighborGraph() {}

        inline const std::vector<std::vector<size_t>>& getNeighborIndices() const { return neighbor_indices_; }

        inline const std::vector<std::vector<ScalarT>>& getNeighborDistances() const { return neighbor_distances_; }

        inline const NeighborhoodSpecification<ScalarT>& getNeighborhoodSpecification() const { return nh_; }

    private:
        std::vector<std::vector<size_t>> neighbor_indices_;
        std::vector<std::vector<ScalarT>> neighbor_distances_;
        NeighborhoodSpecification<ScalarT> nh_;

        inline void compute_(const KDTree<ScalarT,EigenDim,DistAdaptor> &ref_tree,
                             const ConstVectorSetMatrixMap<ScalarT,EigenDim> &query_points)
        {
            switch (nh_.type) {
                case NeighborhoodType::KNN:
                    compute_<NeighborhoodType::KNN>(ref_tree, query_points);
                    break;
                case NeighborhoodType::RADIUS:
                    compute_<NeighborhoodType::RADIUS>(ref_tree, query_points);
                    break;
                case NeighborhoodType::KNN_IN_RADIUS:
                    compute_<NeighborhoodType::KNN_IN_RADIUS>(ref_tree, query_points);
                    break;
            }
        }

        template <NeighborhoodType NT>
        void compute_(const KDTree<ScalarT,EigenDim,DistAdaptor> &ref_tree,
                      const ConstVectorSetMatrixMap<ScalarT,EigenDim> &query_points)
        {
            neighbor_indices_.resize(query_points.cols());
            neighbor_distances_.resize(query_points.cols());
            std::vector<size_t> neighbors;
            std::vector<ScalarT> distances;
            neighbors.reserve(query_points.cols());
            distances.reserve(query_points.cols());

#pragma omp parallel for private (neighbors, distances)
            for (size_t i = 0; i < query_points.cols(); i++) {
                ref_tree.template search<NT>(query_points.col(i), nh_, neighbors, distances);
                neighbor_indices_[i] = neighbors;
                neighbor_distances_[i] = distances;
            }
        }
    };

    typedef NearestNeighborGraph<float,2> NearestNeighborGraph2D;
    typedef NearestNeighborGraph<float,3> NearestNeighborGraph3D;
}
