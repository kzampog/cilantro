#pragma once

#include <cilantro/principal_component_analysis.hpp>
#include <cilantro/kd_tree.hpp>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim>
    class NormalEstimation {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        NormalEstimation(const ConstPointSetMatrixMap<ScalarT,EigenDim> &points)
                : points_(points),
                  kd_tree_ptr_(new KDTree<ScalarT,EigenDim,KDTreeDistanceAdaptors::L2>(points)),
                  kd_tree_owned_(true),
                  view_point_(Point<ScalarT,EigenDim>::Zero())
        {}

        NormalEstimation(const ConstPointSetMatrixMap<ScalarT,EigenDim> &points, const KDTree<ScalarT,EigenDim,KDTreeDistanceAdaptors::L2> &kd_tree)
                : points_(points),
                  kd_tree_ptr_(&kd_tree),
                  kd_tree_owned_(false),
                  view_point_(Point<ScalarT,EigenDim>::Zero())
        {}

        ~NormalEstimation() {
            if (kd_tree_owned_) delete kd_tree_ptr_;
        }

        inline const Point<ScalarT,EigenDim>& getViewPoint() const { return view_point_; }

        inline NormalEstimation& setViewPoint(const Eigen::Ref<const Point<ScalarT,EigenDim>> &vp) { view_point_ = vp; return *this; }

        PointSet<ScalarT,EigenDim> estimateNormalsKNN(size_t num_neighbors) const {
            size_t dim = points_.rows();
            size_t num_points = points_.cols();

            PointSet<ScalarT,EigenDim> normals(dim, num_points);
            Point<ScalarT,EigenDim> nan(Point<ScalarT,EigenDim>::Constant(dim, 1, std::numeric_limits<ScalarT>::quiet_NaN()));
            if (num_points < dim) {
                for (size_t i = 0; i < num_points; i++) normals.col(i) = nan;
                return normals;
            }

            std::vector<size_t> neighbors;
            std::vector<ScalarT> distances;
#pragma omp parallel for shared (normals) private (neighbors, distances)
            for (size_t i = 0; i < num_points; i++) {
                kd_tree_ptr_->kNNSearch(points_.col(i), num_neighbors, neighbors, distances);
                PointSet<ScalarT,EigenDim> neighborhood(dim, neighbors.size());
                for (size_t j = 0; j < neighbors.size(); j++) {
                    neighborhood.col(j) = points_.col(neighbors[j]);
                }
                PrincipalComponentAnalysis<ScalarT,EigenDim> pca(neighborhood);
                normals.col(i) = pca.getEigenVectors().col(dim-1);
                if (normals.col(i).dot(view_point_ - points_.col(i)) < 0.0) {
                    normals.col(i) *= -1.0;
                }
            }

            return normals;
        }

        PointSet<ScalarT,EigenDim> estimateNormalsRadius(ScalarT radius) const {
            size_t dim = points_.rows();
            size_t num_points = points_.cols();
            ScalarT radius_sq = radius*radius;

            PointSet<ScalarT,EigenDim> normals(dim, num_points);
            Point<ScalarT,EigenDim> nan(Point<ScalarT,EigenDim>::Constant(dim, 1, std::numeric_limits<ScalarT>::quiet_NaN()));

            std::vector<size_t> neighbors;
            std::vector<ScalarT> distances;
#pragma omp parallel for shared (normals) private (neighbors, distances)
            for (size_t i = 0; i < points_.cols(); i++) {
                kd_tree_ptr_->radiusSearch(points_.col(i), radius_sq, neighbors, distances);
                if (neighbors.size() < EigenDim) {
                    normals.col(i) = nan;
                    continue;
                }
                PointSet<ScalarT,EigenDim> neighborhood(dim, neighbors.size());
                for (size_t j = 0; j < neighbors.size(); j++) {
                    neighborhood.col(j) = points_.col(neighbors[j]);
                }
                PrincipalComponentAnalysis<ScalarT,EigenDim> pca(neighborhood);
                normals.col(i) = pca.getEigenVectors().col(dim-1);
                if (normals.col(i).dot(view_point_ - points_.col(i)) < 0.0) {
                    normals.col(i) *= -1.0;
                }
            }

            return normals;
        }

        PointSet<ScalarT,EigenDim> estimateNormalsKNNInRadius(size_t k, ScalarT radius) const {
            size_t dim = points_.rows();
            size_t num_points = points_.cols();
            ScalarT radius_sq = radius*radius;

            PointSet<ScalarT,EigenDim> normals(dim, num_points);
            Point<ScalarT,EigenDim> nan(Point<ScalarT,EigenDim>::Constant(dim, 1, std::numeric_limits<ScalarT>::quiet_NaN()));

            std::vector<size_t> neighbors;
            std::vector<ScalarT> distances;
#pragma omp parallel for shared (normals) private (neighbors, distances)
            for (size_t i = 0; i < points_.cols(); i++) {
                kd_tree_ptr_->kNNInRadiusSearch(points_.col(i), k, radius_sq, neighbors, distances);
                if (neighbors.size() < EigenDim) {
                    normals.col(i) = nan;
                    continue;
                }
                PointSet<ScalarT,EigenDim> neighborhood(dim, neighbors.size());
                for (size_t j = 0; j < neighbors.size(); j++) {
                    neighborhood.col(j) = points_.col(neighbors[j]);
                }
                PrincipalComponentAnalysis<ScalarT,EigenDim> pca(neighborhood);
                normals.col(i) = pca.getEigenVectors().col(dim-1);
                if (normals.col(i).dot(view_point_ - points_.col(i)) < 0.0) {
                    normals.col(i) *= -1.0;
                }
            }

            return normals;
        }

        PointSet<ScalarT,EigenDim> estimateNormals(const typename KDTree<ScalarT,EigenDim,KDTreeDistanceAdaptors::L2>::Neighborhood &nh) const {
            switch (nh.type) {
                case KDTree<ScalarT,EigenDim,KDTreeDistanceAdaptors::L2>::NeighborhoodType::KNN:
                    return estimateNormalsKNN(nh.maxNumberOfNeighbors);
                case KDTree<ScalarT,EigenDim,KDTreeDistanceAdaptors::L2>::NeighborhoodType::RADIUS:
                    return estimateNormalsRadius(nh.radius);
                case KDTree<ScalarT,EigenDim,KDTreeDistanceAdaptors::L2>::NeighborhoodType::KNN_IN_RADIUS:
                    return estimateNormalsKNNInRadius(nh.maxNumberOfNeighbors, nh.radius);
            }
        }

    private:
        ConstPointSetMatrixMap<ScalarT,EigenDim> points_;
        const KDTree<ScalarT,EigenDim,KDTreeDistanceAdaptors::L2> *kd_tree_ptr_;
        bool kd_tree_owned_;
        Point<ScalarT,EigenDim> view_point_;
    };

    typedef NormalEstimation<float,2> NormalEstimation2D;
    typedef NormalEstimation<float,3> NormalEstimation3D;
}
