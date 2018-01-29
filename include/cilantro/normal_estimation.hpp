#pragma once

#include <cilantro/principal_component_analysis.hpp>
#include <cilantro/kd_tree.hpp>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim>
    class NormalEstimation {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        NormalEstimation(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points)
                : points_(points),
                  kd_tree_ptr_(new KDTree<ScalarT,EigenDim,KDTreeDistanceAdaptors::L2>(points)),
                  kd_tree_owned_(true),
                  view_point_(Vector<ScalarT,EigenDim>::Zero())
        {}

        NormalEstimation(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points, const KDTree<ScalarT,EigenDim,KDTreeDistanceAdaptors::L2> &kd_tree)
                : points_(points),
                  kd_tree_ptr_(&kd_tree),
                  kd_tree_owned_(false),
                  view_point_(Vector<ScalarT,EigenDim>::Zero())
        {}

        ~NormalEstimation() {
            if (kd_tree_owned_) delete kd_tree_ptr_;
        }

        inline const Vector<ScalarT,EigenDim>& getViewPoint() const { return view_point_; }

        inline NormalEstimation& setViewPoint(const Eigen::Ref<const Vector<ScalarT,EigenDim>> &vp) { view_point_ = vp; return *this; }

        inline const NormalEstimation& estimateNormalsAndCurvatureKNN(VectorSet<ScalarT,EigenDim> &normals, VectorSet<ScalarT,1> &curvatures, size_t k) const {
            compute_knn_(normals, curvatures, k);
            return *this;
        }

        inline VectorSet<ScalarT,EigenDim> estimateNormalsKNN(size_t k) const {
            VectorSet<ScalarT,EigenDim> normals;
            VectorSet<ScalarT,1> curvatures;
            compute_knn_(normals, curvatures, k);
            return normals;
        }

        inline VectorSet<ScalarT,1> estimateCurvatureKNN(size_t k) const {
            VectorSet<ScalarT,EigenDim> normals;
            VectorSet<ScalarT,1> curvatures;
            compute_knn_(normals, curvatures, k);
            return curvatures;
        }

        inline const NormalEstimation& estimateNormalsAndCurvatureRadius(VectorSet<ScalarT,EigenDim> &normals, VectorSet<ScalarT,1> &curvatures, ScalarT radius) const {
            compute_radius_(normals, curvatures, radius);
            return *this;
        }

        inline VectorSet<ScalarT,EigenDim> estimateNormalsRadius(ScalarT radius) const {
            VectorSet<ScalarT,EigenDim> normals;
            VectorSet<ScalarT,1> curvatures;
            compute_radius_(normals, curvatures, radius);
            return normals;
        }

        inline VectorSet<ScalarT,1> estimateCurvatureRadius(ScalarT radius) const {
            VectorSet<ScalarT,EigenDim> normals;
            VectorSet<ScalarT,1> curvatures;
            compute_radius_(normals, curvatures, radius);
            return curvatures;
        }

        inline const NormalEstimation& estimateNormalsAndCurvatureKNNInRadius(VectorSet<ScalarT,EigenDim> &normals, VectorSet<ScalarT,1> &curvatures, size_t k, ScalarT radius) const {
            compute_knn_in_radius_(normals, curvatures, k, radius);
            return *this;
        }

        inline VectorSet<ScalarT,EigenDim> estimateNormalsKNNInRadius(size_t k, ScalarT radius) const {
            VectorSet<ScalarT,EigenDim> normals;
            VectorSet<ScalarT,1> curvatures;
            compute_knn_in_radius_(normals, curvatures, k, radius);
            return normals;
        }

        inline VectorSet<ScalarT,1> estimateCurvatureKNNInRadius(size_t k, ScalarT radius) const {
            VectorSet<ScalarT,EigenDim> normals;
            VectorSet<ScalarT,1> curvatures;
            compute_knn_in_radius_(normals, curvatures, k, radius);
            return curvatures;
        }

        inline const NormalEstimation& estimateNormalsAndCurvature(VectorSet<ScalarT,EigenDim> &normals, VectorSet<ScalarT,1> &curvatures, const typename KDTree<ScalarT,EigenDim,KDTreeDistanceAdaptors::L2>::Neighborhood &nh) const {
            compute_nh_(normals, curvatures, nh);
            return *this;
        }

        inline VectorSet<ScalarT,EigenDim> estimateNormals(const typename KDTree<ScalarT,EigenDim,KDTreeDistanceAdaptors::L2>::Neighborhood &nh) const {
            VectorSet<ScalarT,EigenDim> normals;
            VectorSet<ScalarT,1> curvatures;
            compute_nh_(normals, curvatures, nh);
            return normals;
        }

        inline VectorSet<ScalarT,1> estimateCurvature(const typename KDTree<ScalarT,EigenDim,KDTreeDistanceAdaptors::L2>::Neighborhood &nh) const {
            VectorSet<ScalarT,EigenDim> normals;
            VectorSet<ScalarT,1> curvatures;
            compute_nh_(normals, curvatures, nh);
            return curvatures;
        }

    private:
        ConstVectorSetMatrixMap<ScalarT,EigenDim> points_;
        const KDTree<ScalarT,EigenDim,KDTreeDistanceAdaptors::L2> *kd_tree_ptr_;
        bool kd_tree_owned_;
        Vector<ScalarT,EigenDim> view_point_;

        void compute_knn_(VectorSet<ScalarT,EigenDim> &normals, VectorSet<ScalarT,1> &curvatures, size_t k) const {
            size_t dim = points_.rows();
            size_t num_points = points_.cols();

            normals.resize(dim, num_points);
            curvatures.resize(1, num_points);

            if (num_points < dim) {
                for (size_t i = 0; i < num_points; i++) {
                    normals.col(i).setConstant(std::numeric_limits<ScalarT>::quiet_NaN());
                    curvatures[i] = std::numeric_limits<ScalarT>::quiet_NaN();
                }
                return;
            }

            std::vector<size_t> neighbors;
            std::vector<ScalarT> distances;
#pragma omp parallel for shared (normals) private (neighbors, distances)
            for (size_t i = 0; i < num_points; i++) {
                kd_tree_ptr_->kNNSearch(points_.col(i), k, neighbors, distances);
                VectorSet<ScalarT,EigenDim> neighborhood(dim, neighbors.size());
                for (size_t j = 0; j < neighbors.size(); j++) {
                    neighborhood.col(j) = points_.col(neighbors[j]);
                }
                PrincipalComponentAnalysis<ScalarT,EigenDim> pca(neighborhood);
                normals.col(i) = pca.getEigenVectors().col(dim-1);
                if (normals.col(i).dot(view_point_ - points_.col(i)) < 0.0) {
                    normals.col(i) *= -1.0;
                }
                curvatures[i] = pca.getEigenValues()[dim-1]/pca.getEigenValues().sum();
            }
        }

        void compute_radius_(VectorSet<ScalarT,EigenDim> &normals, VectorSet<ScalarT,1> &curvatures, ScalarT radius) const {
            size_t dim = points_.rows();
            size_t num_points = points_.cols();
            ScalarT radius_sq = radius*radius;

            normals.resize(dim, num_points);
            curvatures.resize(1, num_points);

            std::vector<size_t> neighbors;
            std::vector<ScalarT> distances;
#pragma omp parallel for shared (normals) private (neighbors, distances)
            for (size_t i = 0; i < num_points; i++) {
                kd_tree_ptr_->radiusSearch(points_.col(i), radius_sq, neighbors, distances);
                if (neighbors.size() < EigenDim) {
                    normals.col(i).setConstant(std::numeric_limits<ScalarT>::quiet_NaN());
                    curvatures[i] = std::numeric_limits<ScalarT>::quiet_NaN();
                    continue;
                }
                VectorSet<ScalarT,EigenDim> neighborhood(dim, neighbors.size());
                for (size_t j = 0; j < neighbors.size(); j++) {
                    neighborhood.col(j) = points_.col(neighbors[j]);
                }
                PrincipalComponentAnalysis<ScalarT,EigenDim> pca(neighborhood);
                normals.col(i) = pca.getEigenVectors().col(dim-1);
                if (normals.col(i).dot(view_point_ - points_.col(i)) < 0.0) {
                    normals.col(i) *= -1.0;
                }
                curvatures[i] = pca.getEigenValues()[dim-1]/pca.getEigenValues().sum();
            }
        }

        void compute_knn_in_radius_(VectorSet<ScalarT,EigenDim> &normals, VectorSet<ScalarT,1> &curvatures, size_t k, ScalarT radius) const {
            size_t dim = points_.rows();
            size_t num_points = points_.cols();
            ScalarT radius_sq = radius*radius;

            normals.resize(dim, num_points);
            curvatures.resize(1, num_points);

            std::vector<size_t> neighbors;
            std::vector<ScalarT> distances;
#pragma omp parallel for shared (normals) private (neighbors, distances)
            for (size_t i = 0; i < num_points; i++) {
                kd_tree_ptr_->kNNInRadiusSearch(points_.col(i), k, radius_sq, neighbors, distances);
                if (neighbors.size() < EigenDim) {
                    normals.col(i).setConstant(std::numeric_limits<ScalarT>::quiet_NaN());
                    curvatures[i] = std::numeric_limits<ScalarT>::quiet_NaN();
                    continue;
                }
                VectorSet<ScalarT,EigenDim> neighborhood(dim, neighbors.size());
                for (size_t j = 0; j < neighbors.size(); j++) {
                    neighborhood.col(j) = points_.col(neighbors[j]);
                }
                PrincipalComponentAnalysis<ScalarT,EigenDim> pca(neighborhood);
                normals.col(i) = pca.getEigenVectors().col(dim-1);
                if (normals.col(i).dot(view_point_ - points_.col(i)) < 0.0) {
                    normals.col(i) *= -1.0;
                }
                curvatures[i] = pca.getEigenValues()[dim-1]/pca.getEigenValues().sum();
            }
        }

        void compute_nh_(VectorSet<ScalarT,EigenDim> &normals, VectorSet<ScalarT,1> &curvatures, const typename KDTree<ScalarT,EigenDim,KDTreeDistanceAdaptors::L2>::Neighborhood &nh) const {
            switch (nh.type) {
                case KDTree<ScalarT,EigenDim,KDTreeDistanceAdaptors::L2>::NeighborhoodType::KNN:
                    compute_knn_(normals, curvatures, nh.maxNumberOfNeighbors);
                    break;
                case KDTree<ScalarT,EigenDim,KDTreeDistanceAdaptors::L2>::NeighborhoodType::RADIUS:
                    compute_radius_(normals, curvatures, nh.radius);
                    break;
                case KDTree<ScalarT,EigenDim,KDTreeDistanceAdaptors::L2>::NeighborhoodType::KNN_IN_RADIUS:
                    compute_knn_in_radius_(normals, curvatures, nh.maxNumberOfNeighbors, nh.radius);
                    break;
            }
        }
    };

    typedef NormalEstimation<float,2> NormalEstimation2D;
    typedef NormalEstimation<float,3> NormalEstimation3D;
}
