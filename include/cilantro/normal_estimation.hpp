#pragma once

#include <cilantro/principal_component_analysis.hpp>
#include <cilantro/kd_tree.hpp>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim>
    class NormalEstimation {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        NormalEstimation(const ConstDataMatrixMap<ScalarT,EigenDim> &points)
                : points_(points),
                  kd_tree_ptr_(new KDTree<ScalarT,EigenDim,KDTreeDistanceAdaptors::L2>(points)),
                  kd_tree_owned_(true),
                  view_point_(Eigen::Matrix<ScalarT,EigenDim,1>::Zero())
        {}

        NormalEstimation(const ConstDataMatrixMap<ScalarT,EigenDim> &points, const KDTree<ScalarT,EigenDim,KDTreeDistanceAdaptors::L2> &kd_tree)
                : points_(points),
                  kd_tree_ptr_(&kd_tree),
                  kd_tree_owned_(false),
                  view_point_(Eigen::Matrix<ScalarT,EigenDim,1>::Zero())
        {}

        ~NormalEstimation() {
            if (kd_tree_owned_) delete kd_tree_ptr_;
        }

        inline const Eigen::Matrix<ScalarT,EigenDim,1>& getViewPoint() const { return view_point_; }
        inline NormalEstimation& setViewPoint(const Eigen::Ref<const Eigen::Matrix<ScalarT,EigenDim,1> > &vp) { view_point_ = vp; return *this; }

        std::vector<Eigen::Matrix<ScalarT,EigenDim,1> > estimateNormalsKNN(size_t num_neighbors) const {
            std::vector<Eigen::Matrix<ScalarT,EigenDim,1> > normals(points_.cols());

            Eigen::Matrix<ScalarT,EigenDim,1> nan(Eigen::Matrix<ScalarT,EigenDim,1>::Constant(std::numeric_limits<ScalarT>::quiet_NaN()));
            if (points_.cols() < EigenDim) {
                for (size_t i = 0; i < normals.size(); i++) normals[i] = nan;
                return normals;
            }

            std::vector<size_t> neighbors;
            std::vector<ScalarT> distances;
#pragma omp parallel for shared (normals) private (neighbors, distances)
            for (size_t i = 0; i < points_.cols(); i++) {
                kd_tree_ptr_->kNNSearch(points_.col(i), num_neighbors, neighbors, distances);

                std::vector<Eigen::Matrix<ScalarT,EigenDim,1> > neighborhood(neighbors.size());
                for (size_t j = 0; j < neighbors.size(); j++) {
                    neighborhood[j] = points_.col(neighbors[j]);
                }

                PrincipalComponentAnalysis<ScalarT,EigenDim> pca(neighborhood);
                normals[i] = pca.getEigenVectorsMatrix().col(EigenDim-1);

//                points_.col(i) = pca.reconstruct<EigenDim-1>(pca.project<EigenDim-1>(points_.col(i)));

                if (normals[i].dot(view_point_ - points_.col(i)) < 0.0) {
                    normals[i] *= -1.0;
                }
            }

            return normals;
        }

        std::vector<Eigen::Matrix<ScalarT,EigenDim,1> > estimateNormalsRadius(ScalarT radius) const {
            ScalarT radius_sq = radius*radius;

            std::vector<Eigen::Matrix<ScalarT,EigenDim,1> > normals(points_.cols());

            Eigen::Matrix<ScalarT,EigenDim,1> nan(Eigen::Matrix<ScalarT,EigenDim,1>::Constant(std::numeric_limits<ScalarT>::quiet_NaN()));

            std::vector<size_t> neighbors;
            std::vector<ScalarT> distances;
#pragma omp parallel for shared (normals) private (neighbors, distances)
            for (size_t i = 0; i < points_.cols(); i++) {
                kd_tree_ptr_->radiusSearch(points_.col(i), radius_sq, neighbors, distances);

                if (neighbors.size() < EigenDim) {
                    normals[i] = nan;
                    continue;
                }

                std::vector<Eigen::Matrix<ScalarT,EigenDim,1> > neighborhood(neighbors.size());
                for (size_t j = 0; j < neighbors.size(); j++) {
                    neighborhood[j] = points_.col(neighbors[j]);
                }

                PrincipalComponentAnalysis<ScalarT,EigenDim> pca(neighborhood);
                normals[i] = pca.getEigenVectorsMatrix().col(EigenDim-1);

//                points_.col(i) = pca.reconstruct<EigenDim-1>(pca.project<EigenDim-1>(points_.col(i)));

                if (normals[i].dot(view_point_ - points_.col(i)) < 0.0) {
                    normals[i] *= -1.0;
                }
            }

            return normals;
        }

        std::vector<Eigen::Matrix<ScalarT,EigenDim,1> > estimateNormalsKNNInRadius(size_t k, ScalarT radius) const {
            ScalarT radius_sq = radius*radius;

            std::vector<Eigen::Matrix<ScalarT,EigenDim,1> > normals(points_.cols());

            Eigen::Matrix<ScalarT,EigenDim,1> nan(Eigen::Matrix<ScalarT,EigenDim,1>::Constant(std::numeric_limits<ScalarT>::quiet_NaN()));

            std::vector<size_t> neighbors;
            std::vector<ScalarT> distances;
#pragma omp parallel for shared (normals) private (neighbors, distances)
            for (size_t i = 0; i < points_.cols(); i++) {
                kd_tree_ptr_->kNNInRadiusSearch(points_.col(i), k, radius_sq, neighbors, distances);

                if (neighbors.size() < EigenDim) {
                    normals[i] = nan;
                    continue;
                }

                std::vector<Eigen::Matrix<ScalarT,EigenDim,1> > neighborhood(neighbors.size());
                for (size_t j = 0; j < neighbors.size(); j++) {
                    neighborhood[j] = points_.col(neighbors[j]);
                }

                PrincipalComponentAnalysis<ScalarT,EigenDim> pca(neighborhood);
                normals[i] = pca.getEigenVectorsMatrix().col(EigenDim-1);

//                points_.col(i) = pca.reconstruct<EigenDim-1>(pca.project<EigenDim-1>(points_.col(i)));

                if (normals[i].dot(view_point_ - points_.col(i)) < 0.0) {
                    normals[i] *= -1.0;
                }
            }

            return normals;
        }

        std::vector<Eigen::Matrix<ScalarT,EigenDim,1> > estimateNormals(const KDTree3D::Neighborhood &nh) const {
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
        ConstDataMatrixMap<ScalarT,EigenDim> points_;
        const KDTree<ScalarT,EigenDim,KDTreeDistanceAdaptors::L2> *kd_tree_ptr_;
        bool kd_tree_owned_;
        Eigen::Matrix<ScalarT,EigenDim,1> view_point_;
    };

    typedef NormalEstimation<float,2> NormalEstimation2D;
    typedef NormalEstimation<float,3> NormalEstimation3D;
}
