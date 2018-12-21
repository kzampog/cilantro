#pragma once

#include <cilantro/kd_tree.hpp>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim>
    class NormalEstimation {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef ScalarT Scalar;

        enum { Dimension = EigenDim };

        NormalEstimation(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points, size_t max_leaf_size = 10)
                : points_(points),
                  kd_tree_ptr_(new KDTree<ScalarT,EigenDim,KDTreeDistanceAdaptors::L2>(points, max_leaf_size)),
                  kd_tree_owned_(true),
                  view_point_(Vector<ScalarT,EigenDim>::Zero(points_.rows(), 1))
        {}

        NormalEstimation(const KDTree<ScalarT,EigenDim,KDTreeDistanceAdaptors::L2> &kd_tree)
                : points_(kd_tree.getPointsMatrixMap()),
                  kd_tree_ptr_(&kd_tree),
                  kd_tree_owned_(false),
                  view_point_(Vector<ScalarT,EigenDim>::Zero(points_.rows(), 1))
        {}

        ~NormalEstimation() {
            if (kd_tree_owned_) delete kd_tree_ptr_;
        }

        inline const Vector<ScalarT,EigenDim>& getViewPoint() const { return view_point_; }

        inline NormalEstimation& setViewPoint(const Eigen::Ref<const Vector<ScalarT,EigenDim>> &vp) {
            view_point_ = vp;
            return *this;
        }

        inline const NormalEstimation& estimateNormalsAndCurvatureKNN(VectorSet<ScalarT,EigenDim> &normals,
                                                                      VectorSet<ScalarT,1> &curvatures,
                                                                      size_t k) const
        {
            compute_(normals, curvatures, KNNNeighborhoodSpecification(k));
            return *this;
        }

        inline VectorSet<ScalarT,EigenDim> estimateNormalsKNN(size_t k) const {
            VectorSet<ScalarT,EigenDim> normals;
            VectorSet<ScalarT,1> curvatures;
            compute_(normals, curvatures, KNNNeighborhoodSpecification(k));
            return normals;
        }

        inline VectorSet<ScalarT,1> estimateCurvatureKNN(size_t k) const {
            VectorSet<ScalarT,EigenDim> normals;
            VectorSet<ScalarT,1> curvatures;
            compute_(normals, curvatures, KNNNeighborhoodSpecification(k));
            return curvatures;
        }

        inline const NormalEstimation& estimateNormalsAndCurvatureRadius(VectorSet<ScalarT,EigenDim> &normals,
                                                                         VectorSet<ScalarT,1> &curvatures,
                                                                         ScalarT radius) const
        {
            compute_(normals, curvatures, RadiusNeighborhoodSpecification<ScalarT>(radius*radius));
            return *this;
        }

        inline VectorSet<ScalarT,EigenDim> estimateNormalsRadius(ScalarT radius) const {
            VectorSet<ScalarT,EigenDim> normals;
            VectorSet<ScalarT,1> curvatures;
            compute_(normals, curvatures, RadiusNeighborhoodSpecification<ScalarT>(radius*radius));
            return normals;
        }

        inline VectorSet<ScalarT,1> estimateCurvatureRadius(ScalarT radius) const {
            VectorSet<ScalarT,EigenDim> normals;
            VectorSet<ScalarT,1> curvatures;
            compute_(normals, curvatures, RadiusNeighborhoodSpecification<ScalarT>(radius*radius));
            return curvatures;
        }

        inline const NormalEstimation& estimateNormalsAndCurvatureKNNInRadius(VectorSet<ScalarT,EigenDim> &normals,
                                                                              VectorSet<ScalarT,1> &curvatures,
                                                                              size_t k,
                                                                              ScalarT radius) const
        {
            compute_(normals, curvatures, KNNInRadiusNeighborhoodSpecification<ScalarT>(k, radius*radius));
            return *this;
        }

        inline VectorSet<ScalarT,EigenDim> estimateNormalsKNNInRadius(size_t k, ScalarT radius) const {
            VectorSet<ScalarT,EigenDim> normals;
            VectorSet<ScalarT,1> curvatures;
            compute_(normals, curvatures, KNNInRadiusNeighborhoodSpecification<ScalarT>(k, radius*radius));
            return normals;
        }

        inline VectorSet<ScalarT,1> estimateCurvatureKNNInRadius(size_t k, ScalarT radius) const {
            VectorSet<ScalarT,EigenDim> normals;
            VectorSet<ScalarT,1> curvatures;
            compute_(normals, curvatures, KNNInRadiusNeighborhoodSpecification<ScalarT>(k, radius*radius));
            return curvatures;
        }

        template <typename NeighborhoodSpecT>
        inline const NormalEstimation& estimateNormalsAndCurvature(VectorSet<ScalarT,EigenDim> &normals,
                                                                   VectorSet<ScalarT,1> &curvatures,
                                                                   const NeighborhoodSpecT &nh) const
        {
            compute_(normals, curvatures, nh);
            return *this;
        }

        template <typename NeighborhoodSpecT>
        inline VectorSet<ScalarT,EigenDim> estimateNormals(const NeighborhoodSpecT &nh) const {
            VectorSet<ScalarT,EigenDim> normals;
            VectorSet<ScalarT,1> curvatures;
            compute_(normals, curvatures, nh);
            return normals;
        }

        template <typename NeighborhoodSpecT>
        inline VectorSet<ScalarT,1> estimateCurvature(const NeighborhoodSpecT &nh) const {
            VectorSet<ScalarT,EigenDim> normals;
            VectorSet<ScalarT,1> curvatures;
            compute_(normals, curvatures, nh);
            return curvatures;
        }

    private:
        ConstVectorSetMatrixMap<ScalarT,EigenDim> points_;
        const KDTree<ScalarT,EigenDim,KDTreeDistanceAdaptors::L2> *kd_tree_ptr_;
        bool kd_tree_owned_;
        Vector<ScalarT,EigenDim> view_point_;

        template <typename NeighborhoodSpecT>
        void compute_(VectorSet<ScalarT,EigenDim> &normals,
                      VectorSet<ScalarT,1> &curvatures,
                      const NeighborhoodSpecT &nh) const
        {
            const size_t dim = points_.rows();
            const size_t num_points = points_.cols();

            normals.resize(dim, num_points);
            curvatures.resize(1, num_points);

            Neighborhood<ScalarT> nn;
#pragma omp parallel for shared (normals) private (nn)
            for (size_t i = 0; i < num_points; i++) {
                kd_tree_ptr_->template search<NeighborhoodSpecT>(points_.col(i), nh, nn);
                if (nn.size() < dim) {
                    normals.col(i).setConstant(std::numeric_limits<ScalarT>::quiet_NaN());
                    curvatures[i] = std::numeric_limits<ScalarT>::quiet_NaN();
                    continue;
                }

//                VectorSet<ScalarT,EigenDim> neighborhood(dim, nn.size());
//                for (size_t j = 0; j < nn.size(); j++) {
//                    neighborhood.col(j) = points_.col(nn[j].index);
//                }
//                PrincipalComponentAnalysis<ScalarT,EigenDim> pca(neighborhood, false);
//                normals.col(i) = pca.getEigenVectors().col(dim-1);
//                if (normals.col(i).dot(view_point_ - points_.col(i)) < (ScalarT)0.0) {
//                    normals.col(i) *= (ScalarT)(-1.0);
//                }
//                curvatures[i] = pca.getEigenValues()[dim-1]/pca.getEigenValues().sum();

                Vector<ScalarT,EigenDim> mean(Vector<ScalarT,EigenDim>::Zero(points_.rows(), 1));
                for (size_t j = 0; j < nn.size(); j++) {
                    mean += points_.col(nn[j].index);
                }
                mean *= (ScalarT)(1.0)/(nn.size());

                Eigen::Matrix<ScalarT,EigenDim,EigenDim> cov(Eigen::Matrix<ScalarT,EigenDim,EigenDim>::Zero(points_.rows(), points_.rows()));
                for (size_t j = 0; j < nn.size(); j++) {
                    Vector<ScalarT,EigenDim> tmp = points_.col(nn[j].index) - mean;
                    cov += tmp*tmp.transpose();
                }
                cov *= (ScalarT)(1.0)/(nn.size() - 1);

                Eigen::SelfAdjointEigenSolver<Eigen::Matrix<ScalarT,EigenDim,EigenDim>> eig(cov);
                if (eig.eigenvectors().col(0).dot(view_point_ - points_.col(i)) < (ScalarT)0.0) {
                    normals.col(i) = -eig.eigenvectors().col(0);
                } else {
                    normals.col(i) = eig.eigenvectors().col(0);
                }
                curvatures[i] = eig.eigenvalues()[0]/eig.eigenvalues().sum();
            }
        }

    };

    typedef NormalEstimation<float,2> NormalEstimation2f;
    typedef NormalEstimation<double,2> NormalEstimation2d;
    typedef NormalEstimation<float,3> NormalEstimation3f;
    typedef NormalEstimation<double,3> NormalEstimation3d;
    typedef NormalEstimation<float,Eigen::Dynamic> NormalEstimationXf;
    typedef NormalEstimation<double,Eigen::Dynamic> NormalEstimationXd;
}
