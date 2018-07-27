#pragma once

#include <cilantro/principal_component_analysis.hpp>
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
                  view_point_(Vector<ScalarT,EigenDim>::Zero())
        {}

        NormalEstimation(const KDTree<ScalarT,EigenDim,KDTreeDistanceAdaptors::L2> &kd_tree)
                : points_(kd_tree.getPointsMatrixMap()),
                  kd_tree_ptr_(&kd_tree),
                  kd_tree_owned_(false),
                  view_point_(Vector<ScalarT,EigenDim>::Zero())
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
            compute_<NeighborhoodType::KNN>(normals,
                                            curvatures,
                                            NeighborhoodSpecification<ScalarT>(NeighborhoodType::KNN, k, (ScalarT)0.0));
            return *this;
        }

        inline VectorSet<ScalarT,EigenDim> estimateNormalsKNN(size_t k) const {
            VectorSet<ScalarT,EigenDim> normals;
            VectorSet<ScalarT,1> curvatures;
            compute_<NeighborhoodType::KNN>(normals,
                                            curvatures,
                                            NeighborhoodSpecification<ScalarT>(NeighborhoodType::KNN, k, (ScalarT)0.0));
            return normals;
        }

        inline VectorSet<ScalarT,1> estimateCurvatureKNN(size_t k) const {
            VectorSet<ScalarT,EigenDim> normals;
            VectorSet<ScalarT,1> curvatures;
            compute_<NeighborhoodType::KNN>(normals,
                                            curvatures,
                                            NeighborhoodSpecification<ScalarT>(NeighborhoodType::KNN, k, (ScalarT)0.0));
            return curvatures;
        }

        inline const NormalEstimation& estimateNormalsAndCurvatureRadius(VectorSet<ScalarT,EigenDim> &normals,
                                                                         VectorSet<ScalarT,1> &curvatures,
                                                                         ScalarT radius) const
        {
            compute_<NeighborhoodType::RADIUS>(normals,
                                               curvatures,
                                               NeighborhoodSpecification<ScalarT>(NeighborhoodType::RADIUS, 0, radius));
            return *this;
        }

        inline VectorSet<ScalarT,EigenDim> estimateNormalsRadius(ScalarT radius) const {
            VectorSet<ScalarT,EigenDim> normals;
            VectorSet<ScalarT,1> curvatures;
            compute_<NeighborhoodType::RADIUS>(normals,
                                               curvatures,
                                               NeighborhoodSpecification<ScalarT>(NeighborhoodType::RADIUS, 0, radius));
            return normals;
        }

        inline VectorSet<ScalarT,1> estimateCurvatureRadius(ScalarT radius) const {
            VectorSet<ScalarT,EigenDim> normals;
            VectorSet<ScalarT,1> curvatures;
            compute_<NeighborhoodType::RADIUS>(normals,
                                               curvatures,
                                               NeighborhoodSpecification<ScalarT>(NeighborhoodType::RADIUS, 0, radius));
            return curvatures;
        }

        inline const NormalEstimation& estimateNormalsAndCurvatureKNNInRadius(VectorSet<ScalarT,EigenDim> &normals,
                                                                              VectorSet<ScalarT,1> &curvatures,
                                                                              size_t k,
                                                                              ScalarT radius) const
        {
            compute_<NeighborhoodType::KNN_IN_RADIUS>(normals,
                                                      curvatures,
                                                      NeighborhoodSpecification<ScalarT>(NeighborhoodType::KNN_IN_RADIUS, k, radius));
            return *this;
        }

        inline VectorSet<ScalarT,EigenDim> estimateNormalsKNNInRadius(size_t k, ScalarT radius) const {
            VectorSet<ScalarT,EigenDim> normals;
            VectorSet<ScalarT,1> curvatures;
            compute_<NeighborhoodType::KNN_IN_RADIUS>(normals,
                                                      curvatures,
                                                      NeighborhoodSpecification<ScalarT>(NeighborhoodType::KNN_IN_RADIUS, k, radius));
            return normals;
        }

        inline VectorSet<ScalarT,1> estimateCurvatureKNNInRadius(size_t k, ScalarT radius) const {
            VectorSet<ScalarT,EigenDim> normals;
            VectorSet<ScalarT,1> curvatures;
            compute_<NeighborhoodType::KNN_IN_RADIUS>(normals,
                                                      curvatures,
                                                      NeighborhoodSpecification<ScalarT>(NeighborhoodType::KNN_IN_RADIUS, k, radius));
            return curvatures;
        }

        template <NeighborhoodType NT>
        inline const NormalEstimation& estimateNormalsAndCurvature(VectorSet<ScalarT,EigenDim> &normals,
                                                                   VectorSet<ScalarT,1> &curvatures,
                                                                   const NeighborhoodSpecification<ScalarT> &nh) const
        {
            compute_<NT>(normals, curvatures, nh);
            return *this;
        }

        template <NeighborhoodType NT>
        inline VectorSet<ScalarT,EigenDim> estimateNormals(const NeighborhoodSpecification<ScalarT> &nh) const {
            VectorSet<ScalarT,EigenDim> normals;
            VectorSet<ScalarT,1> curvatures;
            compute_<NT>(normals, curvatures, nh);
            return normals;
        }

        template <NeighborhoodType NT>
        inline VectorSet<ScalarT,1> estimateCurvature(const NeighborhoodSpecification<ScalarT> &nh) const {
            VectorSet<ScalarT,EigenDim> normals;
            VectorSet<ScalarT,1> curvatures;
            compute_<NT>(normals, curvatures, nh);
            return curvatures;
        }

        inline const NormalEstimation& estimateNormalsAndCurvature(VectorSet<ScalarT,EigenDim> &normals,
                                                                   VectorSet<ScalarT,1> &curvatures,
                                                                   const NeighborhoodSpecification<ScalarT> &nh) const
        {
            compute_nh_(normals, curvatures, nh);
            return *this;
        }

        inline VectorSet<ScalarT,EigenDim> estimateNormals(const NeighborhoodSpecification<ScalarT> &nh) const {
            VectorSet<ScalarT,EigenDim> normals;
            VectorSet<ScalarT,1> curvatures;
            compute_nh_(normals, curvatures, nh);
            return normals;
        }

        inline VectorSet<ScalarT,1> estimateCurvature(const NeighborhoodSpecification<ScalarT> &nh) const {
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

        void compute_nh_(VectorSet<ScalarT,EigenDim> &normals,
                         VectorSet<ScalarT,1> &curvatures,
                         const NeighborhoodSpecification<ScalarT> &nh) const
        {
            switch (nh.type) {
                case NeighborhoodType::KNN:
                    compute_<NeighborhoodType::KNN>(normals, curvatures, nh);
                    break;
                case NeighborhoodType::RADIUS:
                    compute_<NeighborhoodType::RADIUS>(normals, curvatures, nh);
                    break;
                case NeighborhoodType::KNN_IN_RADIUS:
                    compute_<NeighborhoodType::KNN_IN_RADIUS>(normals, curvatures, nh);
                    break;
            }
        }

        template <NeighborhoodType NT>
        void compute_(VectorSet<ScalarT,EigenDim> &normals,
                      VectorSet<ScalarT,1> &curvatures,
                      const NeighborhoodSpecification<ScalarT> &nh) const
        {
            NeighborhoodSpecification<ScalarT> nh_sq(nh);
            nh_sq.radius = nh_sq.radius*nh_sq.radius;

            const size_t dim = points_.rows();
            const size_t num_points = points_.cols();

            normals.resize(dim, num_points);
            curvatures.resize(1, num_points);

            NeighborSet<ScalarT> nn;
#pragma omp parallel for shared (normals) private (nn)
            for (size_t i = 0; i < num_points; i++) {
                kd_tree_ptr_->template search<NT>(points_.col(i), nh_sq, nn);
                if (nn.size() < dim) {
                    normals.col(i).setConstant(std::numeric_limits<ScalarT>::quiet_NaN());
                    curvatures[i] = std::numeric_limits<ScalarT>::quiet_NaN();
                    continue;
                }
                VectorSet<ScalarT,EigenDim> neighborhood(dim, nn.size());
                for (size_t j = 0; j < nn.size(); j++) {
                    neighborhood.col(j) = points_.col(nn[j].index);
                }
                PrincipalComponentAnalysis<ScalarT,EigenDim> pca(neighborhood);
                normals.col(i) = pca.getEigenVectors().col(dim-1);
                if (normals.col(i).dot(view_point_ - points_.col(i)) < (ScalarT)0.0) {
                    normals.col(i) *= (ScalarT)(-1.0);
                }
                curvatures[i] = pca.getEigenValues()[dim-1]/pca.getEigenValues().sum();
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
