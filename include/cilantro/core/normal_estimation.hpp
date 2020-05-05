#pragma once

#include <cilantro/core/covariance.hpp>
#include <cilantro/core/kd_tree.hpp>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim, typename CovarianceT = Covariance<ScalarT, EigenDim>, typename IndexT = size_t>
    class NormalEstimation {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef ScalarT Scalar;
        typedef IndexT Index;
        typedef CovarianceT Covariance;
        typedef KDTree<ScalarT,EigenDim,KDTreeDistanceAdaptors::L2,IndexT> SearchTree;

        enum { Dimension = EigenDim };

        NormalEstimation(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points, size_t max_leaf_size = 10)
                : points_(points),
                  kd_tree_ptr_(new SearchTree(points, max_leaf_size)),
                  kd_tree_owned_(true),
                  view_point_(Vector<ScalarT,EigenDim>::Constant(points_.rows(), 1, std::numeric_limits<ScalarT>::quiet_NaN())),
                  ref_normals_(NULL)
        {}

        NormalEstimation(const SearchTree &kd_tree)
                : points_(kd_tree.getPointsMatrixMap()),
                  kd_tree_ptr_(&kd_tree),
                  kd_tree_owned_(false),
                  view_point_(Vector<ScalarT,EigenDim>::Constant(points_.rows(), 1, std::numeric_limits<ScalarT>::quiet_NaN())),
                  ref_normals_(NULL)
        {}

        ~NormalEstimation() {
            if (kd_tree_owned_) delete kd_tree_ptr_;
        }

        inline const CovarianceT& covarianceMethod() const { return compute_mean_and_covariance_; }

        inline CovarianceT& covarianceMethod() { return compute_mean_and_covariance_; }

        // Used for rudimentary normal consistency
        // Enforces normals to point towards the given view point
        inline const Vector<ScalarT,EigenDim>& getViewPoint() const { return view_point_; }

        inline NormalEstimation& setViewPoint(const Eigen::Ref<const Vector<ScalarT,EigenDim>> &vp) {
            view_point_ = vp;
            return *this;
        }

        // Used for rudimentary normal consistency
        // Enforces normals to point to the same surface side as the given reference normals
        inline const ConstVectorSetMatrixMap<ScalarT,EigenDim>& getReferenceNormals() const { return ref_normals_; }

        inline NormalEstimation& setReferenceNormals(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &ref_normals) {
            if (ref_normals.cols() == points_.cols()) {
                new (&ref_normals_) ConstVectorSetMatrixMap<ScalarT,EigenDim>(ref_normals);
            }
            return *this;
        }

        template <typename CountT = size_t>
        inline const NormalEstimation& getNormalsAndCurvatureKNN(VectorSet<ScalarT,EigenDim> &normals,
                                                                 VectorSet<ScalarT,1> &curvature,
                                                                 CountT k) const
        {
            normals.resize(points_.rows(), points_.cols());
            curvature.resize(1, points_.cols());
            compute_normals_curvature_(normals, curvature, KNNNeighborhoodSpecification<CountT>(k));
            return *this;
        }

        // External buffers
        template <typename CountT = size_t>
        inline const NormalEstimation& estimateNormalsAndCurvatureKNN(VectorSetMatrixMap<ScalarT,EigenDim> normals,
                                                                      VectorSetMatrixMap<ScalarT,1> curvature,
                                                                      CountT k) const
        {
            compute_normals_curvature_(normals, curvature, KNNNeighborhoodSpecification<CountT>(k));
            return *this;
        }

        template <typename CountT = size_t>
        inline VectorSet<ScalarT,EigenDim> getNormalsKNN(CountT k) const {
            VectorSet<ScalarT,EigenDim> normals(points_.rows(), points_.cols());
            compute_normals_(normals, KNNNeighborhoodSpecification<CountT>(k));
            return normals;
        }

        // External buffer
        template <typename CountT = size_t>
        inline const NormalEstimation& estimateNormalsKNN(VectorSetMatrixMap<ScalarT,EigenDim> normals,
                                                          CountT k) const
        {
            compute_normals_(normals, KNNNeighborhoodSpecification<CountT>(k));
            return *this;
        }

        template <typename CountT = size_t>
        inline VectorSet<ScalarT,1> getCurvatureKNN(CountT k) const {
            VectorSet<ScalarT,1> curvature(1, points_.cols());
            compute_curvature_(curvature, KNNNeighborhoodSpecification<CountT>(k));
            return curvature;
        }

        // External buffer
        template <typename CountT = size_t>
        inline const NormalEstimation& estimateCurvatureKNN(VectorSetMatrixMap<ScalarT,1> curvature,
                                                            CountT k) const
        {
            compute_curvature_(curvature, KNNNeighborhoodSpecification<CountT>(k));
            return *this;
        }

        inline const NormalEstimation& getNormalsAndCurvatureRadius(VectorSet<ScalarT,EigenDim> &normals,
                                                                    VectorSet<ScalarT,1> &curvature,
                                                                    ScalarT radius) const
        {
            normals.resize(points_.rows(), points_.cols());
            curvature.resize(1, points_.cols());
            compute_normals_curvature_(normals, curvature, RadiusNeighborhoodSpecification<ScalarT>(radius*radius));
            return *this;
        }

        // External buffers
        inline const NormalEstimation& estimateNormalsAndCurvatureRadius(VectorSetMatrixMap<ScalarT,EigenDim> normals,
                                                                         VectorSetMatrixMap<ScalarT,1> curvature,
                                                                         ScalarT radius) const
        {
            compute_normals_curvature_(normals, curvature, RadiusNeighborhoodSpecification<ScalarT>(radius*radius));
            return *this;
        }

        inline VectorSet<ScalarT,EigenDim> getNormalsRadius(ScalarT radius) const {
            VectorSet<ScalarT,EigenDim> normals(points_.rows(), points_.cols());
            compute_normals_(normals, RadiusNeighborhoodSpecification<ScalarT>(radius*radius));
            return normals;
        }

        // External buffer
        inline const NormalEstimation& estimateNormalsRadius(VectorSetMatrixMap<ScalarT,EigenDim> normals,
                                                             ScalarT radius) const
        {
            compute_normals_(normals, RadiusNeighborhoodSpecification<ScalarT>(radius*radius));
            return *this;
        }

        inline VectorSet<ScalarT,1> getCurvatureRadius(ScalarT radius) const {
            VectorSet<ScalarT,1> curvature(1, points_.cols());
            compute_curvature_(curvature, RadiusNeighborhoodSpecification<ScalarT>(radius*radius));
            return curvature;
        }

        // External buffer
        inline const NormalEstimation& estimateCurvatureRadius(VectorSetMatrixMap<ScalarT,1> curvature,
                                                               ScalarT radius) const
        {
            compute_curvature_(curvature, RadiusNeighborhoodSpecification<ScalarT>(radius*radius));
            return *this;
        }

        template <typename CountT = size_t>
        inline const NormalEstimation& getNormalsAndCurvatureKNNInRadius(VectorSet<ScalarT,EigenDim> &normals,
                                                                         VectorSet<ScalarT,1> &curvature,
                                                                         CountT k,
                                                                         ScalarT radius) const
        {
            normals.resize(points_.rows(), points_.cols());
            curvature.resize(1, points_.cols());
            compute_normals_curvature_(normals, curvature, KNNInRadiusNeighborhoodSpecification<ScalarT,CountT>(k, radius*radius));
            return *this;
        }

        // External buffers
        template <typename CountT = size_t>
        inline const NormalEstimation& estimateNormalsAndCurvatureKNNInRadius(VectorSetMatrixMap<ScalarT,EigenDim> normals,
                                                                              VectorSetMatrixMap<ScalarT,1> curvature,
                                                                              CountT k,
                                                                              ScalarT radius) const
        {
            compute_normals_curvature_(normals, curvature, KNNInRadiusNeighborhoodSpecification<ScalarT,CountT>(k, radius*radius));
            return *this;
        }

        template <typename CountT = size_t>
        inline VectorSet<ScalarT,EigenDim> getNormalsKNNInRadius(CountT k, ScalarT radius) const {
            VectorSet<ScalarT,EigenDim> normals(points_.rows(), points_.cols());
            compute_normals_(normals, KNNInRadiusNeighborhoodSpecification<ScalarT,CountT>(k, radius*radius));
            return normals;
        }

        // External buffer
        template <typename CountT = size_t>
        inline const NormalEstimation& estimateNormalsKNNInRadius(VectorSetMatrixMap<ScalarT,EigenDim> normals,
                                                                  CountT k, ScalarT radius) const
        {
            compute_normals_(normals, KNNInRadiusNeighborhoodSpecification<ScalarT,CountT>(k, radius*radius));
            return *this;
        }

        template <typename CountT = size_t>
        inline VectorSet<ScalarT,1> getCurvatureKNNInRadius(CountT k, ScalarT radius) const {
            VectorSet<ScalarT,1> curvature(1, points_.cols());
            compute_curvature_(curvature, KNNInRadiusNeighborhoodSpecification<ScalarT,CountT>(k, radius*radius));
            return curvature;
        }

        // External buffer
        template <typename CountT = size_t>
        inline const NormalEstimation& estimateCurvatureKNNInRadius(VectorSetMatrixMap<ScalarT,1> curvature,
                                                                    CountT k, ScalarT radius) const
        {
            compute_curvature_(curvature, KNNInRadiusNeighborhoodSpecification<ScalarT,CountT>(k, radius*radius));
            return *this;
        }

        template <typename NeighborhoodSpecT>
        inline const NormalEstimation& getNormalsAndCurvature(VectorSet<ScalarT,EigenDim> &normals,
                                                              VectorSet<ScalarT,1> &curvature,
                                                              const NeighborhoodSpecT &nh) const
        {
            normals.resize(points_.rows(), points_.cols());
            curvature.resize(1, points_.cols());
            compute_normals_curvature_(normals, curvature, nh);
            return *this;
        }

        // External buffers
        template <typename NeighborhoodSpecT>
        inline const NormalEstimation& estimateNormalsAndCurvature(VectorSetMatrixMap<ScalarT,EigenDim> normals,
                                                                   VectorSetMatrixMap<ScalarT,1> curvature,
                                                                   const NeighborhoodSpecT &nh) const
        {
            compute_normals_curvature_(normals, curvature, nh);
            return *this;
        }

        template <typename NeighborhoodSpecT>
        inline VectorSet<ScalarT,EigenDim> getNormals(const NeighborhoodSpecT &nh) const {
            VectorSet<ScalarT,EigenDim> normals(points_.rows(), points_.cols());
            compute_normals_(normals, nh);
            return normals;
        }

        // External buffer
        template <typename NeighborhoodSpecT>
        inline const NormalEstimation& estimateNormals(VectorSetMatrixMap<ScalarT,EigenDim> normals,
                                                       const NeighborhoodSpecT &nh)
        {
            compute_normals_(normals, nh);
            return *this;
        }

        template <typename NeighborhoodSpecT>
        inline VectorSet<ScalarT,1> getCurvature(const NeighborhoodSpecT &nh) const {
            VectorSet<ScalarT,1> curvature(1, points_.cols());
            compute_curvature_(curvature, nh);
            return curvature;
        }

        // External buffer
        template <typename NeighborhoodSpecT>
        inline const NormalEstimation& estimateCurvature(VectorSetMatrixMap<ScalarT,1> curvature,
                                                         const NeighborhoodSpecT &nh) const
        {
            compute_curvature_(curvature, nh);
            return *this;
        }

    private:
        ConstVectorSetMatrixMap<ScalarT,EigenDim> points_;
        const SearchTree *kd_tree_ptr_;
        bool kd_tree_owned_;
        Vector<ScalarT,EigenDim> view_point_;
        ConstVectorSetMatrixMap<ScalarT,EigenDim> ref_normals_;
        CovarianceT compute_mean_and_covariance_;

        // Normals only, no normal consistency unless view point or reference normals were set
        template <typename NeighborhoodSpecT>
        void compute_normals_(VectorSetMatrixMap<ScalarT,EigenDim> normals,
                              const NeighborhoodSpecT &nh) const
        {
            // Check if we can enforce consistency; reference normals take precedence
            if (ref_normals_.data() == NULL) {
                if (view_point_.allFinite()) {
                    compute_normals_view_point_(normals, nh);
                    return;
                }
            } else {
                compute_normals_reference_normals_(normals, nh);
                return;
            }

            typename SearchTree::NeighborhoodResult nn;
            Vector<ScalarT,EigenDim> mean;
            Eigen::Matrix<ScalarT,EigenDim,EigenDim> cov;
#pragma omp parallel for shared (normals) private (nn, mean, cov)
            for (size_t i = 0; i < points_.cols(); i++) {
                kd_tree_ptr_->search(points_.col(i), nh, nn);
                if (!compute_mean_and_covariance_(points_, nn, mean, cov)) {
                    normals.col(i).setConstant(std::numeric_limits<ScalarT>::quiet_NaN());
                    continue;
                }

                Eigen::SelfAdjointEigenSolver<Eigen::Matrix<ScalarT,EigenDim,EigenDim>> eig(cov);
                normals.col(i) = eig.eigenvectors().col(0);
            }
        }

        // Normals only, normal consistency by view point
        template <typename NeighborhoodSpecT>
        void compute_normals_view_point_(VectorSetMatrixMap<ScalarT,EigenDim> normals,
                                        const NeighborhoodSpecT &nh) const
        {
            typename SearchTree::NeighborhoodResult nn;
            Vector<ScalarT,EigenDim> mean;
            Eigen::Matrix<ScalarT,EigenDim,EigenDim> cov;
#pragma omp parallel for shared (normals) private (nn, mean, cov)
            for (size_t i = 0; i < points_.cols(); i++) {
                kd_tree_ptr_->search(points_.col(i), nh, nn);
                if (!compute_mean_and_covariance_(points_, nn, mean, cov)) {
                    normals.col(i).setConstant(std::numeric_limits<ScalarT>::quiet_NaN());
                    continue;
                }

                Eigen::SelfAdjointEigenSolver<Eigen::Matrix<ScalarT,EigenDim,EigenDim>> eig(cov);
                if (eig.eigenvectors().col(0).dot(view_point_ - points_.col(i)) < (ScalarT)0.0) {
                    normals.col(i) = -eig.eigenvectors().col(0);
                } else {
                    normals.col(i) = eig.eigenvectors().col(0);
                }
            }
        }

        // Normals only, normal consistency by reference normals
        template <typename NeighborhoodSpecT>
        void compute_normals_reference_normals_(VectorSetMatrixMap<ScalarT,EigenDim> normals,
                                                const NeighborhoodSpecT &nh) const
        {
            typename SearchTree::NeighborhoodResult nn;
            Vector<ScalarT,EigenDim> mean;
            Eigen::Matrix<ScalarT,EigenDim,EigenDim> cov;
#pragma omp parallel for shared (normals) private (nn, mean, cov)
            for (size_t i = 0; i < points_.cols(); i++) {
                kd_tree_ptr_->search(points_.col(i), nh, nn);
                if (!compute_mean_and_covariance_(points_, nn, mean, cov)) {
                    normals.col(i).setConstant(std::numeric_limits<ScalarT>::quiet_NaN());
                    // normals.col(i) = ref_normals_.col(i).normalized();
                    continue;
                }

                Eigen::SelfAdjointEigenSolver<Eigen::Matrix<ScalarT,EigenDim,EigenDim>> eig(cov);
                if (eig.eigenvectors().col(0).dot(ref_normals_.col(i)) < (ScalarT)0.0) {
                    normals.col(i) = -eig.eigenvectors().col(0);
                } else {
                    normals.col(i) = eig.eigenvectors().col(0);
                }
            }
        }

        // Normals and curvature, no normal consistency unless view point or reference normals were set
        template <typename NeighborhoodSpecT>
        void compute_normals_curvature_(VectorSetMatrixMap<ScalarT,EigenDim> normals,
                                        VectorSetMatrixMap<ScalarT,1> curvature,
                                        const NeighborhoodSpecT &nh) const
        {
            // Check if we can enforce consistency; reference normals take precedence
            if (ref_normals_.data() == NULL) {
                if (view_point_.allFinite()) {
                    compute_normals_curvature_view_point_(normals, curvature, nh);
                    return;
                }
            } else {
                compute_normals_curvature_reference_normals_(normals, curvature, nh);
                return;
            }

            typename SearchTree::NeighborhoodResult nn;
            Vector<ScalarT,EigenDim> mean;
            Eigen::Matrix<ScalarT,EigenDim,EigenDim> cov;
#pragma omp parallel for shared (normals, curvature) private (nn, mean, cov)
            for (size_t i = 0; i < points_.cols(); i++) {
                kd_tree_ptr_->search(points_.col(i), nh, nn);
                if (!compute_mean_and_covariance_(points_, nn, mean, cov)) {
                    normals.col(i).setConstant(std::numeric_limits<ScalarT>::quiet_NaN());
                    curvature[i] = std::numeric_limits<ScalarT>::quiet_NaN();
                    continue;
                }

                Eigen::SelfAdjointEigenSolver<Eigen::Matrix<ScalarT,EigenDim,EigenDim>> eig(cov);
                normals.col(i) = eig.eigenvectors().col(0);
                curvature[i] = eig.eigenvalues()[0]/eig.eigenvalues().sum();
            }
        }

        // Normals and curvature, normal consistency by view point
        template <typename NeighborhoodSpecT>
        void compute_normals_curvature_view_point_(VectorSetMatrixMap<ScalarT,EigenDim> normals,
                                                   VectorSetMatrixMap<ScalarT,1> curvature,
                                                   const NeighborhoodSpecT &nh) const
        {
            typename SearchTree::NeighborhoodResult nn;
            Vector<ScalarT,EigenDim> mean;
            Eigen::Matrix<ScalarT,EigenDim,EigenDim> cov;
#pragma omp parallel for shared (normals, curvature) private (nn, mean, cov)
            for (size_t i = 0; i < points_.cols(); i++) {
                kd_tree_ptr_->search(points_.col(i), nh, nn);
                if (!compute_mean_and_covariance_(points_, nn, mean, cov)) {
                    normals.col(i).setConstant(std::numeric_limits<ScalarT>::quiet_NaN());
                    curvature[i] = std::numeric_limits<ScalarT>::quiet_NaN();
                    continue;
                }

                Eigen::SelfAdjointEigenSolver<Eigen::Matrix<ScalarT,EigenDim,EigenDim>> eig(cov);
                if (eig.eigenvectors().col(0).dot(view_point_ - points_.col(i)) < (ScalarT)0.0) {
                    normals.col(i) = -eig.eigenvectors().col(0);
                } else {
                    normals.col(i) = eig.eigenvectors().col(0);
                }
                curvature[i] = eig.eigenvalues()[0]/eig.eigenvalues().sum();
            }
        }

        // Normals and curvature, normal consistency by reference normals
        template <typename NeighborhoodSpecT>
        void compute_normals_curvature_reference_normals_(VectorSetMatrixMap<ScalarT,EigenDim> normals,
                                        VectorSetMatrixMap<ScalarT,1> curvature,
                                        const NeighborhoodSpecT &nh) const
        {
            typename SearchTree::NeighborhoodResult nn;
            Vector<ScalarT,EigenDim> mean;
            Eigen::Matrix<ScalarT,EigenDim,EigenDim> cov;
#pragma omp parallel for shared (normals, curvature) private (nn, mean, cov)
            for (size_t i = 0; i < points_.cols(); i++) {
                kd_tree_ptr_->search(points_.col(i), nh, nn);
                if (!compute_mean_and_covariance_(points_, nn, mean, cov)) {
                    normals.col(i).setConstant(std::numeric_limits<ScalarT>::quiet_NaN());
                    curvature[i] = std::numeric_limits<ScalarT>::quiet_NaN();
                    continue;
                }

                Eigen::SelfAdjointEigenSolver<Eigen::Matrix<ScalarT,EigenDim,EigenDim>> eig(cov);
                if (eig.eigenvectors().col(0).dot(ref_normals_.col(i)) < (ScalarT)0.0) {
                    normals.col(i) = -eig.eigenvectors().col(0);
                } else {
                    normals.col(i) = eig.eigenvectors().col(0);
                }
                curvature[i] = eig.eigenvalues()[0]/eig.eigenvalues().sum();
            }
        }

        // Curvature only
        template <typename NeighborhoodSpecT>
        void compute_curvature_(VectorSetMatrixMap<ScalarT,1> curvature,
                                const NeighborhoodSpecT &nh) const
        {
            typename SearchTree::NeighborhoodResult nn;
            Vector<ScalarT,EigenDim> mean;
            Eigen::Matrix<ScalarT,EigenDim,EigenDim> cov;
#pragma omp parallel for shared (curvature) private (nn, mean, cov)
            for (size_t i = 0; i < points_.cols(); i++) {
                kd_tree_ptr_->search(points_.col(i), nh, nn);
                if (!compute_mean_and_covariance_(points_, nn, mean, cov)) {
                    curvature[i] = std::numeric_limits<ScalarT>::quiet_NaN();
                    continue;
                }

                Eigen::SelfAdjointEigenSolver<Eigen::Matrix<ScalarT,EigenDim,EigenDim>> eig(cov, Eigen::EigenvaluesOnly);
                curvature[i] = eig.eigenvalues()[0]/eig.eigenvalues().sum();
            }
        }
    };

    template <typename CovarianceT = Covariance<float,2>, typename IndexT = size_t>
    using NormalEstimation2f = NormalEstimation<float,2,CovarianceT,IndexT>;

    template <typename CovarianceT = Covariance<double,2>, typename IndexT = size_t>
    using NormalEstimation2d = NormalEstimation<double,2,CovarianceT,IndexT>;

    template <typename CovarianceT = Covariance<float,3>, typename IndexT = size_t>
    using NormalEstimation3f = NormalEstimation<float,3,CovarianceT,IndexT>;

    template <typename CovarianceT = Covariance<double,3>, typename IndexT = size_t>
    using NormalEstimation3d = NormalEstimation<double,3,CovarianceT,IndexT>;

    template <typename CovarianceT = Covariance<float,Eigen::Dynamic>, typename IndexT = size_t>
    using NormalEstimationXf = NormalEstimation<float,Eigen::Dynamic,CovarianceT,IndexT>;

    template <typename CovarianceT = Covariance<double,Eigen::Dynamic>, typename IndexT = size_t>
    using NormalEstimationXd = NormalEstimation<double,Eigen::Dynamic,CovarianceT,IndexT>;
}
