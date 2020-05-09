#pragma once

#include <cilantro/model_estimation/ransac_base.hpp>
#include <cilantro/core/principal_component_analysis.hpp>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim, typename IndexT = size_t>
    class HyperplaneRANSACEstimator : public RandomSampleConsensusBase<HyperplaneRANSACEstimator<ScalarT,EigenDim>,Eigen::Hyperplane<ScalarT,EigenDim>,ScalarT,IndexT> {
        typedef RandomSampleConsensusBase<HyperplaneRANSACEstimator<ScalarT,EigenDim>,Eigen::Hyperplane<ScalarT,EigenDim>,ScalarT,IndexT> Base;
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        HyperplaneRANSACEstimator(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points)
                : Base(points.rows(), points.cols()/2 + points.cols()%2, 100, 0.1, true),
                  points_(points)
        {}

        inline HyperplaneRANSACEstimator& estimateModel(Eigen::Hyperplane<ScalarT,EigenDim> &model_params) {
            estimate_params_(model_params);
            return *this;
        }

        inline Eigen::Hyperplane<ScalarT,EigenDim> estimateModel() {
            Eigen::Hyperplane<ScalarT,EigenDim> model_params;
            estimate_params_(model_params);
            return model_params;
        }

        inline HyperplaneRANSACEstimator& estimateModel(const typename Base::IndexVector &sample_ind,
                                                        Eigen::Hyperplane<ScalarT,EigenDim> &model_params)
        {
            estimate_params_(sample_ind, model_params);
            return *this;
        }

        inline Eigen::Hyperplane<ScalarT,EigenDim> estimateModel(const typename Base::IndexVector &sample_ind) {
            Eigen::Hyperplane<ScalarT,EigenDim> model_params;
            estimate_params_(sample_ind, model_params);
            return model_params;
        }

        inline HyperplaneRANSACEstimator& computeResiduals(const Eigen::Hyperplane<ScalarT,EigenDim> &model_params,
                                                           typename Base::ResidualVector &residuals)
        {
            residuals.resize(points_.cols());
            for (size_t i = 0; i < points_.cols(); i++) {
                residuals[i] = model_params.absDistance(points_.col(i));
            }
            return *this;
        }

        inline typename Base::ResidualVector computeResiduals(const Eigen::Hyperplane<ScalarT,EigenDim> &model_params) {
            typename Base::ResidualVector residuals;
            computeResiduals(model_params, residuals);
            return residuals;
        }

        inline size_t getDataPointsCount() const { return points_.cols(); }

    private:
        ConstVectorSetMatrixMap<ScalarT,EigenDim> points_;

        inline void estimate_params_(Eigen::Hyperplane<ScalarT,EigenDim> &model_params) {
            if (EigenDim == Eigen::Dynamic) model_params.coeffs().resize(points_.rows());
            PrincipalComponentAnalysis<ScalarT,EigenDim> pca(points_);
            model_params.normal() = pca.getEigenVectors().col(points_.rows() - 1);
            model_params.offset() = -model_params.normal().dot(pca.getDataMean());
        }

        inline void estimate_params_(const typename Base::IndexVector &sample_ind,
                                     Eigen::Hyperplane<ScalarT,EigenDim> &model_params)
        {
            if (EigenDim == Eigen::Dynamic) model_params.coeffs().resize(points_.rows());
            PrincipalComponentAnalysis<ScalarT,EigenDim> pca(points_, sample_ind);
            model_params.normal() = pca.getEigenVectors().col(points_.rows() - 1);
            model_params.offset() = -model_params.normal().dot(pca.getDataMean());
        }
    };

    template <typename IndexT = size_t>
    using HyperplaneRANSACEstimator2f = HyperplaneRANSACEstimator<float,2,IndexT>;

    template <typename IndexT = size_t>
    using LineRANSACEstimator2f = HyperplaneRANSACEstimator<float,2,IndexT>;

    template <typename IndexT = size_t>
    using HyperplaneRANSACEstimator2d = HyperplaneRANSACEstimator<double,2,IndexT>;

    template <typename IndexT = size_t>
    using LineRANSACEstimator2d = HyperplaneRANSACEstimator<double,2,IndexT>;

    template <typename IndexT = size_t>
    using HyperplaneRANSACEstimator3f = HyperplaneRANSACEstimator<float,3,IndexT>;

    template <typename IndexT = size_t>
    using PlaneRANSACEstimator3f = HyperplaneRANSACEstimator<float,3,IndexT>;

    template <typename IndexT = size_t>
    using HyperplaneRANSACEstimator3d = HyperplaneRANSACEstimator<double,3,IndexT>;

    template <typename IndexT = size_t>
    using PlaneRANSACEstimator3d = HyperplaneRANSACEstimator<double,3,IndexT>;

    template <typename IndexT = size_t>
    using HyperplaneRANSACEstimatorXf = HyperplaneRANSACEstimator<float,Eigen::Dynamic,IndexT>;

    template <typename IndexT = size_t>
    using HyperplaneRANSACEstimatorXd = HyperplaneRANSACEstimator<double,Eigen::Dynamic,IndexT>;
}
