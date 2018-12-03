#pragma once

#include <cilantro/ransac_base.hpp>
#include <cilantro/principal_component_analysis.hpp>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim>
    class HyperplaneRANSACEstimator : public RandomSampleConsensusBase<HyperplaneRANSACEstimator<ScalarT,EigenDim>,Eigen::Hyperplane<ScalarT,EigenDim>,ScalarT> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        HyperplaneRANSACEstimator(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points)
                : RandomSampleConsensusBase<HyperplaneRANSACEstimator<ScalarT,EigenDim>,Eigen::Hyperplane<ScalarT,EigenDim>,ScalarT>(points.rows(), points.cols()/2 + points.cols()%2, 100, 0.1, true),
                  points_(points)
        {}

        inline HyperplaneRANSACEstimator& estimateModel(Eigen::Hyperplane<ScalarT,EigenDim> &model_params) {
            estimate_params_(points_, model_params);
            return *this;
        }

        inline Eigen::Hyperplane<ScalarT,EigenDim> estimateModel() {
            Eigen::Hyperplane<ScalarT,EigenDim> model_params;
            estimateModel(model_params);
            return model_params;
        }

        HyperplaneRANSACEstimator& estimateModel(const std::vector<size_t> &sample_ind,
                                                 Eigen::Hyperplane<ScalarT,EigenDim> &model_params)
        {
            VectorSet<ScalarT,EigenDim> points(points_.rows(), sample_ind.size());
            for (size_t i = 0; i < sample_ind.size(); i++) {
                points.col(i) = points_.col(sample_ind[i]);
            }
            estimate_params_(points, model_params);
            return *this;
        }

        inline Eigen::Hyperplane<ScalarT,EigenDim> estimateModel(const std::vector<size_t> &sample_ind) {
            Eigen::Hyperplane<ScalarT,EigenDim> model_params;
            estimateModel(sample_ind, model_params);
            return model_params;
        }

        inline HyperplaneRANSACEstimator& computeResiduals(const Eigen::Hyperplane<ScalarT,EigenDim> &model_params,
                                                           std::vector<ScalarT> &residuals)
        {
            residuals.resize(points_.cols());
            for (size_t i = 0; i < points_.cols(); i++) {
                residuals[i] = model_params.absDistance(points_.col(i));
            }
            return *this;
        }

        inline std::vector<ScalarT> computeResiduals(const Eigen::Hyperplane<ScalarT,EigenDim> &model_params) {
            std::vector<ScalarT> residuals;
            computeResiduals(model_params, residuals);
            return residuals;
        }

        inline size_t getDataPointsCount() const { return points_.cols(); }

    private:
        ConstVectorSetMatrixMap<ScalarT,EigenDim> points_;

        void estimate_params_(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points,
                              Eigen::Hyperplane<ScalarT,EigenDim> &model_params)
        {
            if (EigenDim == Eigen::Dynamic) model_params.coeffs().resize(points.rows());
            PrincipalComponentAnalysis<ScalarT,EigenDim> pca(points);
            model_params.normal() = pca.getEigenVectors().col(points_.rows()-1);
            model_params.offset() = -model_params.normal().dot(pca.getDataMean());
        }
    };

    typedef HyperplaneRANSACEstimator<float,2> HyperplaneRANSACEstimator2f;
    typedef HyperplaneRANSACEstimator<float,2> LineRANSACEstimator2f;
    typedef HyperplaneRANSACEstimator<double,2> HyperplaneRANSACEstimator2d;
    typedef HyperplaneRANSACEstimator<double,2> LineRANSACEstimator2d;
    typedef HyperplaneRANSACEstimator<float,3> HyperplaneRANSACEstimator3f;
    typedef HyperplaneRANSACEstimator<float,3> PlaneRANSACEstimator3f;
    typedef HyperplaneRANSACEstimator<double,3> HyperplaneRANSACEstimator3d;
    typedef HyperplaneRANSACEstimator<double,3> PlaneRANSACEstimator3d;
    typedef HyperplaneRANSACEstimator<float,Eigen::Dynamic> HyperplaneRANSACEstimatorXf;
    typedef HyperplaneRANSACEstimator<double,Eigen::Dynamic> HyperplaneRANSACEstimatorXd;
}
