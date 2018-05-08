#pragma once

#include <cilantro/ransac_base.hpp>
#include <cilantro/principal_component_analysis.hpp>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim>
    class PlaneEstimator : public RandomSampleConsensusBase<PlaneEstimator<ScalarT,EigenDim>,HomogeneousVector<ScalarT,EigenDim>,ScalarT> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        PlaneEstimator(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points)
                : RandomSampleConsensusBase<PlaneEstimator<ScalarT,EigenDim>,HomogeneousVector<ScalarT,EigenDim>,ScalarT>(points.rows(), points.cols()/2 + points.cols()%2, 100, 0.1, true),
                  points_(points)
        {}

        inline PlaneEstimator& fitModelParameters(HomogeneousVector<ScalarT,EigenDim> &model_params) {
            estimate_params_(points_, model_params);
            return *this;
        }

        inline HomogeneousVector<ScalarT,EigenDim> fitModelParameters() {
            HomogeneousVector<ScalarT,EigenDim> model_params;
            fitModelParameters(model_params);
            return model_params;
        }

        PlaneEstimator& fitModelParameters(const std::vector<size_t> &sample_ind,
                                           HomogeneousVector<ScalarT,EigenDim> &model_params)
        {
            VectorSet<ScalarT,EigenDim> points(points_.rows(), sample_ind.size());
            for (size_t i = 0; i < sample_ind.size(); i++) {
                points.col(i) = points_.col(sample_ind[i]);
            }
            estimate_params_(points, model_params);
            return *this;
        }

        inline HomogeneousVector<ScalarT,EigenDim> fitModelParameters(const std::vector<size_t> &sample_ind) {
            HomogeneousVector<ScalarT,EigenDim> model_params;
            fitModelParameters(sample_ind, model_params);
            return model_params;
        }

        inline PlaneEstimator& computeResiduals(const HomogeneousVector<ScalarT,EigenDim> &model_params,
                                                std::vector<ScalarT> &residuals)
        {
            residuals.resize(points_.cols());
            const ScalarT norm_inv = (ScalarT)(1.0)/model_params.head(points_.rows()).norm();
            const Eigen::Matrix<ScalarT,1,EigenDim> n = norm_inv*model_params.head(points_.rows());
            const ScalarT offset = norm_inv*model_params[points_.rows()];
            for (size_t i = 0; i < points_.cols(); i++) {
                residuals[i] = std::abs(n.dot(points_.col(i)) + offset);
            }
            return *this;
        }

        inline std::vector<ScalarT> computeResiduals(const HomogeneousVector<ScalarT,EigenDim> &model_params) {
            std::vector<ScalarT> residuals;
            computeResiduals(model_params, residuals);
            return residuals;
        }

        inline size_t getDataPointsCount() const { return points_.cols(); }

    private:
        ConstVectorSetMatrixMap<ScalarT,EigenDim> points_;

        void estimate_params_(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points,
                              HomogeneousVector<ScalarT,EigenDim> &model_params)
        {
            PrincipalComponentAnalysis<ScalarT,EigenDim> pca(points);
            const auto& normal = pca.getEigenVectors().col(points_.rows()-1);
            model_params.head(points_.rows()) = normal;
            model_params[points_.rows()] = -normal.dot(pca.getDataMean());
        }
    };

    typedef PlaneEstimator<float,2> PlaneEstimator2f;
    typedef PlaneEstimator<double,2> PlaneEstimator2d;
    typedef PlaneEstimator<float,3> PlaneEstimator3f;
    typedef PlaneEstimator<double,3> PlaneEstimator3d;
    typedef PlaneEstimator<float,Eigen::Dynamic> PlaneEstimatorXf;
    typedef PlaneEstimator<double,Eigen::Dynamic> PlaneEstimatorXd;
}
