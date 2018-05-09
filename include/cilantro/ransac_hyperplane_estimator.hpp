#pragma once

#include <cilantro/ransac_base.hpp>
#include <cilantro/principal_component_analysis.hpp>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim>
    class HyperplaneRANSACEstimator : public RandomSampleConsensusBase<HyperplaneRANSACEstimator<ScalarT,EigenDim>,HomogeneousVector<ScalarT,EigenDim>,ScalarT> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        HyperplaneRANSACEstimator(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points)
                : RandomSampleConsensusBase<HyperplaneRANSACEstimator<ScalarT,EigenDim>,HomogeneousVector<ScalarT,EigenDim>,ScalarT>(points.rows(), points.cols()/2 + points.cols()%2, 100, 0.1, true),
                  points_(points)
        {}

        inline HyperplaneRANSACEstimator& fitModelParameters(HomogeneousVector<ScalarT,EigenDim> &model_params) {
            estimate_params_(points_, model_params);
            return *this;
        }

        inline HomogeneousVector<ScalarT,EigenDim> fitModelParameters() {
            HomogeneousVector<ScalarT,EigenDim> model_params;
            fitModelParameters(model_params);
            return model_params;
        }

        HyperplaneRANSACEstimator& fitModelParameters(const std::vector<size_t> &sample_ind,
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

        inline HyperplaneRANSACEstimator& computeResiduals(const HomogeneousVector<ScalarT,EigenDim> &model_params,
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
