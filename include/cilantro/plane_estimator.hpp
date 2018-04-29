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

        inline PlaneEstimator& estimateModelParameters(HomogeneousVector<ScalarT,EigenDim> &model_params) {
            estimate_params_(points_, model_params);
            return *this;
        }

        inline HomogeneousVector<ScalarT,EigenDim> estimateModelParameters() {
            HomogeneousVector<ScalarT,EigenDim> model_params;
            estimateModelParameters(model_params);
            return model_params;
        }

        PlaneEstimator& estimateModelParameters(const std::vector<size_t> &sample_ind, HomogeneousVector<ScalarT,EigenDim> &model_params) {
            VectorSet<ScalarT,EigenDim> points(points_.rows(), sample_ind.size());
            for (size_t i = 0; i < sample_ind.size(); i++) {
                points.col(i) = points_.col(sample_ind[i]);
            }
            estimate_params_(points, model_params);
            return *this;
        }

        inline HomogeneousVector<ScalarT,EigenDim> estimateModelParameters(const std::vector<size_t> &sample_ind) {
            HomogeneousVector<ScalarT,EigenDim> model_params;
            estimateModelParameters(sample_ind, model_params);
            return model_params;
        }

        PlaneEstimator& computeResiduals(const HomogeneousVector<ScalarT,EigenDim> &model_params, std::vector<ScalarT> &residuals) {
            residuals.resize(points_.cols());
            Eigen::Matrix<ScalarT,1,EigenDim> n_t = model_params.head(points_.rows()).transpose();
            ScalarT norm = n_t.norm();
            Eigen::Map<Eigen::Matrix<ScalarT,1,Eigen::Dynamic> >(residuals.data(),1,residuals.size()) = ((n_t*points_).array() + model_params[points_.rows()]).cwiseAbs()/norm;
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

        void estimate_params_(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points, HomogeneousVector<ScalarT,EigenDim> &model_params) {
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
