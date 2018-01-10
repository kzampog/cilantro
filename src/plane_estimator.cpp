#include <cilantro/plane_estimator.hpp>
#include <cilantro/principal_component_analysis.hpp>

namespace cilantro {
    PlaneEstimator::PlaneEstimator(const ConstVectorSetMatrixMap<float,3> &points)
            : RandomSampleConsensus(3, points.cols()/2 + points.cols()%2, 100, 0.1, true),
              points_(points)
    {}

    PlaneEstimator& PlaneEstimator::estimateModelParameters(HomogeneousVector<float,3> &model_params) {
        estimate_params_(points_, model_params);
        return *this;
    }

    HomogeneousVector<float,3> PlaneEstimator::estimateModelParameters() {
        HomogeneousVector<float,3> model_params;
        estimateModelParameters(model_params);
        return model_params;
    }

    PlaneEstimator& PlaneEstimator::estimateModelParameters(const std::vector<size_t> &sample_ind, HomogeneousVector<float,3> &model_params) {
        VectorSet<float,3> points(3, sample_ind.size());
        for (size_t i = 0; i < sample_ind.size(); i++) {
            points.col(i) = points_.col(sample_ind[i]);
        }
        estimate_params_(points, model_params);
        return *this;
    }

    HomogeneousVector<float,3> PlaneEstimator::estimateModelParameters(const std::vector<size_t> &sample_ind) {
        HomogeneousVector<float,3> model_params;
        estimateModelParameters(sample_ind, model_params);
        return model_params;
    }

    PlaneEstimator& PlaneEstimator::computeResiduals(const HomogeneousVector<float,3> &model_params, std::vector<float> &residuals) {
        residuals.resize(points_.cols());
        Eigen::Matrix<float,1,3> n_t = model_params.head(3).transpose();
        float norm = n_t.norm();
        Eigen::Map<Eigen::Matrix<float,1,Eigen::Dynamic> >(residuals.data(),1,residuals.size()) = ((n_t*points_).array() + model_params[3]).cwiseAbs()/norm;
        return *this;
    }

    std::vector<float> PlaneEstimator::computeResiduals(const HomogeneousVector<float,3> &model_params) {
        std::vector<float> residuals;
        computeResiduals(model_params, residuals);
        return residuals;
    }

    void PlaneEstimator::estimate_params_(const ConstVectorSetMatrixMap<float,3> &points, HomogeneousVector<float,3> &model_params) {
        PrincipalComponentAnalysis3D pca(points);
        const Eigen::Vector3f& normal = pca.getEigenVectors().col(2);
        model_params.head(3) = normal;
        model_params[3] = -normal.dot(pca.getDataMean());
    }
}
