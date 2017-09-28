#include <cilantro/plane_estimator.hpp>
#include <cilantro/principal_component_analysis.hpp>

PlaneEstimator::PlaneEstimator(const std::vector<Eigen::Vector3f> &points)
        : RandomSampleConsensus(3, points.size()/2 + points.size()%2, 100, 0.1, true),
          points_(&points)
{}

PlaneEstimator::PlaneEstimator(const PointCloud &cloud)
        : RandomSampleConsensus(3, cloud.size()/2 + cloud.size()%2, 100, 0.1, true),
          points_(&cloud.points)
{}

PlaneEstimator& PlaneEstimator::estimateModelParameters(PlaneParameters &model_params) {
    estimate_params_(*points_, model_params);
    return *this;
}

PlaneParameters PlaneEstimator::estimateModelParameters() {
    PlaneParameters model_params;
    estimateModelParameters(model_params);
    return model_params;
}

PlaneEstimator& PlaneEstimator::estimateModelParameters(const std::vector<size_t> &sample_ind, PlaneParameters &model_params) {
    std::vector<Eigen::Vector3f> points(sample_ind.size());
    for (size_t i = 0; i < sample_ind.size(); i++) {
        points[i] = (*points_)[sample_ind[i]];
    }
    estimate_params_(points, model_params);
    return *this;
}

PlaneParameters PlaneEstimator::estimateModelParameters(const std::vector<size_t> &sample_ind) {
    PlaneParameters model_params;
    estimateModelParameters(sample_ind, model_params);
    return model_params;
}

PlaneEstimator& PlaneEstimator::computeResiduals(const PlaneParameters &model_params, std::vector<float> &residuals) {
    residuals.resize(points_->size());
    Eigen::Matrix<float,1,3> n_t = model_params.head(3).transpose();
    float norm = n_t.norm();
    Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> > pts((float *)points_->data(), 3, points_->size());
    Eigen::Map<Eigen::Matrix<float,1,Eigen::Dynamic> >(residuals.data(),1,residuals.size()) = ((n_t*pts).array() + model_params[3]).cwiseAbs()/norm;
    return *this;
}

std::vector<float> PlaneEstimator::computeResiduals(const PlaneParameters &model_params) {
    std::vector<float> residuals;
    computeResiduals(model_params, residuals);
    return residuals;
}

void PlaneEstimator::estimate_params_(const std::vector<Eigen::Vector3f> &points, PlaneParameters &model_params) {
    PrincipalComponentAnalysis3D pca(points);
    const Eigen::Vector3f& normal = pca.getEigenVectorsMatrix().col(2);
    model_params.head(3) = normal;
    model_params[3] = -normal.dot(pca.getDataMean());
}
