#pragma once

#include <cilantro/random_sample_consensus.hpp>
#include <cilantro/point_cloud.hpp>

typedef Eigen::Vector4f PlaneParameters;

class PlaneEstimator : public RandomSampleConsensus<PlaneEstimator,PlaneParameters,float> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    PlaneEstimator(const std::vector<Eigen::Vector3f> &points);
    PlaneEstimator(const PointCloud &cloud);

    PlaneEstimator& estimateModelParameters(PlaneParameters &model_params);
    PlaneParameters estimateModelParameters();

    PlaneEstimator& estimateModelParameters(const std::vector<size_t> &sample_ind, PlaneParameters &model_params);
    PlaneParameters estimateModelParameters(const std::vector<size_t> &sample_ind);

    PlaneEstimator& computeResiduals(const PlaneParameters &model_params, std::vector<float> &residuals);
    std::vector<float> computeResiduals(const PlaneParameters &model_params);

    inline size_t getDataPointsCount() const { return points_->size(); }

private:
    const std::vector<Eigen::Vector3f> *points_;

    void estimate_params_(const std::vector<Eigen::Vector3f> &points, PlaneParameters &model_params);
};
