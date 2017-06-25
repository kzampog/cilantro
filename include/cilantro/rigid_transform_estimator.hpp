#pragma once

#include <cilantro/random_sample_consensus.hpp>
#include <cilantro/point_cloud.hpp>

struct RigidTransformParameters {
    Eigen::Matrix3f rotation;
    Eigen::Vector3f translation;
};

class RigidTransformEstimator : public RandomSampleConsensus<RigidTransformEstimator,RigidTransformParameters,float> {
public:
    RigidTransformEstimator(const std::vector<Eigen::Vector3f> &dst_points, const std::vector<Eigen::Vector3f> &src_points);
    RigidTransformEstimator(const std::vector<Eigen::Vector3f> &dst_points, const std::vector<Eigen::Vector3f> &src_points, const std::vector<size_t> &dst_ind, const std::vector<size_t> &src_ind);
    RigidTransformEstimator(const PointCloud &dst, const PointCloud &src);
    RigidTransformEstimator(const PointCloud &dst, const PointCloud &src, const std::vector<size_t> &dst_ind, const std::vector<size_t> &src_ind);

    RigidTransformEstimator& estimateModelParameters(RigidTransformParameters &model_params);
    RigidTransformParameters estimateModelParameters();

    RigidTransformEstimator& estimateModelParameters(const std::vector<size_t> &sample_ind, RigidTransformParameters &model_params);
    RigidTransformParameters estimateModelParameters(const std::vector<size_t> &sample_ind);

    RigidTransformEstimator& computeResiduals(const RigidTransformParameters &model_params, std::vector<float> &residuals);
    std::vector<float> computeResiduals(const RigidTransformParameters &model_params);

    inline size_t getDataPointsCount() const { return dst_points_->size(); }

private:
    std::vector<Eigen::Vector3f> dst_points_tmp_;
    std::vector<Eigen::Vector3f> src_points_tmp_;
    const std::vector<Eigen::Vector3f> *dst_points_;
    const std::vector<Eigen::Vector3f> *src_points_;
};
