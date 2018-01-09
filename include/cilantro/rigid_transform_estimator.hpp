#pragma once

#include <cilantro/random_sample_consensus.hpp>
#include <cilantro/data_containers.hpp>

namespace cilantro {
    struct RigidTransformParameters {
        Eigen::Matrix3f rotation;
        Eigen::Vector3f translation;
    };

    class RigidTransformEstimator : public RandomSampleConsensus<RigidTransformEstimator,RigidTransformParameters,float> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        RigidTransformEstimator(const ConstPointSetMatrixMap<float,3> &dst_points,
                                const ConstPointSetMatrixMap<float,3> &src_points);

        RigidTransformEstimator(const ConstPointSetMatrixMap<float,3> &dst_points,
                                const ConstPointSetMatrixMap<float,3> &src_points,
                                const std::vector<size_t> &dst_ind,
                                const std::vector<size_t> &src_ind);

        RigidTransformEstimator& estimateModelParameters(RigidTransformParameters &model_params);

        RigidTransformParameters estimateModelParameters();

        RigidTransformEstimator& estimateModelParameters(const std::vector<size_t> &sample_ind, RigidTransformParameters &model_params);

        RigidTransformParameters estimateModelParameters(const std::vector<size_t> &sample_ind);

        RigidTransformEstimator& computeResiduals(const RigidTransformParameters &model_params, std::vector<float> &residuals);

        std::vector<float> computeResiduals(const RigidTransformParameters &model_params);

        inline size_t getDataPointsCount() const { return dst_points_.cols(); }

    private:
        PointSet<float,3> dst_points_tmp_;
        PointSet<float,3> src_points_tmp_;
        ConstPointSetMatrixMap<float,3> dst_points_;
        ConstPointSetMatrixMap<float,3> src_points_;
    };
}
