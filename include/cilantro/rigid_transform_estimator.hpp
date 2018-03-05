#pragma once

#include <cilantro/random_sample_consensus.hpp>
#include <cilantro/data_containers.hpp>
#include <cilantro/rigid_transformation.hpp>

namespace cilantro {
    class RigidTransformEstimator : public RandomSampleConsensus<RigidTransformEstimator,RigidTransformation<float,3>,float> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        RigidTransformEstimator(const ConstVectorSetMatrixMap<float,3> &dst_points,
                                const ConstVectorSetMatrixMap<float,3> &src_points);

        RigidTransformEstimator(const ConstVectorSetMatrixMap<float,3> &dst_points,
                                const ConstVectorSetMatrixMap<float,3> &src_points,
                                const std::vector<size_t> &dst_ind,
                                const std::vector<size_t> &src_ind);

        RigidTransformEstimator& estimateModelParameters(RigidTransformation<float,3> &model_params);

        RigidTransformation<float,3> estimateModelParameters();

        RigidTransformEstimator& estimateModelParameters(const std::vector<size_t> &sample_ind, RigidTransformation<float,3> &model_params);

        RigidTransformation<float,3> estimateModelParameters(const std::vector<size_t> &sample_ind);

        RigidTransformEstimator& computeResiduals(const RigidTransformation<float,3> &model_params, std::vector<float> &residuals);

        std::vector<float> computeResiduals(const RigidTransformation<float,3> &model_params);

        inline size_t getDataPointsCount() const { return dst_points_.cols(); }

    private:
        VectorSet<float,3> dst_points_tmp_;
        VectorSet<float,3> src_points_tmp_;
        ConstVectorSetMatrixMap<float,3> dst_points_;
        ConstVectorSetMatrixMap<float,3> src_points_;
    };
}
