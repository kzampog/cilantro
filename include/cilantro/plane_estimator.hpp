#pragma once

#include <cilantro/random_sample_consensus.hpp>
#include <cilantro/data_containers.hpp>

namespace cilantro {
    class PlaneEstimator : public RandomSampleConsensus<PlaneEstimator,LinearConstraint<float,3>,float> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        PlaneEstimator(const ConstVectorSetMatrixMap<float,3> &points);

        PlaneEstimator& estimateModelParameters(LinearConstraint<float,3> &model_params);

        LinearConstraint<float,3> estimateModelParameters();

        PlaneEstimator& estimateModelParameters(const std::vector<size_t> &sample_ind, LinearConstraint<float,3> &model_params);

        LinearConstraint<float,3> estimateModelParameters(const std::vector<size_t> &sample_ind);

        PlaneEstimator& computeResiduals(const LinearConstraint<float,3> &model_params, std::vector<float> &residuals);

        std::vector<float> computeResiduals(const LinearConstraint<float,3> &model_params);

        inline size_t getDataPointsCount() const { return points_.cols(); }

    private:
        ConstVectorSetMatrixMap<float,3> points_;

        void estimate_params_(const ConstVectorSetMatrixMap<float,3> &points, LinearConstraint<float,3> &model_params);
    };
}
